import json
import os
import time
from functools import partial
import urllib.request
from gpt_download import download_and_load_gpt2
from GPT_model import GPTModel
import numpy as np
from TrainLLM import calc_loss_loader, train_model_simple
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

# Print PyTorch info at startup
print("=== PYTORCH INFO ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    # Clear any cached memory
    torch.cuda.empty_cache()
    print(f"Initial allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

tokenizer = tiktoken.get_encoding("gpt2")

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] * 
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text

def main():
    print("=== STARTING MAIN FUNCTION ===")
    
    # Device selection with fallback
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()
        else:
            device = torch.device("cpu")
            print("CUDA not available, using CPU")
    except Exception as e:
        print(f"Error detecting device: {e}")
        device = torch.device("cpu")
        print("Falling back to CPU")
    
    print(f"Device: {device}")
    
    # Data loading
    file_path = "instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )

    try:
        data = download_and_load_file(file_path, url)
        print("Number of entries:", len(data))
        print("Example entry:\n", data[50])
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Split data
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    val_portion = len(data) - train_portion - test_portion

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print("=== DATASETS CREATED ===")

    # Use context_length = 1024 (must match pretrained model)
    num_workers = 0
    batch_size = 4  # Reduced for memory
    context_length = 1024  # MUST BE 1024 for pretrained GPT-2
    
    torch.manual_seed(123)
    customized_collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=context_length  # This will truncate sequences if needed
    )

    # Create dataloaders
    try:
        train_dataset = InstructionDataset(train_data, tokenizer)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=customized_collate_fn,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers
        )

        val_dataset = InstructionDataset(val_data, tokenizer)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=customized_collate_fn,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers
        )

        test_dataset = InstructionDataset(test_data, tokenizer)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=customized_collate_fn,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers
        )
        print("=== DATALOADERS CREATED ===")
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        import traceback
        traceback.print_exc()
        return

    # Model configuration - MUST match pretrained model
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,  # MUST be 1024 to match pretrained weights
        "drop_rate": 0.1,
        "qkv_bias": True
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    CHOOSE_MODEL = "gpt2-small (124M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    print(f"=== LOADING MODEL: {model_size} ===")
    print(f"Context length: {BASE_CONFIG['context_length']}")
    
    try:
        settings, params = download_and_load_gpt2(
            model_size=model_size, 
            models_dir="gpt2"
        )
        print("=== MODEL PARAMS LOADED ===")
    except Exception as e:
        print(f"Error loading model params: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create and load model
    try:
        model = GPTModel(BASE_CONFIG)
        print("=== MODEL CREATED ===")
        print(f"Model context length: {model.pos_emb.weight.shape[0]}")
        
        load_weights_into_gpt(model, params)
        print("=== WEIGHTS LOADED ===")
    except Exception as e:
        print(f"Error creating/loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Move model to device
    try:
        print("=== ATTEMPTING model.eval() ===")
        model.eval()
        print("=== model.eval() SUCCESSFUL ===")
        
        print(f"=== ATTEMPTING model.to({device}) ===")
        model.to(device)
        print("=== model.to(device) SUCCESSFUL ===")
        
        # Test forward pass
        print("=== TESTING FORWARD PASS ===")
        test_input = torch.randint(0, 50257, (1, 10)).to(device)
        with torch.no_grad():
            test_output = model(test_input)
        print(f"=== FORWARD PASS SUCCESSFUL, output shape: {test_output.shape} ===")
        
        if device.type == "cuda":
            print(f"GPU memory after model load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            
    except Exception as e:
        print(f"=== ERROR moving model to device: {e} ===")
        import traceback
        traceback.print_exc()
        
        if device.type == "cuda":
            print("=== TRYING CPU INSTEAD ===")
            try:
                device = torch.device("cpu")
                model.to(device)
                print("=== MODEL MOVED TO CPU SUCCESSFULLY ===")
            except Exception as e2:
                print(f"=== ERROR moving to CPU: {e2} ===")
                return
        else:
            return

    # Calculate initial loss
    try:
        with torch.no_grad():
            print("=== CALCULATING INITIAL LOSS ===")
            train_loss = calc_loss_loader(
                train_loader, model, device, num_batches=5
            )
            val_loss = calc_loss_loader(
                val_loader, model, device, num_batches=5
            )
            print(f"Initial train loss: {train_loss:.4f}")
            print(f"Initial val loss: {val_loss:.4f}")
    except Exception as e:
        print(f"Error calculating initial loss: {e}")
        import traceback
        traceback.print_exc()
        return

    # Switch to training mode
    model.train()
    print("=== STARTING TRAINING ===")

    # Training
    try:
        start_time = time.time()
        torch.manual_seed(123)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.00005, weight_decay=0.1
        )
        num_epochs = 2

        train_losses, val_losses, tokens_seen = train_model_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=num_epochs, eval_freq=5, eval_iter=5,
            start_context=format_input(val_data[0]), tokenizer=tokenizer
        )

        end_time = time.time()
        execution_time_minutes = (end_time - start_time) / 60
        print(f"Training completed in {execution_time_minutes:.2f} minutes.")
        print("=== MAIN FUNCTION COMPLETED ===")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    print("=== SCRIPT STARTED ===")
    try:
        main()
    except Exception as e:
        print(f"=== UNHANDLED ERROR: {e} ===")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=== SCRIPT STARTED ===")
    try:
        main()
    except Exception as e:
        print(f"=== UNHANDLED ERROR: {e} ===")
        import traceback
        traceback.print_exc()

    