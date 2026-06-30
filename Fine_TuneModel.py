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


MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def finetune_answer_only(
    data_path="instruction-data.json",
    data_url=(
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    ),
    model_choice="gpt2-small (124M)",
    num_epochs=2,
    batch_size=4,
    lr=5e-5,
    context_length=1024,
    save_path="fine_tuned_gpt3/answer_only_model.pt",
    device=None,
):
    """Fine-tune with ANSWER-ONLY loss (loss computed only on the response
    tokens, the prompt/context is masked out). This is the fix for the model
    echoing/continuing the prompt instead of answering.

    Everything is parameterised so you can run it however you want, e.g.:

        from Fine_TuneModel import finetune_answer_only
        finetune_answer_only(num_epochs=3, lr=3e-5,
                             model_choice="gpt2-medium (355M)",
                             save_path="my_qa_model.pt")

    or from the shell:  python Fine_TuneModel.py answer-only

    Returns (model, train_losses, val_losses).
    """
    # masked dataset + collate live in answer_only_loss.py (single source)
    from answer_only_loss import (
        InstructionDataset as AnswerOnlyDataset,
        answer_only_collate_fn,
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[answer-only] Device: {device}")

    if model_choice not in MODEL_CONFIGS:
        raise ValueError(f"model_choice must be one of {list(MODEL_CONFIGS)}")

    config = {
        "vocab_size": 50257,
        "context_length": context_length,
        "drop_rate": 0.1,
        "qkv_bias": True,
    }
    config.update(MODEL_CONFIGS[model_choice])
    model_size = model_choice.split(" ")[-1].strip("()")

    # --- data ---
    data = download_and_load_file(data_path, data_url)
    print(f"[answer-only] Examples: {len(data)}")

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.10)
    train_data = data[:train_portion]
    val_data = data[train_portion + test_portion:] or data[-1:]

    torch.manual_seed(123)
    collate = partial(
        answer_only_collate_fn, device=device, allowed_max_length=context_length
    )
    train_loader = DataLoader(
        AnswerOnlyDataset(train_data, tokenizer),
        batch_size=batch_size, collate_fn=collate,
        shuffle=True, drop_last=True, num_workers=0,
    )
    val_loader = DataLoader(
        AnswerOnlyDataset(val_data, tokenizer),
        batch_size=batch_size, collate_fn=collate,
        shuffle=False, drop_last=False, num_workers=0,
    )
    if len(train_loader) == 0:
        raise ValueError("Not enough data for one batch — lower batch_size.")

    # --- pretrained GPT-2 weights ---
    print(f"[answer-only] Loading GPT-2 weights: {model_size}")
    _, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
    model = GPTModel(config)
    load_weights_into_gpt(model, params)
    model.to(device)

    with torch.no_grad():
        tl = calc_loss_loader(train_loader, model, device, num_batches=5)
        vl = calc_loss_loader(val_loader, model, device, num_batches=5)
    print(f"[answer-only] Initial loss — train {tl:.4f}  val {vl:.4f}")

    # --- train (answer-only loss comes from the masked targets) ---
    start_time = time.time()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    train_losses, val_losses, _ = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer,
    )
    print(f"[answer-only] Trained in {(time.time() - start_time) / 60:.2f} min")

    # --- save ---
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(
            {"model_state_dict": model.state_dict(), "model_config": config},
            save_path,
        )
        print(f"[answer-only] Saved to {save_path}")

    return model, train_losses, val_losses


def main():
    print("=== STARTING MAIN FUNCTION ===")
    
    # Device selection
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()
        else:
            device = torch.device("cpu")
            print("CUDA not available, using CPU")
            torch.set_num_threads(8)
    except Exception as e:
        print(f"Error detecting device: {e}")
        device = torch.device("cpu")
        print("Falling back to CPU")
    
    print(f"Device: {device}")
    
    # Model configuration
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
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
    
    # Check if fine-tuned model exists
    model_save_dir = "fine_tuned_gpt3"
    model_path = os.path.join(model_save_dir, 'fine_tuned_gpt2/model.pt')
    
    if os.path.exists(model_path):
        print(f"=== FINE-TUNED MODEL FOUND AT {model_path} ===")
        response = input("Load fine-tuned model? (y/n): ")
        if response.lower() == 'y':
            try:
                # Load the saved model
                checkpoint = torch.load(model_path, map_location=device)
                model = GPTModel(BASE_CONFIG)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
                model.eval()
                print("=== FINE-TUNED MODEL LOADED SUCCESSFULLY ===")
                
                # Test the loaded model
                test_input = torch.randint(0, 50257, (1, 10)).to(device)
                with torch.no_grad():
                    test_output = model(test_input)
                print(f"=== MODEL TEST PASSED, output shape: {test_output.shape} ===")
                
                # You can now use the model for inference
                print("=== MODEL READY FOR INFERENCE ===")
                return
            except Exception as e:
                print(f"Error loading fine-tuned model: {e}")
                print("Training from scratch...")
    
    # If no fine-tuned model found or user chose to train
    print("=== TRAINING NEW MODEL ===")
    
    # Data loading
    file_path = "alpaxa_data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )
    
    try:
        data = download_and_load_file(file_path, url)
        print("Number of entries:", len(data))
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
    
    # Create dataloaders
    num_workers = 0
    batch_size = 4
    context_length = 1024
    
    torch.manual_seed(123)
    customized_collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=context_length
    )
    
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
    
    print("=== DATALOADERS CREATED ===")
    
    # Load pretrained model
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    print(f"=== LOADING PRETRAINED MODEL: {model_size} ===")
    
    try:
        settings, params = download_and_load_gpt2(
            model_size=model_size, 
            models_dir="gpt2"
        )
        print("=== MODEL PARAMS LOADED ===")
    except Exception as e:
        print(f"Error loading model params: {e}")
        return
    
    # Create and load weights
    try:
        model = GPTModel(BASE_CONFIG)
        print("=== MODEL CREATED ===")
        load_weights_into_gpt(model, params)
        print("=== WEIGHTS LOADED ===")
    except Exception as e:
        print(f"Error creating/loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Move model to device
    model.to(device)
    print(f"=== MODEL MOVED TO {device} ===")
    
    # Calculate initial loss
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
    
    # Training
    print("=== STARTING TRAINING ===")
    model.train()
    
    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.00005, weight_decay=0.1
    )
    num_epochs = 1
    
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )
    
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")
    
    # SAVE THE MODEL!
    print("=== SAVING FINE-TUNED MODEL ===")
    os.makedirs(model_save_dir, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': BASE_CONFIG,
        'vocab_size': 50257,
        'embed_dim': BASE_CONFIG['emb_dim'],
        'n_layers': BASE_CONFIG['n_layers'],
        'n_heads': BASE_CONFIG['n_heads'],
        'context_length': BASE_CONFIG['context_length'],
    }, model_path)
    
    print(f"=== MODEL SAVED TO {model_path} ===")
    print("=== MAIN FUNCTION COMPLETED ===")

if __name__ == "__main__":
    import sys
    print("=== SCRIPT STARTED ===")
    try:
        # `python Fine_TuneModel.py answer-only`  -> answer-only loss fine-tune
        # `python Fine_TuneModel.py`              -> original padding-only flow
        if len(sys.argv) > 1 and sys.argv[1] in ("answer-only", "answer_only"):
            finetune_answer_only()
        else:
            main()
    except Exception as e:
        print(f"=== UNHANDLED ERROR: {e} ===")
        import traceback
        traceback.print_exc()



    