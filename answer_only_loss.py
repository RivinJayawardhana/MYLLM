"""
Answer-only loss for instruction / Q&A fine-tuning.

Problem this solves
-------------------
The default collate in Fine_TuneModel.py only masks *padding*. So the model
spends most of its loss learning to reproduce the instruction + context, and
relatively little on the actual answer. The result is a model that echoes the
prompt and "continues" text instead of answering.

This module computes loss ONLY on the response tokens by setting every token
before "### Response:" to ignore_index (-100), which torch's cross-entropy
skips. Train with this and the model learns: given a prompt, produce the answer.

What's here
-----------
- format_input            : same Alpaca-style prompt as the rest of the project
- InstructionDataset      : also records where the response starts (prompt_len)
- answer_only_collate_fn  : pads AND masks the prompt region in the targets
- main()                  : end-to-end fine-tune using answer-only loss + save

Use the collate function as a drop-in replacement:

    from functools import partial
    from answer_only_loss import InstructionDataset, answer_only_collate_fn

    collate = partial(answer_only_collate_fn, device=device, allowed_max_length=1024)
    loader = DataLoader(InstructionDataset(data, tokenizer),
                        batch_size=4, collate_fn=collate, shuffle=True)

Everything else (model, training loop, loss) is unchanged — masking the targets
is all that's needed.
"""

import os
import json
import time
import urllib.request
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

from GPT_model import GPTModel
from TrainLLM import calc_loss_loader, train_model_simple

PAD_TOKEN_ID = 50256
IGNORE_INDEX = -100
RESPONSE_HEADER = "\n\n### Response:\n"


# --------------------------------------------------------------------------- #
# Prompt formatting
# --------------------------------------------------------------------------- #

def format_input(entry):
    """Instruction (+ optional input) part of the prompt — no response."""
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
    return instruction_text + input_text


# --------------------------------------------------------------------------- #
# Dataset: records the prompt length so the collate fn knows what to mask
# --------------------------------------------------------------------------- #

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        self.prompt_lengths = []
        for entry in data:
            prompt = format_input(entry)
            # response header is part of the prompt the model conditions on,
            # so the *answer* the model must learn starts right after it.
            prompt_with_header = prompt + RESPONSE_HEADER
            full_text = prompt_with_header + entry["output"]

            self.encoded_texts.append(tokenizer.encode(full_text))
            self.prompt_lengths.append(len(tokenizer.encode(prompt_with_header)))

    def __getitem__(self, index):
        return self.encoded_texts[index], self.prompt_lengths[index]

    def __len__(self):
        return len(self.data)


# --------------------------------------------------------------------------- #
# Collate: pad, then mask everything before the answer (and the padding)
# --------------------------------------------------------------------------- #

def answer_only_collate_fn(
    batch,
    pad_token_id=PAD_TOKEN_ID,
    ignore_index=IGNORE_INDEX,
    allowed_max_length=None,
    device="cpu",
):
    batch_max_length = max(len(item) + 1 for item, _ in batch)
    inputs_lst, targets_lst = [], []

    for item, prompt_len in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        # 1) mask the prompt region of the targets.
        # targets are shifted by one, so the first answer target sits at
        # index (prompt_len - 1); mask everything before it.
        mask_until = max(prompt_len - 1, 0)
        targets[:mask_until] = ignore_index

        # 2) mask padding, but keep the first end-of-text token as a real
        #    target so the model learns to STOP.
        pad_mask = targets == pad_token_id
        pad_indices = torch.nonzero(pad_mask).squeeze(-1)
        if pad_indices.numel() > 1:
            targets[pad_indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


# --------------------------------------------------------------------------- #
# Self-check: prove the masking lines up with the answer tokens
# --------------------------------------------------------------------------- #

def _demo_mask():
    tokenizer = tiktoken.get_encoding("gpt2")
    data = [{
        "instruction": "What is the capital of France?",
        "input": "",
        "output": "The capital of France is Paris.",
    }]
    ds = InstructionDataset(data, tokenizer)
    inputs, targets = answer_only_collate_fn(
        [ds[0]], allowed_max_length=None, device="cpu"
    )
    kept = targets[0][targets[0] != IGNORE_INDEX]
    print("Unmasked (answer) tokens decode to:")
    print(repr(tokenizer.decode(kept[kept != PAD_TOKEN_ID].tolist())))
    print(f"Total targets: {targets.shape[1]}, "
          f"used in loss: {(targets[0] != IGNORE_INDEX).sum().item()}")


# --------------------------------------------------------------------------- #
# End-to-end fine-tune using answer-only loss
# --------------------------------------------------------------------------- #

def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text_data)
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.1,
        "qkv_bias": True,
        "emb_dim": 768,
        "n_layers": 12,
        "n_heads": 12,
    }
    model_size = "124M"

    # --- data ---
    file_path = "instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )
    data = download_and_load_file(file_path, url)
    print("Examples:", len(data))

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.10)
    train_data = data[:train_portion]
    val_data = data[train_portion + test_portion:]

    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)
    collate = partial(
        answer_only_collate_fn,
        device=device,
        allowed_max_length=BASE_CONFIG["context_length"],
    )

    train_loader = DataLoader(
        InstructionDataset(train_data, tokenizer),
        batch_size=4, collate_fn=collate, shuffle=True,
        drop_last=True, num_workers=0,
    )
    val_loader = DataLoader(
        InstructionDataset(val_data, tokenizer),
        batch_size=4, collate_fn=collate, shuffle=False,
        drop_last=False, num_workers=0,
    )

    # --- pretrained GPT-2 weights (lazy import keeps tensorflow optional) ---
    from gpt_download import download_and_load_gpt2
    from Fine_TuneModel import load_weights_into_gpt

    _, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.to(device)

    with torch.no_grad():
        tl = calc_loss_loader(train_loader, model, device, num_batches=5)
        vl = calc_loss_loader(val_loader, model, device, num_batches=5)
    print(f"Initial loss — train: {tl:.4f}  val: {vl:.4f}")

    # --- train (answer-only loss comes entirely from the masked targets) ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    start = time.time()
    model.train()
    train_losses, val_losses, _ = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=2, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer,
    )
    print(f"Done in {(time.time() - start) / 60:.2f} min")

    # --- save ---
    save_path = "fine_tuned_gpt3/answer_only_model.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "model_config": BASE_CONFIG},
        save_path,
    )
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        _demo_mask()
    else:
        main()
