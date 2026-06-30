"""
Evaluate a fine-tuned model qualitatively.

Loss tells you the model is learning; it doesn't tell you the answers are good.
This runs the HELD-OUT test split (examples that were NOT used in training)
through the model and prints, for each:

    Instruction / Input / Reference answer / Generated answer

so you can judge quality directly.

Run:
    python evaluate_model.py
    python evaluate_model.py --model_path fine_tuned_gpt3/answer_only_model.pt --num_samples 10
    python evaluate_model.py --temperature 0.0 --max_new_tokens 80
"""

import json
import random
import argparse

import torch
import tiktoken

from GPT_model import GPTModel, generate
from TrainLLM import text_to_token_ids, token_ids_to_text
from Fine_TuneModel import format_input

RESPONSE_MARKER = "### Response:"


def build_prompt(entry):
    """Prompt ending in '### Response:' so the model is cued to answer."""
    return format_input(entry) + "\n\n" + RESPONSE_MARKER


def extract_response(full_text):
    if RESPONSE_MARKER in full_text:
        full_text = full_text.split(RESPONSE_MARKER, 1)[1]
    for stop in ("### Instruction", "### Input", "### Response",
                 "Below is an instruction"):
        if stop in full_text:
            full_text = full_text.split(stop, 1)[0]
    return full_text.replace("<|endoftext|>", "").strip()


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    if not (isinstance(ckpt, dict) and "model_state_dict" in ckpt
            and "model_config" in ckpt):
        raise ValueError(
            "Checkpoint must contain 'model_state_dict' and 'model_config'."
        )
    cfg = dict(ckpt["model_config"])
    cfg["drop_rate"] = 0.0  # inference
    model = GPTModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg


def get_test_split(data_path, max_examples, shuffle_before_subset=True):
    """Reproduce the exact same split finetune_answer_only used, and return the
    test slice (the 10% that training never touched)."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if max_examples is not None and len(data) > max_examples:
        if shuffle_before_subset:
            random.Random(123).shuffle(data)
        data = data[:max_examples]
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.10)
    return data[train_portion:train_portion + test_portion]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="fine_tuned_gpt3/answer_only_model.pt")
    parser.add_argument("--data_path", default="alpaxa_data.json")
    parser.add_argument("--max_examples", type=int, default=3000,
                        help="Must match what you trained with, to rebuild the "
                             "same held-out test split.")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 = greedy (best for factual answers).")
    parser.add_argument("--top_k", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = tiktoken.get_encoding("gpt2")
    model, cfg = load_model(args.model_path, device)
    print(f"Loaded {args.model_path} "
          f"({cfg['n_layers']} layers, ctx {cfg['context_length']})")

    test_data = get_test_split(args.data_path, args.max_examples)
    if not test_data:
        print("No held-out test examples — lower --max_examples.")
        return
    samples = test_data[:args.num_samples]
    print(f"Showing {len(samples)} held-out (unseen) examples\n")

    for i, entry in enumerate(samples, 1):
        prompt = build_prompt(entry)
        idx = text_to_token_ids(prompt, tokenizer).to(device)
        out = generate(
            model=model,
            idx=idx,
            max_new_tokens=args.max_new_tokens,
            context_size=cfg["context_length"],
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            eos_id=50256,
        )
        generated = extract_response(token_ids_to_text(out, tokenizer))

        print("=" * 70)
        print(f"[{i}] Instruction: {entry['instruction']}")
        if entry.get("input"):
            print(f"    Input:       {entry['input']}")
        print(f"    Reference:   {entry['output']}")
        print(f"    Generated:   {generated}")
    print("=" * 70)


if __name__ == "__main__":
    main()
