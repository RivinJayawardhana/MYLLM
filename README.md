# MYLLM — Build a GPT Large Language Model From Scratch

A hands-on implementation of a GPT-style Large Language Model built **from the ground up** in PyTorch — from raw text tokenization all the way to instruction fine-tuning. Every component (tokenizer, self-attention, transformer blocks, the GPT model, the training loop, and the OpenAI GPT-2 weight loader) is implemented by hand so you can see exactly how a modern LLM works.

This project follows the architecture popularized by Sebastian Raschka's *Build a Large Language Model From Scratch*.

---

## What this project does

The pipeline takes you through the **full lifecycle** of an LLM:

1. **Tokenize** raw text into token IDs.
2. **Build** the attention mechanism and transformer blocks.
3. **Assemble** the full GPT model (124M parameters).
4. **Pretrain** the model on plain text to predict the next token.
5. **Load** real pretrained OpenAI GPT-2 weights into the custom model.
6. **Fine-tune** the model on instruction–response data (Alpaca-style) so it can follow instructions.
7. **Generate** text from the trained/fine-tuned model.

---

## Project structure

| File / Folder | Purpose |
|---|---|
| `Text_Processing.py` | Tokenizers (`SimpleTokenizerV1/V2`) and the `GPTDatasetV1` + `create_dataloader_v1` data-loading pipeline (uses the `tiktoken` GPT-2 BPE tokenizer). |
| `Attention_machenism.py` | Attention from scratch: `SelfAttention_v1/v2`, `CausalAttention`, and `MultiHeadAttention`. |
| `GPT_model.py` | The full model: `LayerNorm`, `GELU`, `FeedForward`, `TransformerBlock`, `GPTModel`, plus `generate_text_simple` / `generate` (with temperature & top-k sampling). |
| `TrainLLM.py` | Pretraining loop: loss functions, evaluation, the `train_model_simple` training loop, and a `main()` that pretrains on `the-verdict.txt`. |
| `download.py` | Helper to fetch `gpt_download.py` from the upstream repo. |
| `gpt_download.py` | Downloads and loads the official OpenAI GPT-2 checkpoints (124M / 355M / 774M / 1558M). |
| `Fine_TuneModel.py` | Instruction fine-tuning: loads GPT-2 weights, builds the `InstructionDataset` with a custom padding/collate function, fine-tunes, and saves the model. |
| `the-verdict.txt` | Small text corpus used for pretraining demos. |
| `instruction-data.json` / `alpaxa_data.json` | Instruction–response datasets for fine-tuning. |
| `LLM.ipynb` | Notebook for interactive experimentation. |
| `model.pth` / `fine_tuned_gpt2/` / `fine_tuned_gpt3/` | Saved model checkpoints. |

> Note: `model.pth` and the `gpt2/` weight folder are git-ignored (they're large binary artifacts).

---

## Model configuration (GPT-2 124M)

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # BPE vocabulary size
    "context_length": 1024,  # max sequence length (256 for the pretraining demo)
    "emb_dim": 768,          # embedding / hidden dimension
    "n_heads": 12,           # attention heads
    "n_layers": 12,          # transformer blocks
    "drop_rate": 0.1,        # dropout
    "qkv_bias": False        # bias on Q/K/V projections
}
```

---

## Prerequisites

- **Python 3.10+** (developed on 3.12)
- A GPU is optional but strongly recommended for fine-tuning. The code auto-detects CUDA and falls back to CPU.

### Install dependencies

```bash
pip install torch tiktoken numpy requests tqdm tensorflow streamlit
```

> `tensorflow` is only required by `gpt_download.py` to read the original GPT-2 TensorFlow checkpoints. `streamlit` is only needed for the web UI (below).

---

## Step-by-step: Create your own LLM

### Step 1 — Understand tokenization
`Text_Processing.py` turns text into the integer token IDs the model consumes. It uses the GPT-2 byte-pair-encoding tokenizer via `tiktoken` and slices long text into overlapping `(input, target)` chunks for next-token prediction.

```python
from Text_Processing import create_dataloader_v1

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    text = f.read()

loader = create_dataloader_v1(text, batch_size=4, max_length=256, stride=128)
inputs, targets = next(iter(loader))
print(inputs.shape, targets.shape)   # (4, 256) (4, 256)
```

### Step 2 — Understand attention
`Attention_machenism.py` implements multi-head **causal** self-attention — the core mechanism that lets each token attend to previous tokens. `MultiHeadAttention` is what the GPT model actually uses.

### Step 3 — Assemble the GPT model
`GPT_model.py` stacks token + positional embeddings, `n_layers` transformer blocks (each = multi-head attention + feed-forward with residual connections and layer norm), and a final linear head projecting back to the vocabulary.

```python
import torch
from GPT_model import GPTModel, GPT_CONFIG_124M

model = GPTModel(GPT_CONFIG_124M)
logits = model(torch.randint(0, 50257, (1, 10)))
print(logits.shape)   # (1, 10, 50257)
```

### Step 4 — Pretrain from scratch
`TrainLLM.py` trains the model to predict the next token on `the-verdict.txt`. It splits data 90/10 train/val, uses AdamW, prints train/val loss, generates samples each epoch, and saves a checkpoint.

```bash
python TrainLLM.py
```

This produces `model_and_optimizer.pth` and prints generated text such as:

```
Output text:
 Every effort moves you ...
```

> Pretraining a 124M model on a tiny corpus is for **learning**, not quality. For a capable model, continue to Step 5.

### Step 5 — Download official GPT-2 weights
Instead of training 124M parameters yourself, load OpenAI's pretrained GPT-2 weights.

```bash
# (optional) fetch the upstream downloader
python download.py

# the weights are downloaded automatically by gpt_download.py when fine-tuning
```

`gpt_download.py` pulls the chosen checkpoint (`124M` by default) into a `gpt2/` folder and returns the parameters as NumPy arrays.

### Step 6 — Instruction fine-tune
`Fine_TuneModel.py` ties it all together:
1. Detects GPU/CPU and prints PyTorch info.
2. Downloads the instruction dataset (saved as `alpaxa_data.json`).
3. Splits it 85% train / 10% test / 5% val.
4. Formats each example into an instruction prompt (`### Instruction / ### Input / ### Response`).
5. Pads batches with a `custom_collate_fn` (pad token `50256`, ignore index `-100`).
6. Downloads GPT-2 124M weights and loads them into the custom `GPTModel`.
7. Fine-tunes for 1 epoch with AdamW (`lr=5e-5`).
8. Saves the result to `fine_tuned_gpt3/fine_tuned_gpt2/model.pt`.

```bash
python Fine_TuneModel.py
```

On the next run, if a saved fine-tuned model is found it will offer to load it instead of retraining:

```
=== FINE-TUNED MODEL FOUND AT ... ===
Load fine-tuned model? (y/n):
```

### Step 7 — Generate text
Use the `generate` helper for sampling with temperature and top-k:

```python
import tiktoken, torch
from GPT_model import GPTModel, generate
from TrainLLM import text_to_token_ids, token_ids_to_text, GPT_CONFIG_124M

tokenizer = tiktoken.get_encoding("gpt2")
model = GPTModel(GPT_CONFIG_124M)            # then load_state_dict(...) from a checkpoint
model.eval()

ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=20,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4,
)
print(token_ids_to_text(ids, tokenizer))
```

---

## Typical end-to-end run

```bash
# 1. Install deps
pip install torch tiktoken numpy requests tqdm tensorflow

# 2. (Learn the basics) Pretrain on the small corpus
python TrainLLM.py

# 3. (Real model) Download GPT-2 weights + instruction fine-tune
python Fine_TuneModel.py
```

---

## Web UI (Streamlit)

`streamlit_app.py` gives you a browser interface for both **inference** and **fine-tuning** — no editing code required.

```bash
streamlit run streamlit_app.py
```

**💬 Generate tab**
- Enter a checkpoint path (default `fine_tuned_gpt2/model.pth`) and click **Load model**.
- Type an **Instruction** (and optional **Input**), tune `max_new_tokens` / `temperature` / `top-k`, and click **Generate response**.
- The app auto-detects the model config from the checkpoint, formats the Alpaca-style prompt, and shows just the model's response (with the full prompt+completion available in an expander).

**📚 Ask your docs (RAG) tab**
- Add knowledge by pasting text, uploading `.txt`/`.md` files, or pointing at a text file path, then click **Build / rebuild index**.
- Ask a question — the app retrieves the most relevant chunks (TF-IDF, see `rag.py`), drops them into the prompt's `### Input:` field, and the model answers **from that context** instead of from memory.
- This is the most reliable Q&A path for a small GPT-2: the model only has to extract/rephrase, not recall facts. Retrieved sources are shown under an expander.
- Retrieval is pure NumPy (no extra dependencies). For higher quality you can swap `TfidfRetriever` for an embedding model — the `.retrieve()` interface stays the same.

**🎯 Fine-tune tab**
1. Pick instruction data — a JSON path (default `instruction-data.json`) or upload your own.
2. Choose starting weights — **download pretrained GPT-2** or **continue from an existing checkpoint**.
3. Set epochs / batch size / learning rate / dropout and a save path.
4. Click **Start fine-tuning** — it trains (live loss prints to the console), plots train/val loss, and saves a checkpoint you can immediately load back in the Generate tab.

> Inference only needs `torch`, `tiktoken`, and `streamlit`. TensorFlow is imported lazily and is only required if you choose to **download pretrained GPT-2** weights in the Fine-tune tab.

The uploaded/loaded data must be a JSON list of `{"instruction", "input", "output"}` objects, e.g.:

```json
[
  {"instruction": "Edit the sentence for grammar.", "input": "He go to school.", "output": "He goes to school."}
]
```

---

## How it works (architecture at a glance)

```
text
  │  tiktoken BPE
  ▼
token IDs ──► token embedding + positional embedding
                         │
                         ▼
            ┌──────────────────────────┐
            │  TransformerBlock × 12    │
            │  • LayerNorm              │
            │  • MultiHeadAttention     │  (causal mask)
            │  • residual               │
            │  • LayerNorm              │
            │  • FeedForward (GELU)     │
            │  • residual               │
            └──────────────────────────┘
                         │
                         ▼
              final LayerNorm ──► Linear head ──► logits (vocab_size)
                         │
                         ▼
            softmax + (top-k / temperature) sampling ──► next token
```

---

## Notes & tips

- **CUDA out of memory?** Lower `batch_size` (in `Fine_TuneModel.py`) or `context_length`, or run on CPU.
- **`tensorflow` install issues?** It is only needed to read the original GPT-2 checkpoints in `gpt_download.py`.
- The pretraining demo uses `context_length = 256` (in `TrainLLM.py`) while fine-tuning uses the full `1024`.
- Checkpoints save both `model_state_dict` and the config so you can rebuild the exact model on load.

---

## Credits

Architecture and weight-loading approach based on **Sebastian Raschka's** *Build a Large Language Model From Scratch* and the accompanying [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) repository (Apache 2.0).
