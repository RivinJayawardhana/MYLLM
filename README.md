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
6. **Fine-tune** the model on instruction–response data (Alpaca-style) so it can follow instructions — with standard or **answer-only loss**.
7. **Evaluate** the fine-tuned model on a held-out test split.
8. **Generate** text, chat from a browser UI, or do **document Q&A with RAG**.

### Two ways to use it

- **As a learning resource** — every layer of a GPT is hand-written and readable.
- **As a small working assistant** — load GPT-2 weights, fine-tune on instructions, and query it via the Streamlit apps (including retrieval-augmented Q&A over your own documents).

---

## Project structure

**Core model & training**

| File | Purpose |
|---|---|
| `Text_Processing.py` | Tokenizers (`SimpleTokenizerV1/V2`) and the `GPTDatasetV1` + `create_dataloader_v1` data-loading pipeline (uses the `tiktoken` GPT-2 BPE tokenizer). |
| `Attention_machenism.py` | Attention from scratch: `SelfAttention_v1/v2`, `CausalAttention`, and `MultiHeadAttention`. |
| `GPT_model.py` | The full model: `LayerNorm`, `GELU`, `FeedForward`, `TransformerBlock`, `GPTModel`, plus `generate_text_simple` / `generate` (with temperature & top-k sampling). |
| `TrainLLM.py` | Pretraining loop: loss functions, evaluation, the `train_model_simple` training loop, and a `main()` that pretrains on `the-verdict.txt`. |
| `download.py` | Helper to fetch `gpt_download.py` from the upstream repo. |
| `gpt_download.py` | Downloads and loads the official OpenAI GPT-2 checkpoints (124M / 355M / 774M / 1558M). |

**Fine-tuning, evaluation & RAG**

| File | Purpose |
|---|---|
| `Fine_TuneModel.py` | Instruction fine-tuning. Standard `main()` (padding-only loss) **and** `finetune_answer_only(...)` for answer-only-loss fine-tuning on `alpaxa_data.json`. |
| `answer_only_loss.py` | The **answer-only loss** mechanism: a masked `InstructionDataset` + `answer_only_collate_fn` that compute loss on the response tokens only. Importable + runnable. |
| `evaluate_model.py` | Qualitative eval: runs the held-out test split through the model and prints *question → reference → generated* answers. |
| `rag.py` | Lightweight **retrieval** for document Q&A: `TfidfRetriever` (pure NumPy) with `chunk_text`, `.retrieve()`, `.context_for()`. |

**Apps & data**

| File / Folder | Purpose |
|---|---|
| `streamlit_app.py` | Full web UI — **Generate**, **Ask your docs (RAG)**, and **Fine-tune** tabs. |
| `rag_app.py` | Focused **RAG Q&A** web app: load a model, add documents, ask grounded questions. |
| `LLM.ipynb` | Notebook for interactive experimentation. |
| `the-verdict.txt` | Small text corpus used for the pretraining demo. |
| `instruction-data.json` | ~1.1k instruction examples (Alpaca-style). |
| `alpaxa_data.json` | ~52k instruction examples (full Alpaca set). |
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
- **Docker** (optional) if you prefer the containerised route
- A GPU is optional but strongly recommended for fine-tuning. The code auto-detects CUDA and falls back to CPU.

---

## Getting started

You can run the project two ways: a **local virtual environment** or **Docker**. Pick one.

### Option A — Local virtual environment (venv)

**1. Get the code**

```bash
git clone <your-repo-url> MYLLM
cd MYLLM
```

**2. Create & activate a virtual environment**

<details open>
<summary><b>Windows (PowerShell)</b></summary>

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> If activation is blocked, allow it for this session:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`
</details>

<details>
<summary><b>macOS / Linux (bash/zsh)</b></summary>

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```
</details>

**3. Install dependencies**

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

<details>
<summary>or install individually</summary>

```bash
pip install torch tiktoken numpy requests tqdm tensorflow streamlit
```
</details>

**4. Run something**

```bash
# fine-tune with answer-only loss (downloads GPT-2 weights on first run)
python Fine_TuneModel.py answer-only

# check answers on held-out examples
python evaluate_model.py

# launch the RAG Q&A web app  ->  http://localhost:8501
streamlit run rag_app.py
```

**5. Deactivate when done**

```bash
deactivate
```

> `tensorflow` is only required by `gpt_download.py` to read the original GPT-2 TensorFlow checkpoints. `streamlit` is only needed for the web UI. `numpy` is pinned `<2.0` for TensorFlow compatibility.

### Option B — Docker

See [Run with Docker](#run-with-docker) below for build & run commands. In short:

```bash
docker build -t myllm .
docker run --rm -p 8501:8501 \
    -v "$(pwd)/fine_tuned_gpt3:/app/fine_tuned_gpt3" \
    -v "$(pwd)/gpt2:/app/gpt2" \
    myllm
# open http://localhost:8501
```

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

> 💡 For noticeably better question-answering, use **answer-only loss** instead — see the [Answer-only loss fine-tuning](#answer-only-loss-fine-tuning-recommended-for-qa) section below.

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
pip install torch tiktoken numpy requests tqdm tensorflow streamlit

# 2. (Learn the basics) Pretrain on the small corpus
python TrainLLM.py

# 3. (Real model) Download GPT-2 weights + fine-tune with answer-only loss
python Fine_TuneModel.py answer-only

# 4. Check the answers on held-out examples
python evaluate_model.py

# 5. Chat / do document Q&A in the browser
streamlit run rag_app.py        # focused RAG Q&A
# or
streamlit run streamlit_app.py  # full UI (generate + RAG + fine-tune)
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

## Answer-only loss fine-tuning (recommended for Q&A)

By default the instruction collate function only masks **padding**, so the model spends most of its loss learning to reproduce the *prompt* (instruction + context) rather than the *answer*. The visible symptom is a model that **echoes the prompt or continues the text instead of answering**.

`answer_only_loss.py` fixes this by masking every token **before `### Response:`** with `ignore_index = -100` (which cross-entropy skips), so loss is computed **only on the answer**. It also keeps the first end-of-text token as a real target, so the model learns to *stop*.

### Run it

`Fine_TuneModel.py` exposes a fully-parameterised function:

```python
from Fine_TuneModel import finetune_answer_only

# defaults: alpaxa_data.json, 10k random examples (seed 123), 2 epochs, lr 5e-5,
# gpt2-small, saves to fine_tuned_gpt3/answer_only_model.pt
finetune_answer_only()

# customise freely
finetune_answer_only(
    max_examples=10000,                 # None = use all ~52k
    num_epochs=1,
    lr=3e-5,
    model_choice="gpt2-medium (355M)",
    batch_size=4,
    save_path="my_qa_model.pt",
)
```

Or from the shell:

```bash
python Fine_TuneModel.py answer-only     # answer-only loss fine-tune
python Fine_TuneModel.py                 # original padding-only flow
```

Key parameters of `finetune_answer_only`:

| Param | Default | Meaning |
|---|---|---|
| `data_path` | `alpaxa_data.json` | Instruction JSON (auto-downloaded if missing). |
| `max_examples` | `10000` | Cap the dataset size; `None` uses all examples. |
| `shuffle_before_subset` | `True` | Random sample (seed 123) vs. first N. |
| `model_choice` | `gpt2-small (124M)` | Also `medium`/`large`/`xl`. |
| `num_epochs` / `batch_size` / `lr` | `2` / `4` / `5e-5` | Standard training knobs. |
| `save_path` | `fine_tuned_gpt3/answer_only_model.pt` | Where to write the checkpoint. |

> **Verify the masking** (decodes the *unmasked* tokens, which should be the answer only):
> ```bash
> python answer_only_loss.py --demo
> ```

---

## Evaluating the model

Loss tells you the model is learning; it doesn't tell you the answers are good. `evaluate_model.py` runs the **held-out test split** (examples never seen in training) through the model and prints, per example, the instruction, the reference answer, and the generated answer.

```bash
python evaluate_model.py
python evaluate_model.py --num_samples 15 --temperature 0.0
python evaluate_model.py --model_path fine_tuned_gpt3/answer_only_model.pt --max_examples 10000
```

> Keep `--max_examples` equal to what you trained with, so the reconstructed test split is genuinely unseen. `--temperature 0.0` (greedy) is best for factual answers.

**Reading typical fine-tuning losses:** a train/val gap (e.g. train `1.6` / val `2.3`) indicates mild overfitting. Val perplexity ≈ e^(val loss); shrink the gap with **more data** (`max_examples=10000+`) or **fewer epochs** (`num_epochs=1`).

---

## RAG Q&A over your own documents (`rag_app.py`)

The most reliable way to get useful answers from a small model: don't make it *recall* facts, make it *answer from retrieved text*.

```bash
streamlit run rag_app.py
```

1. **Load model** (sidebar) — defaults to `fine_tuned_gpt3/answer_only_model.pt`.
2. **Add documents** — paste text, upload `.txt`/`.md`, or give a file path, then **Build index**.
3. **Ask a question** — the app retrieves the top-N relevant chunks (`rag.py`), injects them into the prompt's `### Input:` field, and the model answers from that context. Retrieved **sources** are shown with relevance scores.

How retrieval works (`rag.py`):

```python
from rag import TfidfRetriever

retriever = TfidfRetriever.from_text(open("docs.txt", encoding="utf-8").read())
for chunk, score in retriever.retrieve("What services are offered?", top_k=3):
    print(round(score, 3), chunk[:80])
```

- `chunk_text()` splits documents into overlapping word windows (respecting paragraphs).
- `TfidfRetriever` ranks chunks by TF-IDF cosine similarity — **pure NumPy, no extra dependencies**.
- For better semantic matching (synonyms, paraphrases) you can swap in an embedding model; the `.retrieve()` interface stays the same.

> **Important for document Q&A:** for the model to answer *from* the retrieved context, it needs to be good at reading the `### Input:` field. The Alpaca data has many empty-input examples, so a model fine-tuned only on it under-uses long context. For best results, fine-tune on **context-grounded QA** data (`{context, question, answer}`, e.g. SQuAD-style) and/or use a **larger base model** (355M+).

---

## Run with Docker

A `Dockerfile` (CPU) is included. Because model weights and the large dataset are **not** baked into the image (see `.dockerignore`), mount them at runtime as a volume.

```bash
# 1. Build
docker build -t myllm .

# 2. Run the RAG Q&A app, mounting your trained checkpoint(s) + data
docker run --rm -p 8501:8501 \
    -v "$(pwd)/fine_tuned_gpt3:/app/fine_tuned_gpt3" \
    -v "$(pwd)/gpt2:/app/gpt2" \
    myllm
# open http://localhost:8501
```

The default command launches `rag_app.py`. Override it for other entry points:

```bash
# Full UI (generate + RAG + fine-tune)
docker run --rm -p 8501:8501 -v "$(pwd):/app/data" myllm \
    streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0

# Fine-tune inside the container (mount data + an output dir)
docker run --rm \
    -v "$(pwd)/alpaxa_data.json:/app/alpaxa_data.json" \
    -v "$(pwd)/fine_tuned_gpt3:/app/fine_tuned_gpt3" \
    -v "$(pwd)/gpt2:/app/gpt2" \
    myllm python Fine_TuneModel.py answer-only
```

> **GPU:** the included image is CPU-only. For CUDA, switch the base image to an `nvidia/cuda:*-runtime` image, install the matching CUDA `torch` wheel, and run with `--gpus all`.
> **Windows PowerShell:** replace `$(pwd)` with `${PWD}`.

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

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| **Model echoes the prompt back** / prints the whole `### Instruction …` block | Inference prompt didn't cue `### Response:`, or trained with padding-only loss | The apps now append `### Response:` and strip it on output; fine-tune with **answer-only loss** (`finetune_answer_only`). |
| **Fluent text but ignores your context / makes facts up** | Loaded raw GPT-2 weights (not instruction-tuned), or model under-uses `### Input:` | Fine-tune on instructions; for docs Q&A use context-grounded data + a larger model. |
| **Rambles, never stops, runs into a new "### Instruction"** | Not stopping at EOS, or `max_new_tokens` too high | Lower max tokens; ensure training answers end cleanly (answer-only loss keeps the EOS target). |
| **Gibberish / random tokens** | Config mismatch on load (`qkv_bias`/size) or temperature too high | Load checkpoints saved with `model_config`; set temperature `0.0–0.3`, top-k ≤ 10. |
| **Right idea, wrong details** | 124M is too small to *know* facts | Use RAG (`rag_app.py`) and/or step up to 355M+. |
| **CUDA out of memory** | Batch/context too large | Lower `batch_size` or `context_length`, or run on CPU. |
| **`tensorflow` install issues** | Only needed to read original GPT-2 checkpoints | Imported lazily — inference/RAG don't need it. |

## Notes & tips

- The pretraining demo uses `context_length = 256` (in `TrainLLM.py`) while fine-tuning uses the full `1024`.
- Checkpoints save both `model_state_dict` and the config so the apps can rebuild the exact model on load (with a fallback that infers config from a raw state dict).
- **For factual Q&A**, prefer greedy decoding (`temperature = 0.0`); for creative text, raise temperature and use top-k.
- **Prompt format matters:** inference must use the same `### Instruction / ### Input / ### Response` format the model was trained on — the apps and helpers handle this for you.

---

## Dependencies by task

| You want to… | Needs |
|---|---|
| Run inference / generate / RAG | `torch`, `tiktoken`, `streamlit`, `numpy` |
| Download GPT-2 weights / fine-tune from them | + `tensorflow`, `requests`, `tqdm`, `numpy` |

---

## Credits

Architecture and weight-loading approach based on **Sebastian Raschka's** *Build a Large Language Model From Scratch* and the accompanying [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) repository (Apache 2.0).
