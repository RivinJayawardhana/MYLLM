"""
Streamlit app for the MYLLM project.

Two things in one UI:
  1. Load a (fine-tuned) GPT model and chat with it  -> "Generate" tab
  2. Fine-tune a model on instruction data and save it -> "Fine-tune" tab

Run with:
    streamlit run streamlit_app.py

Inference only needs: torch, tiktoken, streamlit
Fine-tuning from raw GPT-2 weights additionally needs: numpy, tensorflow, tqdm, requests
(tensorflow is imported lazily, only when you download GPT-2 weights.)
"""

import os
import json
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
import streamlit as st

from GPT_model import GPTModel, generate
from TrainLLM import (
    text_to_token_ids,
    token_ids_to_text,
    calc_loss_loader,
    train_model_simple,
)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,        # 0 for inference; overridden for training
    "qkv_bias": True,        # GPT-2 weights use QKV bias
}

MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

DEFAULT_MODEL_PATH = "fine_tuned_gpt2/model.pth"
DEFAULT_DATA_PATH = "instruction-data.json"

PAD_TOKEN_ID = 50256
EOS_TOKEN_ID = 50256


@st.cache_resource
def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Instruction helpers (kept local so inference doesn't import tensorflow)
# --------------------------------------------------------------------------- #

def format_input(entry):
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
    return instruction_text + input_text


RESPONSE_MARKER = "### Response:"


def build_prompt(instruction, user_input=""):
    """Prompt that matches training AND explicitly cues the answer, so the
    model starts at '### Response:' instead of continuing the input text."""
    return format_input(
        {"instruction": instruction, "input": user_input}
    ) + "\n\n" + RESPONSE_MARKER


def extract_response(full_text):
    # take everything after the (first) response marker
    if RESPONSE_MARKER in full_text:
        full_text = full_text.split(RESPONSE_MARKER, 1)[1]
    # if the model rambled into a new section, cut it off there
    for stop in ("### Instruction", "### Input", "### Response", "Below is an instruction"):
        if stop in full_text:
            full_text = full_text.split(stop, 1)[0]
    return full_text.replace("<|endoftext|>", "").strip()


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            prompt = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            self.encoded_texts.append(tokenizer.encode(prompt + response_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def custom_collate_fn(batch, pad_token_id=PAD_TOKEN_ID, ignore_index=-100,
                      allowed_max_length=None, device="cpu"):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
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


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def infer_config_from_state_dict(state_dict, fallback):
    """Best-effort reconstruction of the model config from a raw state dict."""
    cfg = dict(fallback)
    try:
        cfg["vocab_size"], cfg["emb_dim"] = state_dict["tok_emb.weight"].shape
        cfg["context_length"] = state_dict["pos_emb.weight"].shape[0]
        layers = {
            int(k.split(".")[1])
            for k in state_dict
            if k.startswith("trf_blocks.")
        }
        if layers:
            cfg["n_layers"] = max(layers) + 1
        # whether QKV projections carry bias
        cfg["qkv_bias"] = "trf_blocks.0.att.W_query.bias" in state_dict
    except Exception:
        pass
    return cfg


def build_config(model_choice, context_length, qkv_bias, drop_rate=0.0):
    cfg = dict(BASE_CONFIG)
    cfg.update(MODEL_CONFIGS[model_choice])
    cfg["context_length"] = context_length
    cfg["qkv_bias"] = qkv_bias
    cfg["drop_rate"] = drop_rate
    return cfg


def load_model_from_checkpoint(path, fallback_cfg, device):
    """Load a model from a checkpoint that may be:
       - {'model_state_dict': ..., 'model_config': {...}}
       - {'model_state_dict': ...}
       - a raw state_dict
    """
    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        cfg = checkpoint.get("model_config") or infer_config_from_state_dict(
            state_dict, fallback_cfg
        )
    elif isinstance(checkpoint, dict) and "tok_emb.weight" in checkpoint:
        state_dict = checkpoint
        cfg = infer_config_from_state_dict(state_dict, fallback_cfg)
    else:
        raise ValueError("Unrecognized checkpoint format.")

    cfg["drop_rate"] = 0.0  # inference
    model = GPTModel(cfg)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, cfg


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="MYLLM Studio", page_icon="🧠", layout="centered")
st.title("🧠 MYLLM Studio")
st.caption("Load a GPT model, chat with it, or fine-tune on your own instruction data.")

device = get_device()
tokenizer = get_tokenizer()

with st.sidebar:
    st.header("Environment")
    st.write(f"**Device:** `{device}`")
    if device.type == "cuda":
        st.write(f"**GPU:** {torch.cuda.get_device_name(0)}")
    st.divider()
    st.header("Model size")
    model_choice = st.selectbox("Architecture", list(MODEL_CONFIGS.keys()), index=0)
    context_length = st.select_slider(
        "Context length", options=[256, 512, 768, 1024], value=1024
    )
    qkv_bias = st.checkbox("QKV bias (GPT-2 weights = on)", value=True)

if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.loaded_cfg = None
    st.session_state.loaded_path = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
    st.session_state.retriever_info = None


def run_generation(prompt, max_new_tokens, temperature, top_k):
    """Shared generation used by both the Generate and Ask-your-docs tabs."""
    model = st.session_state.model
    cfg = st.session_state.loaded_cfg
    idx = text_to_token_ids(prompt, tokenizer).to(device)
    out = generate(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        context_size=cfg["context_length"],
        temperature=temperature,
        top_k=top_k if top_k > 0 else None,
        eos_id=EOS_TOKEN_ID,
    )
    return token_ids_to_text(out, tokenizer)


tab_generate, tab_rag, tab_finetune = st.tabs(
    ["💬 Generate", "📚 Ask your docs (RAG)", "🎯 Fine-tune"]
)

# --------------------------------------------------------------------------- #
# Tab 1: Generate
# --------------------------------------------------------------------------- #

with tab_generate:
    st.subheader("Load a model")
    col1, col2 = st.columns([3, 1])
    with col1:
        model_path = st.text_input("Checkpoint path (.pth / .pt)", DEFAULT_MODEL_PATH)
    with col2:
        st.write("")
        st.write("")
        load_clicked = st.button("Load model", use_container_width=True)

    if load_clicked:
        if not os.path.exists(model_path):
            st.error(f"File not found: `{model_path}`")
        else:
            with st.spinner("Loading model..."):
                try:
                    fallback = build_config(model_choice, context_length, qkv_bias)
                    model, cfg = load_model_from_checkpoint(model_path, fallback, device)
                    st.session_state.model = model
                    st.session_state.loaded_cfg = cfg
                    st.session_state.loaded_path = model_path
                    st.success("Model loaded.")
                except Exception as e:
                    st.session_state.model = None
                    st.error(f"Failed to load: {e}")

    if st.session_state.model is not None:
        st.caption(
            f"Loaded `{st.session_state.loaded_path}` — "
            f"{st.session_state.loaded_cfg['n_layers']} layers, "
            f"emb {st.session_state.loaded_cfg['emb_dim']}, "
            f"ctx {st.session_state.loaded_cfg['context_length']}"
        )

    st.divider()
    st.subheader("Ask the model")
    instruction = st.text_area(
        "Instruction",
        "Rewrite the following sentence in a more formal way.",
        height=80,
    )
    user_input = st.text_area(
        "Input (optional)", "I wanna grab some food.", height=68
    )

    c1, c2, c3 = st.columns(3)
    max_new_tokens = c1.slider("Max new tokens", 8, 256, 60, step=8)
    temperature = c2.slider("Temperature", 0.0, 2.0, 0.7, step=0.1)
    top_k = c3.slider("Top-k", 0, 100, 40, step=5)

    if st.button("Generate response", type="primary"):
        if st.session_state.model is None:
            st.warning("Load a model first.")
        elif not instruction.strip():
            st.warning("Enter an instruction.")
        else:
            prompt = build_prompt(instruction, user_input)
            with st.spinner("Generating..."):
                full_text = run_generation(
                    prompt, max_new_tokens, temperature, top_k
                )
            st.markdown("**Response**")
            st.write(extract_response(full_text))
            with st.expander("Full prompt + completion"):
                st.text(full_text)

# --------------------------------------------------------------------------- #
# Tab 2: Ask your docs (RAG)
# --------------------------------------------------------------------------- #

with tab_rag:
    st.subheader("1. Add your documents")
    st.caption(
        "The model answers using retrieved text — it doesn't need to *know* the "
        "facts, just extract them. Best path for a small GPT-2."
    )
    rag_mode = st.radio(
        "Source", ["Paste text", "Upload files", "Text file path"],
        horizontal=True,
    )

    raw_texts = []
    if rag_mode == "Paste text":
        pasted = st.text_area("Paste your knowledge text here", height=160)
        if pasted.strip():
            raw_texts = [pasted]
    elif rag_mode == "Upload files":
        files = st.file_uploader(
            "Upload .txt / .md files", type=["txt", "md"], accept_multiple_files=True
        )
        for f in files or []:
            raw_texts.append(f.read().decode("utf-8", errors="ignore"))
    else:
        path = st.text_input("Path to a .txt file", "the-verdict.txt")
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                raw_texts = [fh.read()]
        elif path:
            st.info("Path does not exist yet.")

    rc1, rc2 = st.columns(2)
    chunk_words = rc1.slider("Chunk size (words)", 60, 240, 120, step=20)
    overlap = rc2.slider("Chunk overlap (words)", 0, 80, 40, step=10)

    if st.button("Build / rebuild index"):
        if not raw_texts:
            st.warning("Add some text first.")
        else:
            from rag import TfidfRetriever
            with st.spinner("Indexing..."):
                retriever = TfidfRetriever.from_texts(
                    raw_texts, chunk_words=chunk_words, overlap=overlap
                )
            st.session_state.retriever = retriever
            st.session_state.retriever_info = (
                f"{len(retriever.chunks)} chunks, {len(retriever.vocab)} terms"
            )
            st.success(f"Index built: {st.session_state.retriever_info}")

    if st.session_state.retriever is not None:
        st.caption(f"Index ready — {st.session_state.retriever_info}")

    st.divider()
    st.subheader("2. Ask a question")
    question = st.text_input("Question", "What is this text about?")

    g1, g2, g3, g4 = st.columns(4)
    rag_top_k = g1.slider("Chunks to retrieve", 1, 8, 3)
    rag_max_tokens = g2.slider("Max new tokens", 8, 256, 80, step=8)
    rag_temp = g3.slider("Temperature ", 0.0, 1.5, 0.2, step=0.1)
    rag_topk_gen = g4.slider("Top-k (gen)", 0, 100, 20, step=5)

    if st.button("Answer from context", type="primary"):
        if st.session_state.model is None:
            st.warning("Load a model in the Generate tab first.")
        elif st.session_state.retriever is None:
            st.warning("Build an index first.")
        elif not question.strip():
            st.warning("Enter a question.")
        else:
            retriever = st.session_state.retriever
            hits = retriever.retrieve(question, top_k=rag_top_k)
            context = "\n\n".join(c for c, s in hits if s > 0) or hits[0][0]
            # Same prompt format the model is trained on: question + context
            prompt = build_prompt(question, context)
            with st.spinner("Retrieving + generating..."):
                full_text = run_generation(
                    prompt, rag_max_tokens, rag_temp, rag_topk_gen
                )
            st.markdown("**Answer**")
            st.write(extract_response(full_text))
            with st.expander("Retrieved context (sources)"):
                for i, (chunk, score) in enumerate(hits, 1):
                    st.markdown(f"**[{i}] score {score:.3f}**")
                    st.text(chunk)
            with st.expander("Full prompt + completion"):
                st.text(full_text)

# --------------------------------------------------------------------------- #
# Tab 3: Fine-tune
# --------------------------------------------------------------------------- #

with tab_finetune:
    st.subheader("1. Instruction data")
    data_mode = st.radio(
        "Data source",
        ["Use file path", "Upload JSON"],
        horizontal=True,
    )

    data = None
    if data_mode == "Use file path":
        data_path = st.text_input("JSON path", DEFAULT_DATA_PATH)
        if os.path.exists(data_path):
            try:
                with open(data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                st.error(f"Could not read data: {e}")
        else:
            st.info("Path does not exist yet.")
    else:
        uploaded = st.file_uploader("Upload instruction JSON", type=["json"])
        if uploaded is not None:
            try:
                data = json.load(uploaded)
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

    if data is not None:
        st.success(f"{len(data)} examples loaded.")
        with st.expander("Preview first example"):
            st.json(data[0])

    st.subheader("2. Starting weights")
    init_mode = st.radio(
        "Initialize from",
        ["Download pretrained GPT-2", "Existing checkpoint"],
        horizontal=True,
    )
    init_checkpoint = None
    if init_mode == "Existing checkpoint":
        init_checkpoint = st.text_input(
            "Checkpoint path to continue from", DEFAULT_MODEL_PATH
        )

    st.subheader("3. Training settings")
    t1, t2, t3 = st.columns(3)
    num_epochs = t1.number_input("Epochs", 1, 50, 1)
    batch_size = t2.number_input("Batch size", 1, 32, 4)
    lr = t3.number_input("Learning rate", 1e-6, 1e-2, 5e-5, format="%.6f")

    s1, s2 = st.columns(2)
    train_drop_rate = s1.slider("Dropout (training)", 0.0, 0.5, 0.1, step=0.05)
    save_path = s2.text_input("Save model to", "fine_tuned_gpt3/model.pt")

    st.divider()
    if st.button("Start fine-tuning", type="primary"):
        if data is None or len(data) < 3:
            st.warning("Provide at least a few instruction examples.")
            st.stop()

        cfg = build_config(model_choice, context_length, qkv_bias, train_drop_rate)

        # ---- Build model ----
        status = st.status("Preparing model...", expanded=True)
        try:
            model = GPTModel(cfg)
            if init_mode == "Download pretrained GPT-2":
                status.write("Downloading GPT-2 weights (needs tensorflow)...")
                # lazy import so inference users don't need tensorflow
                from gpt_download import download_and_load_gpt2
                from Fine_TuneModel import load_weights_into_gpt

                size = model_choice.split(" ")[-1].strip("()")
                _, params = download_and_load_gpt2(model_size=size, models_dir="gpt2")
                load_weights_into_gpt(model, params)
                status.write("GPT-2 weights loaded.")
            else:
                if not init_checkpoint or not os.path.exists(init_checkpoint):
                    status.update(label="Checkpoint not found.", state="error")
                    st.stop()
                status.write(f"Loading {init_checkpoint}...")
                loaded, _ = load_model_from_checkpoint(init_checkpoint, cfg, device)
                model.load_state_dict(loaded.state_dict())
            model.to(device)
        except Exception as e:
            status.update(label=f"Model setup failed: {e}", state="error")
            st.exception(e)
            st.stop()

        # ---- Data loaders ----
        status.write("Building data loaders...")
        torch.manual_seed(123)
        n = len(data)
        train_portion = int(n * 0.85)
        test_portion = int(n * 0.10)
        train_data = data[:train_portion]
        val_data = data[train_portion + test_portion:]
        if len(val_data) == 0:
            val_data = data[-1:]

        collate = partial(
            custom_collate_fn, device=device, allowed_max_length=cfg["context_length"]
        )
        train_loader = DataLoader(
            InstructionDataset(train_data, tokenizer),
            batch_size=batch_size, collate_fn=collate,
            shuffle=True, drop_last=True, num_workers=0,
        )
        val_loader = DataLoader(
            InstructionDataset(val_data, tokenizer),
            batch_size=batch_size, collate_fn=collate,
            shuffle=False, drop_last=False, num_workers=0,
        )

        if len(train_loader) == 0:
            status.update(
                label="Not enough data for one batch — lower the batch size.",
                state="error",
            )
            st.stop()

        # ---- Train ----
        status.write(f"Training for {num_epochs} epoch(s) on {device}...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
        model.train()
        try:
            with st.spinner("Training in progress (watch the console for live loss)..."):
                train_losses, val_losses, _ = train_model_simple(
                    model, train_loader, val_loader, optimizer, device,
                    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
                    start_context=format_input(val_data[0]), tokenizer=tokenizer,
                )
        except Exception as e:
            status.update(label=f"Training failed: {e}", state="error")
            st.exception(e)
            st.stop()

        # ---- Save ----
        status.write("Saving model...")
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": cfg,
            },
            save_path,
        )
        status.update(label="Fine-tuning complete.", state="complete")

        st.success(f"Saved to `{save_path}`")
        if train_losses:
            mc1, mc2 = st.columns(2)
            mc1.metric("Final train loss", f"{train_losses[-1]:.4f}")
            if val_losses:
                mc2.metric("Final val loss", f"{val_losses[-1]:.4f}")
            st.line_chart(
                {"train": train_losses, "val": val_losses}
                if val_losses else {"train": train_losses}
            )
        st.info("Switch to the Generate tab and load this path to chat with it.")
