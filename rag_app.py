"""
RAG Q&A — answer questions grounded in your own documents.

The model doesn't need to *know* the answer; it retrieves relevant text and
answers from it. Best path for a small GPT-2.

Run:
    streamlit run rag_app.py

Needs: torch, tiktoken, streamlit, numpy
"""

import os

import torch
import tiktoken
import streamlit as st

from GPT_model import GPTModel, generate
from TrainLLM import text_to_token_ids, token_ids_to_text
from Fine_TuneModel import format_input
from rag import TfidfRetriever

DEFAULT_MODEL_PATH = "fine_tuned_gpt3/answer_only_model.pt"
RESPONSE_MARKER = "### Response:"
EOS_TOKEN_ID = 50256


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

@st.cache_resource
def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_prompt(question, context):
    """Same prompt the model was fine-tuned on, ending at '### Response:' so the
    model is cued to answer instead of continuing the context."""
    return format_input(
        {"instruction": question, "input": context}
    ) + "\n\n" + RESPONSE_MARKER


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
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt and "model_config" in ckpt:
        cfg = dict(ckpt["model_config"])
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "tok_emb.weight" in ckpt:
        # raw state dict — infer the essentials
        state_dict = ckpt
        vocab, emb = state_dict["tok_emb.weight"].shape
        layers = {int(k.split(".")[1]) for k in state_dict if k.startswith("trf_blocks.")}
        cfg = {
            "vocab_size": vocab, "emb_dim": emb,
            "context_length": state_dict["pos_emb.weight"].shape[0],
            "n_layers": (max(layers) + 1) if layers else 12,
            "n_heads": 12, "qkv_bias": "trf_blocks.0.att.W_query.bias" in state_dict,
        }
    else:
        raise ValueError("Unrecognized checkpoint format.")
    cfg["drop_rate"] = 0.0
    model = GPTModel(cfg)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, cfg


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="RAG Q&A", page_icon="📚", layout="centered")
st.title("📚 RAG Q&A")
st.caption("Ask questions and get answers grounded in your documents.")

device = get_device()
tokenizer = get_tokenizer()

for key, val in {
    "model": None, "cfg": None, "model_path": None, "retriever": None,
    "retriever_info": None,
}.items():
    st.session_state.setdefault(key, val)

# ---- Sidebar: model + decoding ---- #
with st.sidebar:
    st.header("Model")
    st.write(f"Device: `{device}`")
    model_path = st.text_input("Checkpoint path", DEFAULT_MODEL_PATH)
    if st.button("Load model", use_container_width=True):
        if not os.path.exists(model_path):
            st.error(f"Not found: {model_path}")
        else:
            with st.spinner("Loading..."):
                try:
                    st.session_state.model, st.session_state.cfg = load_model(
                        model_path, device
                    )
                    st.session_state.model_path = model_path
                    st.success("Model loaded.")
                except Exception as e:
                    st.session_state.model = None
                    st.error(f"Load failed: {e}")
    if st.session_state.model is not None:
        st.caption(f"Loaded `{st.session_state.model_path}`")

    st.divider()
    st.header("Answer settings")
    top_n = st.slider("Chunks to retrieve", 1, 8, 3)
    max_new_tokens = st.slider("Max answer tokens", 8, 256, 80, step=8)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.0, step=0.1)
    top_k = st.slider("Top-k", 0, 100, 0, step=5)
    st.caption("Temp 0 = greedy (most factual).")

# ---- Step 1: documents ---- #
st.subheader("1. Add your documents")
src = st.radio("Source", ["Paste text", "Upload files", "File path"], horizontal=True)

raw_texts = []
if src == "Paste text":
    pasted = st.text_area("Paste knowledge text", height=180)
    if pasted.strip():
        raw_texts = [pasted]
elif src == "Upload files":
    files = st.file_uploader(
        "Upload .txt / .md", type=["txt", "md"], accept_multiple_files=True
    )
    for f in files or []:
        raw_texts.append(f.read().decode("utf-8", errors="ignore"))
else:
    path = st.text_input("Path to .txt file", "the-verdict.txt")
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            raw_texts = [fh.read()]
    elif path:
        st.info("Path does not exist yet.")

c1, c2 = st.columns(2)
chunk_words = c1.slider("Chunk size (words)", 60, 240, 120, step=20)
overlap = c2.slider("Chunk overlap (words)", 0, 80, 40, step=10)

if st.button("Build index", type="secondary"):
    if not raw_texts:
        st.warning("Add some text first.")
    else:
        with st.spinner("Indexing..."):
            r = TfidfRetriever.from_texts(
                raw_texts, chunk_words=chunk_words, overlap=overlap
            )
        st.session_state.retriever = r
        st.session_state.retriever_info = f"{len(r.chunks)} chunks, {len(r.vocab)} terms"
        st.success(f"Index built — {st.session_state.retriever_info}")

if st.session_state.retriever is not None:
    st.caption(f"Index ready — {st.session_state.retriever_info}")

st.divider()

# ---- Step 2: ask ---- #
st.subheader("2. Ask a question")
question = st.text_input("Your question", "What is this text about?")

if st.button("Get answer", type="primary"):
    if st.session_state.model is None:
        st.warning("Load a model in the sidebar first.")
    elif st.session_state.retriever is None:
        st.warning("Build an index first.")
    elif not question.strip():
        st.warning("Type a question.")
    else:
        retriever = st.session_state.retriever
        cfg = st.session_state.cfg
        hits = retriever.retrieve(question, top_k=top_n)
        context = "\n\n".join(c for c, s in hits if s > 0) or hits[0][0]
        prompt = build_prompt(question, context)

        with st.spinner("Retrieving + answering..."):
            idx = text_to_token_ids(prompt, tokenizer).to(device)
            out = generate(
                model=st.session_state.model,
                idx=idx,
                max_new_tokens=max_new_tokens,
                context_size=cfg["context_length"],
                temperature=temperature,
                top_k=top_k if top_k > 0 else None,
                eos_id=EOS_TOKEN_ID,
            )
            answer = extract_response(token_ids_to_text(out, tokenizer))

        st.markdown("### Answer")
        st.write(answer or "_(empty — try more tokens or a different question)_")

        with st.expander(f"Sources ({len(hits)} retrieved chunks)"):
            for i, (chunk, score) in enumerate(hits, 1):
                st.markdown(f"**[{i}] relevance {score:.3f}**")
                st.text(chunk)
