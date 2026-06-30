# MYLLM — GPT from scratch + RAG Q&A
# CPU image. For GPU, use an nvidia/cuda base + the matching CUDA torch wheel.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Build tools some torch / tensorflow wheels expect
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so they cache across code changes
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# App code (weights/data are mounted at runtime — see .dockerignore)
COPY . .

# Streamlit
EXPOSE 8501
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# Default: the focused RAG Q&A app.
# Override to run the full UI:   docker run ... streamlit run streamlit_app.py
# Or to fine-tune:               docker run ... python Fine_TuneModel.py answer-only
CMD ["streamlit", "run", "rag_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
