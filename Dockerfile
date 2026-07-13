# CasesBasis production image.
#
# Heavy but deterministic: CPU-only PyTorch + transformers + faiss + the FastAPI
# app. The two embedding models the app loads at runtime
# (all-MiniLM-L6-v2 for the CXR/PTB-XL FAISS retrievers, BAAI/bge-small-en-v1.5
# for the knowledge-graph embedder) are baked in at build time so a live boot
# never has to reach HuggingFace.
#
# Runtime data (the ~835 MB CXR FAISS index, the PTB-XL index, the raw CXR
# images) and the SQLite DB are NOT baked in — they are mounted as volumes by
# docker-compose.prod.yml so the image stays immutable and small-ish.

FROM python:3.11-slim-bookworm AS base

# System libs needed by matplotlib, pillow, faiss, wfdb and torch at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libjpeg62-turbo \
        libpng16-16 \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Keep torch/OMP from allocating a large semaphore pool (see web_app lifespan).
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    # Bake HF model cache into the image at this path.
    HF_HOME=/opt/hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf-cache/sentence-transformers

WORKDIR /app

# ── Python deps ────────────────────────────────────────────────────────────────
# CPU-only torch first (from the PyTorch CPU wheel index — a fraction of the CUDA
# build), then the pinned application lock. sentence-transformers/transformers
# see torch already present and won't pull the CUDA build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0

COPY requirements.txt .
RUN pip install -r requirements.txt

# ── Bake the embedding models into the image ────────────────────────────────────
# Downloads once at build time into HF_HOME so the running container is offline
# for model weights. Must match the names used in the code:
#   backend/cxr_service.py, backend/ptbxl_service.py -> all-MiniLM-L6-v2
#   knowledge/embedder.py                            -> BAAI/bge-small-en-v1.5
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('BAAI/bge-small-en-v1.5'); AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')"

# ── Application code ────────────────────────────────────────────────────────────
COPY . .

# Runtime state dir (SQLite DB lives here via DB_PATH; mounted as a volume in prod).
RUN mkdir -p /app/var

# Drop privileges.
RUN useradd --create-home --uid 10001 appuser && \
    chown -R appuser:appuser /app /opt/hf-cache
USER appuser

EXPOSE 8000

# Container-level liveness: the app's own ops endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Single worker: session state (SESSIONS, _SESSIONS) is in-process, so the app
# must not be scaled to multiple workers. Robustness comes from the container
# restart policy, not from worker replication.
CMD ["uvicorn", "backend.web_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "75"]
