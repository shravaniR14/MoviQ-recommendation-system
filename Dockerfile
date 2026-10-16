# ── Base image (slim Python, no GPU) ────────────────────────────────────────
FROM python:3.12-slim

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies (CPU-only torch) ────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Pre-download the embedding model at build time ──────────────────────────
# This means the model is baked into the image and first request is fast.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('intfloat/multilingual-e5-base')"

# ── Copy application code ────────────────────────────────────────────────────
COPY api/ ./api/
COPY ml/ ./ml/
COPY data/processed/ ./data/processed/

# ── Environment defaults (real values set as Render secrets) ─────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# ── Expose port ──────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start server ─────────────────────────────────────────────────────────────
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]