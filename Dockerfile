# ─────────────────────────────────────────────────────────────────────
# CareerMappingGenomics — Dockerfile
# Multi-stage build: keeps the final image lean.
#
# Build:  docker build -t careermapping:v1 .
# Run:    docker run --rm -v $(pwd)/output:/app/output careermapping:v1
# API:    docker run --rm -p 8000:8000 careermapping:v1 python src/api/app.py
# Tests:  docker run --rm careermapping:v1 python tests/test_all.py
# ─────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

LABEL maintainer="Abdallah El-Daly"
LABEL description="CareerMappingGenomics: Multi-Modal Deep Learning for Career Recommendation"
LABEL version="1.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies ────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy source ────────────────────────────────────────────────────
COPY . .

# Create output directories
RUN mkdir -p /app/output/reports \
             /app/output/models \
             /app/data/synthetic \
             /app/data/processed

# ── Volume for output ──────────────────────────────────────────────
# Mount with: -v $(pwd)/your_local_output:/app/output
VOLUME ["/app/output"]

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

# Expose API port
EXPOSE 8000

# ── Default command: full pipeline (1000 samples, 20 epochs) ───────
# Override with: docker run ... python main.py --quick
CMD ["python", "main.py", "--samples", "1000", "--epochs", "20"]
