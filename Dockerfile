# ------------------------------------------------------------------ #
#  Hybrid RAG API — Dockerfile                                        #
#  Compatible with:                                                   #
#    - Hugging Face Spaces (CPU, persistent /data volume)            #
#    - Local docker-compose                                           #
# ------------------------------------------------------------------ #

FROM python:3.11-slim

# HF Spaces runs containers as uid 1000
# We match that here so volume permissions work on both local + HF
ARG UID=1000
ARG GID=1000

# System deps:
#   libgl1 / libglib2.0 — required by PyMuPDF (fitz)
#   curl               — used in Docker healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user — never run ML workloads as root
RUN groupadd -g ${GID} appgroup && \
    useradd -u ${UID} -g appgroup -m -s /bin/bash appuser

WORKDIR /app

# Install Python deps first (layer cache — only re-runs on requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source
COPY --chown=appuser:appgroup . .

# HF Spaces serves on port 7860 by default
# Local docker-compose maps this to 8000 externally
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure upload + index dirs exist inside the container
# On HF Spaces these will be under /data (mounted persistent volume)
# On local they live inside the container or bind-mounted
RUN mkdir -p /app/uploads /app/index_store && \
    chown -R appuser:appgroup /app/uploads /app/index_store

USER appuser

EXPOSE 7860

# Healthcheck — Docker and HF Spaces both honour this
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --log-level info"]