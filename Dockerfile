ARG BASE_IMAGE="nvcr.io/nvidia/cuda"
ARG BASE_IMAGE_TAG="12.4.1-runtime-ubuntu22.04"

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} AS base

# Install uv for fast Python dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libopus-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app/moshi/

# Copy project files
COPY moshi/ /app/moshi/

# Create virtualenv with Python 3.12 (needed for audioop, removed in 3.13)
RUN uv venv /app/moshi/.venv --python 3.12
RUN uv sync

# Create directories
RUN mkdir -p /app/ssl /app/recordings

# Expose port 8080 (Cloud Run default)
EXPOSE 8080

# Health check for container orchestrators
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD /app/moshi/.venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Runtime environment
ENV DEVICE=cuda
ENV HF_HOME=/root/.cache/huggingface
ENV RECORDINGS_DIR=/app/recordings

# Enable hf_transfer for fast model downloads (Rust-based, 5-10x speedup)
ENV HF_HUB_ENABLE_HF_TRANSFER=1

ENTRYPOINT []
CMD ["/app/moshi/.venv/bin/python", "-m", "moshi.app", "--host", "0.0.0.0", "--port", "8080"]
