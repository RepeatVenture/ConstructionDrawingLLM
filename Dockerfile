# ==========================================================================
#  BlueprintBid Local LLM – GPU Dockerfile
#
#  Build:
#    docker build -t blueprintbid-llm .
#
#  Run (with NVIDIA GPU):
#    docker run --gpus all -p 8000:8000 \
#      -v $(pwd)/model_cache:/app/model_cache \
#      -e MODEL_BACKEND=qwen2-vl \
#      -e QUANTIZATION=int4 \
#      blueprintbid-llm
# ==========================================================================

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Install Python deps (layer cached until requirements change)
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Model cache directory (mount a volume here for persistence)
RUN mkdir -p /app/model_cache
ENV MODEL_CACHE_DIR=/app/model_cache
ENV HF_HOME=/app/model_cache
ENV TRANSFORMERS_CACHE=/app/model_cache

# Expose server port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start server
CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
