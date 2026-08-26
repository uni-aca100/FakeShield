FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST="12.0" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    python3.11 \
    python3.11-venv \
    python3-pip \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/fakeshield

COPY requirements.txt ./

RUN python -m pip install --upgrade pip setuptools wheel

# Install the modern CUDA 12.8 PyTorch stack required for Blackwell (sm_120)
RUN python -m pip install torch torchvision torchaudio --no-cache-dir

# Install project runtime dependencies. Flash-attn is intentionally left out here because
# it often requires a target-specific CUDA build and may fail on a first pass.
RUN python -m pip install -r requirements.txt

# Install flash-attn separately to avoid build issues on some CUDA architectures
RUN python -m pip install flash-attn --no-cache-dir


EXPOSE 8001

CMD ["uvicorn", "DTE-FDM-service.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
