FROM u2470277607/nvcr.io-nvidia-vllm:26.01-py3-mmcv-12.0-PTX-v1.0

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /workspace/fakeshield

COPY requirements.txt ./
RUN python -m pip install -r requirements.txt --no-cache-dir

EXPOSE 8002

CMD ["uvicorn", "MFLM_service.main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]