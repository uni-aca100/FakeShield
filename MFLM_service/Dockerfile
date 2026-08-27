FROM nvcr.io/nvidia/vllm:26.01-py3

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /workspace/fakeshield

COPY requirements.txt ./
RUN python -m pip install -r requirements.txt --no-cache-dir

EXPOSE 8002

CMD ["uvicorn", "MFLM_service.main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]