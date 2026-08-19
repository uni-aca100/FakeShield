FROM zhipeixu/mflm:v1.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/conda/bin:${PATH}"

WORKDIR /workspace/fakeshield

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Compile MMCV specific to MFLM requirements
RUN git clone https://github.com/open-mmlab/mmcv.git /tmp/mmcv && \
    cd /tmp/mmcv && \
    git checkout v1.4.7 && \
    MMCV_WITH_OPS=1 pip install --no-cache-dir -e . && \
    rm -rf /tmp/mmcv

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

EXPOSE 8002

CMD ["uvicorn", "MFLM-service.main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]