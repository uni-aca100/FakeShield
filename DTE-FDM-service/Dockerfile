FROM zhipeixu/dte-fdm:v1.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/conda/bin:${PATH}"



WORKDIR /workspace/fakeshield
COPY requirements.txt .

# Install PyTorch, on older gpu hardware, you may skip the torch installation and use the pre-installed version in the base image.
# arch list: ['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
RUN /opt/conda/bin/conda create -y -n py311 python=3.11 pip
RUN /opt/conda/bin/conda run -n py311 python -m pip install --no-cache-dir torch torchvision torchaudio
RUN /opt/conda/bin/conda run -n py311 python -m pip install --no-cache-dir -r requirements.txt

EXPOSE 8001
CMD ["/opt/conda/bin/conda", "run", "--no-capture-output", "-n", "py311", "uvicorn", "DTE-FDM-service.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]