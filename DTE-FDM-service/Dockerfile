FROM zhipeixu/dte-fdm:v1.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/conda/bin:${PATH}"

WORKDIR /workspace/fakeshield

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

EXPOSE 8001

CMD ["uvicorn", "DTE-FDM-service.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]