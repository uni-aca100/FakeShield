FROM zhipeixu/dte-fdm:v1.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/conda/bin:${PATH}"

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/fakeshield

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY DTE_FDM/ ./DTE_FDM/
COPY DTE-FDM-service/ ./DTE-FDM-service/

EXPOSE 8001

HEALTHCHECK --interval=80s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "DTE-FDM-service.main:app", "--host", "0.0.0.0", "--port", "8001"]