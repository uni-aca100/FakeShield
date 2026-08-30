FROM python:3.11-slim

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "gateway-service.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]