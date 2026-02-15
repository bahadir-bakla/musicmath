# MusicMath DNA – Backend (FastAPI) + Celery Worker
FROM python:3.11-slim

WORKDIR /app

# gcc for numpy; curl for healthcheck; libopenblas for faster scipy (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

ENV DATA_DIR=/data
ENV PYTHONPATH=/data:/app

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
