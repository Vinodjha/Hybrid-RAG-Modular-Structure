FROM python:3.11-slim-bookworm

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# Runtime deps for faiss-cpu (OpenMP) and common wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Install deps (pin CPU torch first to avoid CUDA pulls)
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 && \
    pip install --no-cache-dir -r requirements.txt

# Copy app code last for better layer caching
COPY app/ /app/app/

EXPOSE 8000
# Add --workers for prod later (e.g., 2 or 3)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
