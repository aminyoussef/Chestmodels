FROM python:3.11-slim-bookworm

WORKDIR /app

# MNE / scipy runtime libs
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .

# CPU-only PyTorch keeps the image smaller on a typical VPS
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-docker.txt

COPY api.py .
COPY eeg_web_output/ eeg_web_output/

ENV PYTHONUNBUFFERED=1 \
    EEG_MODEL_DIR=/app/eeg_web_output \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
