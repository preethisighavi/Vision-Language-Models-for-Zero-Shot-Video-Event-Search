FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# ffmpeg for decoding/thumbnails; fonts so drawtext can label the synthetic seed clips
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD curl -fs http://localhost:8000/api/v1/health || exit 1

# Seeds the libraries on first boot (no-op afterwards), then serves the API.
CMD ["sh", "-c", "python -m seed && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
