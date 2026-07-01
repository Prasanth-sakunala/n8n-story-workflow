FROM python:3.11-slim

# Install ffmpeg and fonts (Liberation Sans = free Arial-equivalent)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-liberation \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data directories
RUN mkdir -p /data/images /data/audio /data/video /data/credentials /data/clips \
    /data/characters /data/stories /data/music /data/thumbnails

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
