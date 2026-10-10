FROM python:3.12-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements/requirements.txt runpod boto3
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app QWEN_DATA_DIR=/runpod-volume/qwen-image-studio
CMD ["python", "runpod/handler.py"]
