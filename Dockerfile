# Dockerfile (GPU-ready)
FROM tensorflow/tensorflow:2.12.0-gpu

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Small system deps used by OpenCV/gTTS
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libsm6 \
    libxext6 \
    libgl1 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements (tensorflow NOT listed here)
COPY requirements.txt /app/requirements.txt

# Install python deps (no tensorflow in requirements)
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r /app/requirements.txt

# Copy app code
COPY . /app

EXPOSE 8501

# Run your script (change if needed)
CMD ["streamlit","run","test6.py","--server.port=8501","--server.address=0.0.0.0"]

