# Dockerfile (Render / CPU-friendly)
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps used by OpenCV / gTTS
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libsm6 \
    libxext6 \
    libgl1 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps (include tensorflow in requirements)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r /app/requirements.txt

# Copy only app code and assets (do NOT COPY the .h5 model file)
COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "test6.py", "--server.port=8501", "--server.address=0.0.0.0"]
