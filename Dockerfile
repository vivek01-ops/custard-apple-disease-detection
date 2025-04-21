FROM python:3.10-slim

WORKDIR /app

# Add system dependencies required for many packages like TensorFlow
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Upgrade pip and install with retries and increased timeout
RUN pip install --upgrade pip \
    && pip install --default-timeout=300 --retries=10 -r requirements.txt

COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
