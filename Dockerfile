FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies needed by pdf2image, OpenCV, and common build tools
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        poppler-utils \
        pkg-config \
        libgl1 \
        libglib2.0-0 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application
COPY . /app

# Create common output folders for mounting / permission purposes
RUN mkdir -p /app/outputs /app/outputs/preprocessed /app/outputs/runs /app/models

# Default environment variables
ENV OLLAMA_ENDPOINT=http://host.docker.internal:11434
ENV POPPLER_PATH=/usr/bin

# Use orchestrator as the container entrypoint; PDF path should be provided when running
ENTRYPOINT ["python", "-u", "tools/orchestrator.py"]

# Default command prints usage when no args provided
CMD ["--help"]
