# Production Dockerfile for Car Price Prediction MLOps Platform
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application source code and artifacts
COPY data/ data/
COPY src/ src/
COPY app/ app/
COPY artifacts/ artifacts/
COPY streamlit_app.py .
COPY run_pipeline.py .
COPY README.md .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default command: Run FastAPI Production REST API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
