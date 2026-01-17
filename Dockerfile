# PDF to JSON Converter - Dockerfile for Railway Deployment
# Uses Python 3.11 with Docling dependencies

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies required by Docling
# - poppler-utils: PDF rendering
# - tesseract-ocr: OCR capabilities
# - libgl1: OpenCV dependency
# - libglib2.0-0: GLib library
# - libpango-1.0-0, libpangocairo-1.0-0: Text rendering
# - Additional libraries for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Pre-download OCR models while still running as root
# This prevents permission errors when the app tries to download them later
RUN python -c "from rapidocr import RapidOCR; ocr = RapidOCR()" || true

# Make the rapidocr models directory world-writable as fallback
RUN chmod -R 777 /usr/local/lib/python3.11/site-packages/rapidocr/models 2>/dev/null || true

# Copy application code
COPY app/ ./app/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

# Set environment for model caching in user-writable directory
ENV HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch \
    XDG_CACHE_HOME=/app/.cache

# Create cache directories with proper permissions
RUN mkdir -p /app/.cache/huggingface /app/.cache/torch && \
    chown -R appuser:appuser /app/.cache

USER appuser

# Expose port (Railway will set PORT env variable)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Run the application with increased timeout for PDF processing
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 300"]
