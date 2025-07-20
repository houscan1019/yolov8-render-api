FROM python:3.9-slim

# Install system dependencies required for OpenCV headless
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    libavcodec58 \
    libavformat58 \
    libswscale5 \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for headless operation
ENV OPENCV_IO_ENABLE_OPENEXR=0
ENV QT_QPA_PLATFORM=offscreen
ENV MPLBACKEND=Agg
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create static directory
RUN mkdir -p static/processed

# Expose port (Railway will set PORT environment variable)
EXPOSE $PORT

# Run the application
CMD ["python", "main.py"]
