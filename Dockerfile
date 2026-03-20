# SafeVision - Safety Goggles Detection
FROM python:3.11-slim

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install opencv-python-headless FIRST so ultralytics doesn't pull the GUI version
RUN pip install --no-cache-dir opencv-python-headless>=4.8.0

# Install remaining dependencies (ultralytics will skip opencv since cv2 already exists)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads results

# Expose port
EXPOSE 5000

# Use Gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
