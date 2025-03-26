# Use official Python 3.9 slim image as the base
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy your script and model checkpoint to the container
COPY glasses_detection_api.py glasses_detector_checkpoint.pth ./

# Install system dependencies (for OpenCV, MediaPipe, etc.)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    fastapi \
    uvicorn \
    opencv-python-headless \
    mediapipe \
    numpy \
    pillow

# Expose port 8000 for the FastAPI server
EXPOSE 8000

# Command to run the application
CMD ["python", "glasses_detection_api.py"]