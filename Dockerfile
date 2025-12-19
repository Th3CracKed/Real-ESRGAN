FROM runpod/base:0.6.3-cuda11.8.0

# Set Python 3.11 as default
RUN ln -sf $(which python3.11) /usr/local/bin/python && \
    ln -sf $(which python3.11) /usr/local/bin/python3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /requirements.txt
RUN uv pip install --upgrade -r /requirements.txt --no-cache-dir --system

# Create weights directory
RUN mkdir -p /weights

# Download Real-ESRGAN model weights using wget (avoids basicsr import issues)
RUN wget -O /weights/RealESRGAN_x4plus.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

# Download anime model as well
RUN wget -O /weights/realesr-animevideov3.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth

# Copy handler
COPY handler.py /handler.py

# Set working directory
WORKDIR /

# Run the handler
CMD python -u /handler.py
