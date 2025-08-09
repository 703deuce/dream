FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Set working directory
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip

# Install core dependencies first
RUN pip install --no-cache-dir packaging setuptools wheel

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional performance optimizations
RUN pip install --no-cache-dir \
    ninja \
    flash-attn \
    triton

# Clone and install diffusers from source for latest DreamBooth Flux support
RUN git clone https://github.com/huggingface/diffusers && \
    cd diffusers && \
    pip install -e . && \
    cd examples/dreambooth && \
    pip install -r requirements_flux.txt && \
    pip install peft>=0.6.0 && \
    cd /workspace

# Download dog example dataset for testing
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('diffusers/dog-example', local_dir='/workspace/dog', repo_type='dataset', ignore_patterns='.gitattributes')"

# Copy application files
COPY handler.py .
COPY train_dreambooth.py .

# Create necessary directories
RUN mkdir -p /tmp/instance_data /tmp/dreambooth_output /tmp/class_data

# Set environment variables for better performance
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_LAUNCH_BLOCKING=0
ENV TORCH_CUDNN_V8_API_ENABLED=1

# Environment variables for API keys and tokens (will be set by RunPod)
ENV HF_TOKEN=""
ENV RUNPOD_API_KEY=""
ENV RUNPOD_ENDPOINT_URL=""

# Expose port (RunPod will handle this)
EXPOSE 8000

# Set the default command
CMD ["python", "handler.py"]
