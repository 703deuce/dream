FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies (Python 3.9+ for bitsandbytes compatibility)
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Verify Python version
RUN python --version

# Set working directory
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip

# Install core dependencies first
RUN pip install --no-cache-dir packaging setuptools wheel

# Install PyTorch with CUDA support (confirmed compatible versions for CUDA 11.8)
# Use compatible versions: torch 2.0.1, torchvision 0.15.2, torchaudio 2.0.2
RUN pip install --no-cache-dir torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir torchaudio==2.0.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch installation and CUDA compatibility
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Install basic dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install performance optimizations (compatible with PyTorch 2.0.1)
RUN pip install --no-cache-dir xformers==0.0.20

# Install bitsandbytes for 8-bit optimizer support (latest stable version)
RUN pip install --no-cache-dir bitsandbytes>=0.46.0

# Install FLUX-specific dependencies
RUN pip install --no-cache-dir prodigyopt

# Install additional performance optimizations (compatible versions for PyTorch 2.0.1)
RUN pip install --no-cache-dir ninja
RUN pip install --no-cache-dir 'triton>=2.0.0,<2.1.0'
RUN pip install --no-cache-dir 'flash-attn>=0.2.4,<0.3.0'

# Install diffusers from local source for latest DreamBooth Flux support
COPY diffusers/ ./diffusers/

# Verify diffusers folder structure
RUN ls -la /workspace/diffusers/

# Go to /workspace/diffusers/ and run pip install -e .
RUN cd /workspace/diffusers && pip install -e .

# Then cd in the example folder /workspace/diffusers/dreambooth/ and run pip install -r requirements_flux.txt
RUN cd /workspace/diffusers/examples/dreambooth && pip install -r requirements_flux.txt

# Return to workspace
RUN cd /workspace

# Note: We're doing full fine-tuning, not LoRA, so PEFT is not needed

# Download dog example dataset for testing (from FLUX README)
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

# Environment variables for API keys and tokens (set by RunPod at runtime)
# Note: These are not set in Dockerfile for security reasons
# RunPod will inject them as environment variables when the container starts

# Expose port (RunPod will handle this)
EXPOSE 8000

# Set the default command
CMD ["python", "handler.py"]
