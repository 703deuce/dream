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

# Set working directory
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip

# Install core dependencies first
RUN python -m pip install --no-cache-dir packaging setuptools wheel

# Clean up any existing PyTorch packages to prevent version conflicts
RUN python -m pip uninstall -y torch torchvision torchaudio || true

# Install PyTorch with CUDA support (confirmed compatible versions for CUDA 11.8)
# Use compatible versions: torch 2.0.1, torchvision 0.15.2, torchaudio 2.0.2
# All packages must use the same CUDA version to avoid compatibility issues
# Using --force-reinstall to ensure clean installation and prevent version conflicts
RUN python -m pip install --force-reinstall --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118



# Install basic dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt

# Fix NumPy version compatibility with PyTorch 2.0.1
# PyTorch 2.0.1 was compiled with NumPy 1.x, so we need to use a compatible version
# Use numpy==1.26.4 which is tested and known to work with PyTorch 2.x and CUDA 11.8
RUN python -m pip install --force-reinstall --no-cache-dir "numpy==1.26.4"

# Clean up any potential conflicting packages and ensure clean environment
RUN python -m pip uninstall -y transformers || true
RUN python -m pip cache purge

# Install performance optimizations (compatible with PyTorch 2.0.1)
RUN python -m pip install --no-cache-dir xformers==0.0.20

# Install bitsandbytes for 8-bit optimizer support (latest stable version)
RUN python -m pip install --no-cache-dir bitsandbytes>=0.46.0

# Install FLUX-specific dependencies
RUN python -m pip install --no-cache-dir prodigyopt

# Install additional performance optimizations (compatible versions for PyTorch 2.0.1)
RUN python -m pip install --no-cache-dir ninja
RUN python -m pip install --no-cache-dir 'triton>=2.0.0,<2.1.0'
RUN python -m pip install --no-cache-dir 'flash-attn>=0.2.4,<0.3.0'

# Install diffusers from local source for latest DreamBooth Flux support
# Copy the entire dream repository structure
COPY . .

# Go to /workspace/diffusers/ and run pip install -e .
RUN cd /workspace/diffusers && python -m pip install -e .

# Then cd in the example folder /workspace/diffusers/examples/dreambooth/ and run pip install -r requirements_flux.txt
RUN cd /workspace/diffusers/examples/dreambooth && python -m pip install -r requirements_flux.txt

# Return to workspace
RUN cd /workspace

# Install additional required dependencies for CLIP and transformers AFTER requirements_flux.txt
# This ensures we get the latest transformers version (>=4.41.2) from requirements_flux.txt
# Use force-reinstall to avoid any corrupted installations
RUN python -m pip install --upgrade --force-reinstall "transformers[torch,vision,audio]>=4.41.2"

# Also install specific CLIP dependencies explicitly
RUN python -m pip install --no-cache-dir "clip @ git+https://github.com/openai/CLIP.git"

# Install a specific transformers version known to have all CLIP modules
RUN python -m pip install --no-cache-dir "transformers==4.42.0"



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
