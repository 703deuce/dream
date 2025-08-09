FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies (Python 3.8 with compatible numpy/scipy versions)
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
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
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# Verify Python version
RUN python --version

# Set working directory
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip

# Install core dependencies first
RUN pip install --no-cache-dir packaging setuptools wheel

# Install PyTorch with CUDA support (exact matching versions for CUDA 11.8)
RUN pip install --no-cache-dir torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install basic dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install performance optimizations with exact versions (compatible with torch 2.0.0)
RUN pip install --no-cache-dir xformers==0.0.20

# Install bitsandbytes with exact version
RUN pip install --no-cache-dir bitsandbytes==0.41.0

# Install FLUX-specific dependencies
RUN pip install --no-cache-dir prodigyopt

# Install additional performance optimizations with exact versions
RUN pip install --no-cache-dir ninja flash-attn==0.2.4 triton==2.0.0

# Install diffusers from local source for latest DreamBooth Flux support
COPY diffusers/ ./diffusers/

# Go to /workspace/diffusers/ and run pip install -e .
RUN cd /workspace/diffusers && pip install -e .

# Then cd in the example folder /workspace/diffusers/dreambooth/ and run pip install -r requirements_flux.txt
RUN cd /workspace/diffusers/examples/dreambooth && pip install -r requirements_flux.txt

# Return to workspace
RUN cd /workspace

# Initialize Accelerate with default configuration for non-interactive environment
RUN accelerate config default

# Install peft with correct version after diffusers requirements
RUN pip install --no-cache-dir peft>=0.17.0

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
