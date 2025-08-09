# DreamBooth Flux - RunPod Serverless API

A high-performance DreamBooth fine-tuning API optimized for maximum likeness fidelity, deployed on RunPod serverless infrastructure.

## Features

- **Full DreamBooth Fine-Tuning**: Complete UNet + Text Encoder training for maximum likeness
- **Optimized for Headshots**: Specifically tuned for facial likeness preservation
- **Serverless Deployment**: Ready-to-deploy on RunPod serverless
- **High-Quality Training**: Implements best practices for subject fidelity
- **Memory Optimized**: Efficient memory usage with gradient checkpointing and mixed precision

## Architecture

This implementation follows the [DreamBooth Flux](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux.md) approach with full fine-tuning for maximum likeness:

- **UNet Training**: Full fine-tuning of the diffusion model
- **Text Encoder Training**: Critical for facial likeness and prompt adherence
- **Prior Preservation**: Maintains class knowledge while learning subject-specific features
- **Mixed Precision**: FP16 training for memory efficiency
- **Gradient Checkpointing**: Memory optimization without sacrificing quality

## Setup

### Quick Start for RunPod Serverless

1. **Get API Keys**: See [RUNPOD_SETUP.md](RUNPOD_SETUP.md) for detailed setup instructions
2. **Setup Environment**: Run `python setup_accelerate.py` to configure dependencies and accelerate
3. **Build & Deploy**: Run `deploy.bat` (Windows) or `deploy.sh` (Linux/Mac)
4. **Configure RunPod**: Follow the setup guide to create your serverless endpoint

### Manual Setup

#### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install diffusers from source for latest DreamBooth Flux
pip install git+https://github.com/huggingface/diffusers.git

# Configure accelerate
accelerate config default
```

#### 2. Build Docker Image

```bash
docker build -t dreambooth-flux .
```

#### 2. Deploy to RunPod Serverless

1. Push your image to a container registry (Docker Hub, GitHub Container Registry, etc.)
2. Create a new RunPod serverless endpoint
3. Configure with:
   - **Container Image**: Your pushed image
   - **GPU Type**: A100 (40GB) or H100 (80GB) recommended
   - **Min Memory**: 32GB
   - **Max Memory**: 64GB
   - **Handler**: `handler.py`
   - **Environment Variables**: 
     - `HF_TOKEN`: Your Hugging Face token
     - `RUNPOD_API_KEY`: Your RunPod API key
     - `RUNPOD_ENDPOINT_URL`: Your endpoint URL

## API Usage

### Training a Model

```python
import requests
import json

# Training request
training_data = {
    "type": "train",
    "instance_prompt": "a photo of sks person",
    "class_prompt": "a photo of a person",
    "image_urls": [
        "https://example.com/photo1.jpg",
        "https://example.com/photo2.jpg",
        "https://example.com/photo3.jpg",
        # ... more images (10-15 recommended)
    ],
    "max_train_steps": 800,
    "learning_rate": 5e-6,
    "train_text_encoder": True,
    "resolution": 512
}

response = requests.post(
    "https://your-runpod-endpoint.runpod.net",
    json={"input": training_data}
)

result = response.json()
print(f"Training status: {result['status']}")
print(f"Model path: {result.get('model_path')}")
```

### Generating Images

```python
# Generation request
generation_data = {
    "type": "generate",
    "prompt": "a photo of sks person in a business suit",
    "negative_prompt": "blurry, bad quality, distorted, low resolution",
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "width": 512,
    "height": 512,
    "num_images": 4,
    "model_path": "/tmp/dreambooth_output"  # Path from training
}

response = requests.post(
    "https://your-runpod-endpoint.runpod.net",
    json={"input": generation_data}
)

result = response.json()
if result["status"] == "success":
    for i, image_data in enumerate(result["images"]):
        # Decode base64 image
        import base64
        from PIL import Image
        import io
        
        image_bytes = base64.b64decode(image_data["image"])
        image = Image.open(io.BytesIO(image_bytes))
        image.save(f"generated_image_{i}.png")
```

## Training Parameters

### Critical Parameters for Maximum Likeness

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `train_text_encoder` | `True` | **Critical** - Enables text encoder training for facial likeness |
| `max_train_steps` | `800` | Optimal training length for headshots |
| `learning_rate` | `5e-6` | Conservative learning rate for stability |
| `resolution` | `512` | Standard resolution for training |
| `train_batch_size` | `1` | Single image per batch for precision |
| `gradient_accumulation_steps` | `1` | No accumulation for simplicity |

### Advanced Parameters

```python
advanced_training = {
    "type": "train",
    "instance_prompt": "a photo of sks person",
    "class_prompt": "a photo of a person",
    "image_urls": [...],
    "max_train_steps": 800,
    "learning_rate": 5e-6,
    "train_text_encoder": True,
    "resolution": 512,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "lr_scheduler": "constant",
    "lr_warmup_steps": 0,
    "mixed_precision": "fp16",
    "gradient_checkpointing": True,
    "use_8bit_adam": True,
    "seed": 42,
    "num_class_images": 50,
    "save_steps": 100,
    "save_total_limit": 2
}
```

## Best Practices for Maximum Likeness

### 1. Training Image Quality
- **Quantity**: 10-15 high-quality images minimum
- **Diversity**: Different angles, lighting, backgrounds, expressions
- **Quality**: High resolution, clear, well-lit photos
- **Consistency**: Avoid filters, excessive makeup, or non-representative styles

### 2. Prompt Engineering
- **Unique Token**: Always use a unique identifier (e.g., "sks")
- **Consistent Format**: "a photo of [token] person" during training and inference
- **Avoid Generics**: Never use generic prompts like "a person"

### 3. Training Configuration
- **Full Fine-Tuning**: Always enable `train_text_encoder=True`
- **Conservative Learning**: Use 5e-6 learning rate
- **Adequate Steps**: 800 steps minimum for headshots
- **No Shortcuts**: Avoid aggressive optimizations that sacrifice quality

### 4. Validation
- **Benchmark Testing**: Generate images with identical prompts to training samples
- **Face Recognition**: Use embedding distance metrics for quantitative assessment
- **Visual Comparison**: Side-by-side comparison with source images

## Performance Optimization

### Memory Management
- **Gradient Checkpointing**: Enabled by default
- **Mixed Precision**: FP16 training for efficiency
- **8-bit Adam**: Memory-efficient optimizer
- **Batch Size**: Single image per batch for precision

### GPU Requirements
- **Minimum**: 24GB VRAM (RTX 3090, A100)
- **Recommended**: 40GB+ VRAM (A100, H100)
- **Memory Usage**: ~20-30GB during training

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size to 1
   - Enable gradient checkpointing
   - Use 8-bit Adam optimizer

2. **Poor Likeness Quality**
   - Ensure `train_text_encoder=True`
   - Increase training steps to 800+
   - Use more diverse training images
   - Check image quality and consistency

3. **Training Instability**
   - Reduce learning rate to 3e-6
   - Increase warmup steps
   - Use constant learning rate scheduler

### Monitoring Training

```python
# Check training progress
status_data = {
    "type": "status",
    "model_path": "/tmp/dreambooth_output"
}

response = requests.post(
    "https://your-runpod-endpoint.runpod.net",
    json={"input": status_data}
)
```

## Model Comparison

| Approach | Likeness Quality | Training Speed | Memory Usage |
|----------|-----------------|----------------|--------------|
| LoRA | Decent | Fast | Low |
| Full DreamBooth (UNet only) | Excellent | Medium | High |
| **DreamBooth + Text Encoder** | **Best** | Slow | Very High |

## License

This project is based on the [DreamBooth Flux](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux.md) implementation from Hugging Face Diffusers.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the [DreamBooth Flux documentation](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux.md)
3. Ensure you're using the recommended hardware specifications
