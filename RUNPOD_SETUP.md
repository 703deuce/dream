# RunPod Serverless Setup Guide

This guide walks you through setting up the DreamBooth Flux API on RunPod serverless infrastructure.

## Prerequisites

1. **RunPod Account**: Sign up at [runpod.io](https://runpod.io)
2. **GitHub Account** with Container Registry access
3. **Hugging Face Account** with API token
4. **RunPod API Key**

## Step 1: Get Your API Keys and Tokens

### Hugging Face Token
1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Create a new token with "read" permissions
3. Copy the token (starts with `hf_`)

### RunPod API Key
1. Go to [RunPod Console](https://runpod.io/console)
2. Navigate to "Account" → "API Keys"
3. Create a new API key
4. Copy the key

## Step 2: Build and Push Docker Image

### Update Configuration
The deployment scripts are already configured to use GitHub Container Registry (`ghcr.io/703deuce`).

If you need to change the registry, edit `deploy.bat` (Windows) or `deploy.sh` (Linux/Mac):

```bash
# Change this line in the script
set REGISTRY=ghcr.io/your-github-username
```

### Run Deployment Script

**Windows:**
```powershell
.\deploy.bat
```

**Linux/Mac:**
```bash
chmod +x deploy.sh
./deploy.sh
```

## Step 3: Configure RunPod Serverless Endpoint

### Create New Endpoint
1. Go to [RunPod Console](https://runpod.io/console)
2. Click "New Endpoint"
3. Select "Serverless"

### Basic Configuration
- **Name**: `dreambooth-flux-api`
- **Container Image**: `ghcr.io/703deuce/dreambooth-flux:latest`
- **Handler**: `handler.py`

### Hardware Configuration
- **GPU**: A100 (40GB) or H100 (80GB) - **Required for full fine-tuning**
- **Min Memory**: 32GB
- **Max Memory**: 64GB
- **Min GPU Count**: 1
- **Max GPU Count**: 1

### Environment Variables
Add these environment variables in the RunPod console:

```
HF_TOKEN=hf_your_huggingface_token_here
RUNPOD_API_KEY=your_runpod_api_key_here
RUNPOD_ENDPOINT_URL=https://your-endpoint-id.runpod.net
```

### Advanced Settings
- **Idle Timeout**: 300 seconds (5 minutes)
- **Flash Boot**: Enabled
- **Volume**: None (ephemeral storage)
- **Network Volume**: None

## Step 4: Deploy and Test

### Deploy Endpoint
1. Click "Deploy" to create the endpoint
2. Wait for the container to build and start (5-10 minutes)
3. Note the endpoint URL (e.g., `https://abc123.runpod.net`)

### Update Test Configuration
Edit `test_api.py`:

```python
# Replace with your actual values
api_url = "https://your-endpoint-id.runpod.net"
api_key = "your_runpod_api_key_here"
```

## Step 5: API Usage Examples

### Training a Model
```python
import requests

# Training request
training_data = {
    "type": "train",
    "instance_prompt": "a photo of sks person",
    "class_prompt": "a photo of a person",
    "image_urls": [
        "https://example.com/photo1.jpg",
        "https://example.com/photo2.jpg",
        # ... more images (10-15 recommended)
    ],
    "max_train_steps": 800,
    "learning_rate": 5e-6,
    "train_text_encoder": True,
    "resolution": 512
}

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.post(
    api_url,
    json={"input": training_data},
    headers=headers,
    timeout=1800  # 30 minutes for training
)

result = response.json()
print(f"Training status: {result['status']}")
```

### Generating Images
```python
# Generation request
generation_data = {
    "type": "generate",
    "prompt": "a photo of sks person in a business suit",
    "negative_prompt": "blurry, bad quality, distorted",
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "width": 512,
    "height": 512,
    "num_images": 4,
    "model_path": "/tmp/dreambooth_output"  # From training response
}

response = requests.post(
    api_url,
    json={"input": generation_data},
    headers=headers,
    timeout=300
)

result = response.json()
if result["status"] == "success":
    for i, image_data in enumerate(result["images"]):
        # Decode and save image
        import base64
        from PIL import Image
        import io
        
        image_bytes = base64.b64decode(image_data["image"])
        image = Image.open(io.BytesIO(image_bytes))
        image.save(f"generated_image_{i}.png")
```

## Step 6: Monitoring and Troubleshooting

### Check Endpoint Status
- Go to RunPod Console → Your Endpoint
- Monitor logs in real-time
- Check GPU utilization and memory usage

### Common Issues

**Out of Memory Errors:**
- Reduce `train_batch_size` to 1
- Enable `gradient_checkpointing` (already enabled)
- Use `use_8bit_adam` (already enabled)

**Training Failures:**
- Check that `HF_TOKEN` is valid
- Ensure image URLs are accessible
- Verify GPU has sufficient VRAM (24GB+)

**Slow Training:**
- This is normal for full DreamBooth fine-tuning
- Training takes 15-30 minutes typically
- Use A100 or H100 for best performance

## Step 7: Production Considerations

### Scaling
- RunPod serverless automatically scales based on demand
- Each request gets a fresh container instance
- No persistent storage between requests

### Cost Optimization
- Use appropriate idle timeout (5-10 minutes)
- Monitor usage in RunPod console
- Consider reserved instances for high-volume usage

### Security
- Keep API keys secure
- Use HTTPS for all requests
- Consider IP whitelisting if needed

## Support

For issues:
1. Check RunPod endpoint logs
2. Verify all environment variables are set
3. Ensure GPU requirements are met
4. Contact RunPod support if needed

## Next Steps

Once deployed:
1. Test with a small training set
2. Validate image quality and likeness
3. Scale up for production use
4. Monitor costs and performance
