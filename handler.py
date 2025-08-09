import os
import json
import base64
import io
import torch
import diffusers
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import logging
from PIL import Image
import requests
from typing import Dict, Any, List
import tempfile
import shutil
from pathlib import Path
from huggingface_hub import login as hf_login
import subprocess
import sys

# Configure Accelerate for serverless RunPod environment (non-interactive)
try:
    from accelerate.utils import write_basic_config
    write_basic_config()
    print("✅ Accelerate configuration initialized for serverless environment")
except Exception as e:
    print(f"⚠️  Warning: Failed to initialize Accelerate config: {e}")

# Configure logging
logging.set_verbosity_info()

# Load environment variables for API keys and tokens
HF_TOKEN = os.getenv("HF_TOKEN")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_URL = os.getenv("RUNPOD_ENDPOINT_URL")

# Login to Hugging Face if token is provided
if HF_TOKEN:
    try:
        hf_login(token=HF_TOKEN)
        print("✅ Successfully logged in to Hugging Face")
    except Exception as e:
        print(f"⚠️  Warning: Failed to login to Hugging Face: {e}")
else:
    print("⚠️  Warning: HF_TOKEN not provided. Some models may not be accessible.")

class DreamBoothFluxHandler:
    def __init__(self):
        # Check GPU availability for RunPod environment
        if torch.cuda.is_available():
            self.device = "cuda"
            print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.device = "cpu"
            print("⚠️  No GPU detected, using CPU (not recommended for training)")
        
        self.model_id = "black-forest-labs/FLUX.1-dev"  # FLUX model for DreamBooth Flux
        self.pipeline = None
        self.trained_model_path = None
        
        # Optional: Configure torch.compile for dramatic speedups (if supported)
        self.enable_torch_compile = os.getenv("ENABLE_TORCH_COMPILE", "false").lower() == "true"
        if self.enable_torch_compile and hasattr(torch, 'compile'):
            print("🚀 Torch compile mode enabled for dramatic speedups")
        elif self.enable_torch_compile:
            print("⚠️  Torch compile requested but not available in this PyTorch version")
        
        print("🎯 Training mode: Full fine-tuning (UNet + Text Encoder) for maximum likeness - NOT LoRA")
        
    def load_model(self, model_path: str = None):
        """Load the trained model or base model"""
        if model_path and os.path.exists(model_path):
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                safety_checker=None
            )
            self.trained_model_path = model_path
        else:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                safety_checker=None
            )
        
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config
        )
        self.pipeline = self.pipeline.to(self.device)
        
    def download_images(self, image_urls: List[str], save_dir: str) -> List[str]:
        """Download training images from URLs"""
        os.makedirs(save_dir, exist_ok=True)
        image_paths = []
        
        for i, url in enumerate(image_urls):
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                image_path = os.path.join(save_dir, f"image_{i:03d}.jpg")
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                image_paths.append(image_path)
                
            except Exception as e:
                print(f"Failed to download image {url}: {e}")
                
        return image_paths
    
    def train_dreambooth(self, job_input: Dict[str, Any]) -> Dict[str, Any]:
        """Train DreamBooth model using command line script for maximum likeness"""
        try:
            # Extract training parameters
            instance_prompt = job_input.get("instance_prompt", "a photo of sks person")
            class_prompt = job_input.get("class_prompt", "a photo of a person")
            instance_data_dir = job_input.get("instance_data_dir", "/tmp/instance_data")
            output_dir = job_input.get("output_dir", "/tmp/dreambooth_output")
            image_urls = job_input.get("image_urls", [])
            
            # Create training directory
            os.makedirs(instance_data_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            # Download training images if URLs provided
            if image_urls:
                self.download_images(image_urls, instance_data_dir)
            
            # Build command line arguments for FLUX DreamBooth FULL FINE-TUNING (not LoRA)
            # This trains both UNet and Text Encoder for maximum likeness
            cmd = [
                     sys.executable,  # Use current Python interpreter
                     "diffusers/examples/dreambooth/train_dreambooth_flux.py",
                "--pretrained_model_name_or_path", self.model_id,
                "--instance_data_dir", instance_data_dir,
                "--output_dir", output_dir,
                "--instance_prompt", instance_prompt,
                "--class_prompt", class_prompt,
                "--resolution", "1024",  # FLUX uses 1024 resolution for best quality
                "--train_batch_size", "1",
                "--gradient_accumulation_steps", "4",
                "--max_train_steps", "1000",  # Increased for better likeness
                "--learning_rate", "1.0",  # FLUX uses 1.0 learning rate with Prodigy
                "--lr_scheduler", "constant",
                "--lr_warmup_steps", "0",
                "--mixed_precision", "bf16",  # FLUX uses bf16
                "--guidance_scale", "1",
                "--optimizer", "prodigy",  # Prodigy optimizer for best results
                "--seed", "0",
                "--save_steps", "100",
                "--save_total_limit", "3",  # Keep more checkpoints for blending
                "--validation_prompt", instance_prompt,  # Validate with instance prompt
                "--validation_epochs", "25",  # Regular validation
                "--num_validation_images", "4",
                "--report_to", "tensorboard",  # Use tensorboard for logging
                "--train_text_encoder",  # CRITICAL: Train text encoder for best likeness
                "--max_sequence_length", "512",  # Support longer prompts
                "--gradient_checkpointing",  # Memory optimization
                "--cache_latents",  # Memory optimization
                "--aspect_ratio_buckets", "672,1568;688,1504;720,1456;752,1392;800,1328;832,1248;880,1184;944,1104;1024,1024;1104,944;1184,880;1248,832;1328,800;1392,752;1456,720;1504,688;1568,672"  # Support different aspect ratios
            ]
            
            print("🚀 Starting FLUX DreamBooth training with maximum likeness settings...")
            print(f"📸 Training on {len(os.listdir(instance_data_dir))} images")
            print(f"🎯 Instance prompt: {instance_prompt}")
            print(f"🔧 Training text encoder: True")
            print(f"📏 Resolution: 1024")
            print(f"⚡ Optimizer: prodigy")
            print(f"🖥️  Command: {' '.join(cmd)}")
            
            # Run the training command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.getcwd()  # Run from current directory
            )
            
            if result.returncode == 0:
                print("✅ Training completed successfully!")
                return {
                    "status": "success",
                    "message": "FLUX DreamBooth training completed successfully with maximum likeness settings",
                    "output_dir": output_dir,
                    "model_path": output_dir,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                print(f"❌ Training failed with return code {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return {
                    "status": "error",
                    "message": f"Training failed with return code {result.returncode}",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode
                }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Training failed: {str(e)}",
                "error_details": str(e)
            }
    
    def generate_image(self, job_input: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image using trained model"""
        try:
            prompt = job_input.get("prompt", "a photo of sks person")
            negative_prompt = job_input.get("negative_prompt", "blurry, bad quality, distorted, low resolution, ugly, deformed")
            num_inference_steps = job_input.get("num_inference_steps", 50)
            guidance_scale = job_input.get("guidance_scale", 7.5)
            width = job_input.get("width", 1024)  # Default to FLUX resolution
            height = job_input.get("height", 1024)
            num_images = job_input.get("num_images", 1)
            
            # Load model if not already loaded
            if self.pipeline is None:
                model_path = job_input.get("model_path")
                self.load_model(model_path)
            
            # Generate images
            images = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                num_images_per_prompt=num_images,
            ).images
            
            # Convert images to base64
            image_data = []
            for i, image in enumerate(images):
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode()
                image_data.append({
                    "image": img_str,
                    "index": i
                })
            
            return {
                "status": "success",
                "images": image_data,
                "prompt": prompt,
                "parameters": {
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "width": width,
                    "height": height
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Generation failed: {str(e)}"
            }

# Global handler instance
handler = DreamBoothFluxHandler()

def handler(event, context):
    """Main RunPod serverless handler"""
    try:
        # Parse input
        job_input = event.get("input", {})
        job_type = job_input.get("type", "generate")
        
        if job_type == "train":
            result = handler.train_dreambooth(job_input)
        elif job_type == "generate":
            result = handler.generate_image(job_input)
        else:
            result = {
                "status": "error",
                "message": f"Unknown job type: {job_type}"
            }
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Handler error: {str(e)}"
        }
