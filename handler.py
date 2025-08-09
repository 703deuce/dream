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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "black-forest-labs/FLUX.1-dev"  # FLUX model for DreamBooth Flux
        self.pipeline = None
        self.trained_model_path = None
        
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
        """Train DreamBooth model with full fine-tuning"""
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
            
            # Training parameters for FLUX DreamBooth
            training_args = {
                "pretrained_model_name_or_path": self.model_id,
                "instance_data_dir": instance_data_dir,
                "output_dir": output_dir,
                "instance_prompt": instance_prompt,
                "class_prompt": class_prompt,
                "resolution": 1024,  # FLUX uses 1024 resolution
                "train_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "max_train_steps": 500,  # FLUX example uses 500 steps
                "learning_rate": 1.0,  # FLUX uses 1.0 learning rate
                "lr_scheduler": "constant",
                "lr_warmup_steps": 0,
                "mixed_precision": "bf16",  # FLUX uses bf16
                "guidance_scale": 1,
                "optimizer": "prodigy",  # FLUX uses Prodigy optimizer
                "seed": 0,
                "save_steps": 100,
                "save_total_limit": 2,
            }
            
            # Run training using accelerate
            from accelerate import Accelerator
            from accelerate.utils import set_seed
            
            accelerator = Accelerator(
                gradient_accumulation_steps=training_args["gradient_accumulation_steps"],
                mixed_precision=training_args["mixed_precision"],
                log_with="tensorboard",
                project_dir=output_dir,
            )
            
            set_seed(training_args["seed"])
            
            # Import and run FLUX training script
            from train_dreambooth_flux import main as train_main
            train_main(training_args, accelerator)
            
            return {
                "status": "success",
                "message": "DreamBooth training completed successfully",
                "output_dir": output_dir,
                "model_path": output_dir
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Training failed: {str(e)}"
            }
    
    def generate_image(self, job_input: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image using trained model"""
        try:
            prompt = job_input.get("prompt", "a photo of sks person")
            negative_prompt = job_input.get("negative_prompt", "blurry, bad quality, distorted")
            num_inference_steps = job_input.get("num_inference_steps", 50)
            guidance_scale = job_input.get("guidance_scale", 7.5)
            width = job_input.get("width", 512)
            height = job_input.get("height", 512)
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
