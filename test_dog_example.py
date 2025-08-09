#!/usr/bin/env python3
"""
Test script for DreamBooth Flux Dog Example
Demonstrates training and generation using the dog dataset
"""

import requests
import json
import base64
import io
from PIL import Image
import time

def test_dog_training(api_url, headers):
    """Test training with the dog example dataset"""
    print("🐕 Testing DreamBooth Flux with Dog Example...")
    
    # Training request using the dog dataset
    training_data = {
        "type": "train",
        "instance_prompt": "a photo of sks dog",
        "class_prompt": "a photo of a dog",
        "instance_data_dir": "/workspace/dog",  # Pre-downloaded dog dataset
        "output_dir": "/tmp/dreambooth_output",
        "max_train_steps": 100,  # Reduced for testing
        "learning_rate": 1.0,
        "resolution": 1024,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "mixed_precision": "bf16",
        "guidance_scale": 1,
        "optimizer": "prodigy",
        "lr_scheduler": "constant",
        "lr_warmup_steps": 0,
        "seed": 0
    }

    try:
        print("🚀 Starting FLUX training with dog dataset...")
        response = requests.post(
            api_url,
            json={"input": training_data},
            headers=headers,
            timeout=1800  # 30 minutes for training
        )

        result = response.json()
        print(f"Training status: {result['status']}")

        if result["status"] == "success":
            print("✅ Dog training test passed!")
            model_path = result.get("model_path")
            print(f"Model saved at: {model_path}")

            # Test generation with trained model
            print("🎨 Testing generation with trained dog model...")
            generation_data = {
                "type": "generate",
                "prompt": "A photo of sks dog in a bucket",
                "negative_prompt": "blurry, bad quality, distorted, low resolution",
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
                "width": 1024,
                "height": 1024,
                "num_images": 1,
                "model_path": model_path
            }

            gen_response = requests.post(
                api_url,
                json={"input": generation_data},
                headers=headers,
                timeout=300
            )

            gen_result = gen_response.json()
            if gen_result["status"] == "success":
                print("✅ Dog generation test passed!")

                # Save generated image
                image_data = gen_result["images"][0]
                image_bytes = base64.b64decode(image_data["image"])
                image = Image.open(io.BytesIO(image_bytes))
                image.save("test_dog_generation.png")
                print("📸 Dog image saved as 'test_dog_generation.png'")

                return True
            else:
                print(f"❌ Dog generation failed: {gen_result.get('message', 'Unknown error')}")
                return False
        else:
            print(f"❌ Dog training failed: {result.get('message', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Dog test error: {str(e)}")
        return False

def test_base_generation(api_url, headers):
    """Test base FLUX model generation"""
    print("🎨 Testing base FLUX model generation...")

    generation_data = {
        "type": "generate",
        "prompt": "A photo of a cute dog in a garden",
        "negative_prompt": "blurry, bad quality, distorted, low resolution",
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "width": 1024,
        "height": 1024,
        "num_images": 1
    }

    try:
        response = requests.post(
            api_url,
            json={"input": generation_data},
            headers=headers,
            timeout=300
        )

        result = response.json()
        print(f"Status: {result['status']}")

        if result["status"] == "success":
            print("✅ Base FLUX generation test passed!")

            # Save test image
            image_data = result["images"][0]
            image_bytes = base64.b64decode(image_data["image"])
            image = Image.open(io.BytesIO(image_bytes))
            image.save("test_base_flux_generation.png")
            print("📸 Base FLUX image saved as 'test_base_flux_generation.png'")

            return True
        else:
            print(f"❌ Base generation failed: {result.get('message', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Base generation error: {str(e)}")
        return False

def main():
    # Replace with your actual RunPod endpoint URL and API key
    api_url = "https://your-runpod-endpoint.runpod.net"
    api_key = "your-runpod-api-key"
    
    # Headers for authentication
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    print("🧪 DreamBooth Flux Dog Example Test Suite")
    print("=" * 50)

    # Test 1: Base FLUX generation
    if not test_base_generation(api_url, headers):
        print("❌ Base FLUX generation failed. Check if the model is properly loaded.")
        return

    # Test 2: Dog training and generation
    if not test_dog_training(api_url, headers):
        print("❌ Dog training workflow failed.")
        return

    print("\n🎉 All dog example tests completed successfully!")
    print("\nGenerated files:")
    print("- test_base_flux_generation.png (Base FLUX model)")
    print("- test_dog_generation.png (Trained dog model)")

if __name__ == "__main__":
    main()
