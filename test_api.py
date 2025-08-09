#!/usr/bin/env python3
"""
Test script for DreamBooth Flux API
"""

import requests
import json
import base64
import io
from PIL import Image
import time

def test_generation_only(api_url, headers):
    """Test image generation with base model (no training)"""
    print("Testing image generation with base model...")

    generation_data = {
        "type": "generate",
        "prompt": "a photo of a person in a business suit",
        "negative_prompt": "blurry, bad quality, distorted, low resolution",
        "num_inference_steps": 20,  # Reduced for testing
        "guidance_scale": 7.5,
        "width": 512,
        "height": 512,
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
            print("✅ Generation test passed!")
            
            # Save test image
            image_data = result["images"][0]
            image_bytes = base64.b64decode(image_data["image"])
            image = Image.open(io.BytesIO(image_bytes))
            image.save("test_generation.png")
            print("📸 Test image saved as 'test_generation.png'")
            
            return True
        else:
            print(f"❌ Generation test failed: {result.get('message', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Generation test error: {str(e)}")
        return False

def test_training_workflow(api_url, headers):
    """Test complete training workflow with sample images"""
    print("\nTesting training workflow...")

    # Sample training data (you would replace with real image URLs)
    training_data = {
        "type": "train",
        "instance_prompt": "a photo of sks person",
        "class_prompt": "a photo of a person",
        "image_urls": [
            # Replace with actual image URLs for testing
            "https://example.com/sample1.jpg",
            "https://example.com/sample2.jpg",
            "https://example.com/sample3.jpg"
        ],
        "max_train_steps": 100,  # Reduced for testing
        "learning_rate": 5e-6,
        "train_text_encoder": True,
        "resolution": 512,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1
    }

    try:
        print("Starting training...")
        response = requests.post(
            api_url,
            json={"input": training_data},
            headers=headers,
            timeout=1800  # 30 minutes timeout for training
        )
        
        result = response.json()
        print(f"Training status: {result['status']}")
        
        if result["status"] == "success":
            print("✅ Training test passed!")
            model_path = result.get("model_path")
            print(f"Model saved at: {model_path}")
            
            # Test generation with trained model
            print("Testing generation with trained model...")
            generation_data = {
                "type": "generate",
                "prompt": "a photo of sks person in a business suit",
                "negative_prompt": "blurry, bad quality, distorted",
                "num_inference_steps": 20,
                "guidance_scale": 7.5,
                "width": 512,
                "height": 512,
                "num_images": 1,
                "model_path": model_path
            }
            
            gen_response = requests.post(
                api_url,
                json={"input": generation_data},
                timeout=300
            )
            
            gen_result = gen_response.json()
            if gen_result["status"] == "success":
                print("✅ Trained model generation test passed!")
                
                # Save trained model image
                image_data = gen_result["images"][0]
                image_bytes = base64.b64decode(image_data["image"])
                image = Image.open(io.BytesIO(image_bytes))
                image.save("test_trained_generation.png")
                print("📸 Trained model image saved as 'test_trained_generation.png'")
                
                return True
            else:
                print(f"❌ Trained model generation failed: {gen_result.get('message', 'Unknown error')}")
                return False
        else:
            print(f"❌ Training test failed: {result.get('message', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Training test error: {str(e)}")
        return False

def test_api_health(api_url):
    """Test basic API connectivity"""
    print("Testing API connectivity...")
    
    try:
        response = requests.get(api_url, timeout=10)
        print(f"✅ API is reachable (Status: {response.status_code})")
        return True
    except Exception as e:
        print(f"❌ API connectivity error: {str(e)}")
        return False

def main():
    # Replace with your actual RunPod endpoint URL and API key
    api_url = "https://your-runpod-endpoint.runpod.net"
    api_key = "your-runpod-api-key"  # Add your RunPod API key here
    
    # Headers for authentication
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    print("🧪 DreamBooth Flux API Test Suite")
    print("=" * 50)
    
    # Test 1: API connectivity
    if not test_api_health(api_url):
        print("❌ Cannot connect to API. Please check the URL and ensure the endpoint is running.")
        return
    
    # Test 2: Generation with base model
    if not test_generation_only(api_url, headers):
        print("❌ Base model generation failed. Check if the model is properly loaded.")
        return
    
    # Test 3: Training workflow (commented out as it requires real image URLs)
    print("\n⚠️  Training workflow test skipped (requires real image URLs)")
    print("To test training, update the image_urls in test_training_workflow() with real URLs")
    
    # Uncomment the following line when you have real image URLs:
    # test_training_workflow(api_url, headers)
    
    print("\n✅ All tests completed!")
    print("\nNext steps:")
    print("1. Update api_url with your actual RunPod endpoint")
    print("2. Add real image URLs to test training functionality")
    print("3. Deploy to RunPod serverless for production use")

if __name__ == "__main__":
    main()
