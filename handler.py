import os
import json
import requests
import subprocess
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gpu_health():
    """Check GPU health and availability"""
    try:
        # Check if nvidia-smi is available
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split('\n')
            logger.info(f"GPU Health Check: Found {len(gpu_info)} GPU(s)")
            for i, gpu in enumerate(gpu_info):
                logger.info(f"GPU {i}: {gpu}")
            return True
        else:
            logger.error(f"GPU Health Check Failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("GPU Health Check: nvidia-smi timed out")
        return False
    except FileNotFoundError:
        logger.error("GPU Health Check: nvidia-smi not found")
        return False
    except Exception as e:
        logger.error(f"GPU Health Check Error: {str(e)}")
        return False

# RunPod API configuration
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")
RUNPOD_API_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"

# Hugging Face token for model access
HF_TOKEN = os.getenv("HF_TOKEN", "")

def start_training(
    instance_prompt: str = "a photo of sks dog",
    max_train_steps: int = 500,
    resolution: int = 1024,
    learning_rate: float = 1.0,
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4
) -> Dict[str, Any]:
    # Check if API keys are provided
    if not RUNPOD_API_KEY:
        return {
            "success": False,
            "error": "RUNPOD_API_KEY environment variable not set"
        }
    """
    Start DreamBooth training on RunPod via API call.
    
    Args:
        instance_prompt: The prompt describing the subject to train
        max_train_steps: Maximum training steps
        resolution: Image resolution for training
        learning_rate: Learning rate for training
        train_batch_size: Training batch size
        gradient_accumulation_steps: Gradient accumulation steps
    
    Returns:
        Dict containing the API response and job status
    """
    
    # Prepare the training command
    training_command = f"""
    cd /workspace/diffusers/dreambooth && \\
    python run_training.py \\
    --instance_prompt "{instance_prompt}" \\
    --max_train_steps {max_train_steps} \\
    --resolution {resolution} \\
    --learning_rate {learning_rate} \\
    --train_batch_size {train_batch_size} \\
    --gradient_accumulation_steps {gradient_accumulation_steps}
    """
    
    # Prepare the API request payload
    payload = {
        "input": {
            "command": training_command.strip(),
            "model_name": "black-forest-labs/FLUX.1-dev",
            "instance_prompt": instance_prompt,
            "max_train_steps": max_train_steps,
            "resolution": resolution,
            "learning_rate": learning_rate,
            "train_batch_size": train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps
        }
    }
    
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        print(f"🚀 Starting training on RunPod...")
        print(f"📝 Instance prompt: {instance_prompt}")
        print(f"⚙️ Training steps: {max_train_steps}")
        print(f"🖼️ Resolution: {resolution}")
        
        # Make the API call to RunPod
        response = requests.post(
            RUNPOD_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Training job started successfully!")
            print(f"🆔 Job ID: {result.get('id', 'Unknown')}")
            return {
                "success": True,
                "job_id": result.get('id'),
                "status": "started",
                "message": "Training job initiated successfully",
                "response": result
            }
        else:
            print(f"❌ Failed to start training: {response.status_code}")
            print(f"Response: {response.text}")
            return {
                "success": False,
                "error": f"API call failed with status {response.status_code}",
                "response": response.text
            }
            
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "success": False,
            "error": error_msg
        }
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "success": False,
            "error": error_msg
        }

def check_training_status(job_id: str) -> Dict[str, Any]:
    """
    Check the status of a training job.
    
    Args:
        job_id: The RunPod job ID to check
    
    Returns:
        Dict containing the job status
    """
    
    status_url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/{job_id}"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    
    try:
        response = requests.get(status_url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "job_id": job_id,
                "status": result.get('status'),
                "response": result
            }
        else:
            return {
                "success": False,
                "error": f"Failed to get status: {response.status_code}",
                "response": response.text
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Error checking status: {str(e)}"
        }

def handler(event, context):
    """
    Main handler function for the API endpoint.
    
    Args:
        event: API Gateway event containing the request
        context: Lambda context
    
    Returns:
        API response with training status
    """
    
    try:
        # First, check GPU health
        print("🔍 Checking GPU health...")
        gpu_healthy = check_gpu_health()
        
        if not gpu_healthy:
            return {
                'statusCode': 500,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'success': False,
                    'error': 'GPU health check failed - container may not have proper GPU access'
                })
            }
        
        # Parse the request body
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', {})
        
        # Extract parameters from the request
        instance_prompt = body.get('instance_prompt', 'a photo of sks dog')
        max_train_steps = int(body.get('max_train_steps', 500))
        resolution = int(body.get('resolution', 1024))
        learning_rate = float(body.get('learning_rate', 1.0))
        train_batch_size = int(body.get('train_batch_size', 1))
        gradient_accumulation_steps = int(body.get('gradient_accumulation_steps', 4))
        
        # Check if this is a status check request
        if body.get('action') == 'check_status' and body.get('job_id'):
            result = check_training_status(body['job_id'])
        else:
            # Start a new training job
            result = start_training(
                instance_prompt=instance_prompt,
                max_train_steps=max_train_steps,
                resolution=resolution,
                learning_rate=learning_rate,
                train_batch_size=train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps
            )
        
        # Return the API response
        return {
            'statusCode': 200 if result.get('success') else 400,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(result)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': False,
                'error': f'Internal server error: {str(e)}'
            })
        }

# For local testing
if __name__ == "__main__":
    # Test the training start
    print("🧪 Testing training API...")
    result = start_training(
        instance_prompt="a photo of sks cat",
        max_train_steps=300,
        resolution=512
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # If we got a job ID, test status check
    if result.get('success') and result.get('job_id'):
        print(f"\n📊 Checking status for job {result['job_id']}...")
        status = check_training_status(result['job_id'])
        print(f"Status: {json.dumps(status, indent=2)}")
