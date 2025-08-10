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

# Hugging Face token for model access
HF_TOKEN = os.getenv("HF_TOKEN", "")

def start_training_directly(
    instance_prompt: str = "a photo of sks dog",
    max_train_steps: int = 500,
    resolution: int = 1024,
    learning_rate: float = 1.0,
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4
) -> Dict[str, Any]:
    """
    Start DreamBooth training directly on this container.
    This function executes the training command directly instead of calling the API.
    """
    
    try:
        print(f"🚀 Starting DreamBooth training directly on container...")
        print(f"📝 Instance prompt: {instance_prompt}")
        print(f"⚙️ Training steps: {max_train_steps}")
        print(f"🖼️ Resolution: {resolution}")
        print(f"📊 Batch size: {train_batch_size}")
        print(f"🔄 Gradient accumulation: {gradient_accumulation_steps}")
        
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
        
        print(f"🔧 Executing command: {training_command}")
        
        # Execute the training command directly
        result = subprocess.run(
            training_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            return {
                "success": True,
                "status": "completed",
                "message": "Training completed successfully",
                "output": result.stdout,
                "stderr": result.stderr
            }
        else:
            print(f"❌ Training failed with return code: {result.returncode}")
            print(f"Error output: {result.stderr}")
            return {
                "success": False,
                "status": "failed",
                "error": f"Training failed with return code {result.returncode}",
                "stderr": result.stderr,
                "stdout": result.stdout
            }
            
    except subprocess.TimeoutExpired:
        error_msg = "Training timed out after 1 hour"
        print(f"❌ {error_msg}")
        return {
            "success": False,
            "status": "timeout",
            "error": error_msg
        }
    except Exception as e:
        error_msg = f"Unexpected error during training: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "success": False,
            "status": "error",
            "error": error_msg
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
        
        # Start training directly on this container
        result = start_training_directly(
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
    result = start_training_directly(
        instance_prompt="a photo of sks cat",
        max_train_steps=300,
        resolution=512
    )
    print(f"Result: {json.dumps(result, indent=2)}")
