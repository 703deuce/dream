#!/usr/bin/env python3
"""
Simple script to run DreamBooth training on RunPod.
This will download the dog dataset and start training.
"""

import os
import subprocess
import sys
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run DreamBooth training on RunPod')
    parser.add_argument('--instance_prompt', type=str, default='a photo of sks dog',
                       help='The prompt describing the subject to train')
    parser.add_argument('--max_train_steps', type=int, default=500,
                       help='Maximum training steps')
    parser.add_argument('--resolution', type=int, default=1024,
                       help='Image resolution for training')
    parser.add_argument('--learning_rate', type=float, default=1.0,
                       help='Learning rate for training')
    parser.add_argument('--train_batch_size', type=int, default=1,
                       help='Training batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    
    args = parser.parse_args()
    
    print("🚀 Starting DreamBooth training on RunPod...")
    print(f"📝 Instance prompt: {args.instance_prompt}")
    print(f"⚙️ Training steps: {args.max_train_steps}")
    print(f"🖼️ Resolution: {args.resolution}")
    print(f"📚 Learning rate: {args.learning_rate}")
    print(f"📦 Batch size: {args.train_batch_size}")
    print(f"🔄 Gradient accumulation: {args.gradient_accumulation_steps}")
    
    # Set environment variables for training
    os.environ["MODEL_NAME"] = "black-forest-labs/FLUX.1-dev"
    os.environ["INSTANCE_DIR"] = "dog"
    os.environ["OUTPUT_DIR"] = "trained-flux"
    
    # Download the dog dataset if it doesn't exist
    if not os.path.exists("dog"):
        print("📥 Downloading dog dataset...")
        try:
            from huggingface_hub import snapshot_download
            local_dir = "./dog"
            snapshot_download(
                "diffusers/dog-example",
                local_dir=local_dir, 
                repo_type="dataset",
                ignore_patterns=".gitattributes",
            )
            print("✅ Dataset downloaded successfully")
        except Exception as e:
            print(f"❌ Failed to download dataset: {e}")
            return
    
    # Check if training script exists
    if not os.path.exists("train_dreambooth.py"):
        print("❌ Training script not found. Make sure you're in the right directory.")
        return
    
    # Run the training with the provided arguments
    print("🔥 Starting training...")
    cmd = [
        "accelerate", "launch", "train_dreambooth.py",
        "--pretrained_model_name_or_path", os.environ["MODEL_NAME"],
        "--instance_data_dir", os.environ["INSTANCE_DIR"],
        "--output_dir", os.environ["OUTPUT_DIR"],
        "--mixed_precision", "bf16",
        "--instance_prompt", args.instance_prompt,
        "--resolution", str(args.resolution),
        "--train_batch_size", str(args.train_batch_size),
        "--guidance_scale", "1",
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--optimizer", "prodigy",
        "--learning_rate", str(args.learning_rate),
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--max_train_steps", str(args.max_train_steps),
        "--validation_prompt", f"A photo of {args.instance_prompt} in a bucket",
        "--validation_epochs", "25",
        "--seed", "0"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("🎉 Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code: {e.returncode}")
        return
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return

if __name__ == "__main__":
    main()
