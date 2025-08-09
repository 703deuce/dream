#!/usr/bin/env python3
"""
Setup script for DreamBooth Flux environment
Automatically configures accelerate for optimal performance
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def setup_accelerate():
    """Configure accelerate for DreamBooth Flux"""
    print("🚀 Setting up DreamBooth Flux environment...")
    print("=" * 50)
    
    # Check if accelerate is installed
    if not run_command("python -c 'import accelerate'", "Checking accelerate installation"):
        print("❌ Accelerate not found. Please install it first:")
        print("   pip install accelerate")
        return False
    
    # Create accelerate config directory
    config_dir = Path.home() / ".cache" / "huggingface" / "accelerate"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Create default accelerate config for DreamBooth Flux
    config_content = """compute_environment: LOCAL_MACHINE
distributed_type: NO
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""
    
    config_file = config_dir / "default_config.yaml"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Accelerate config created at: {config_file}")
    
    # Test accelerate configuration
    if run_command("accelerate env", "Testing accelerate configuration"):
        print("✅ Accelerate configuration is working correctly")
        return True
    else:
        print("❌ Accelerate configuration test failed")
        return False

def install_diffusers_from_source():
    """Install diffusers from source as recommended for DreamBooth Flux"""
    print("\n📦 Installing diffusers from source...")
    print("=" * 40)
    
    # Remove existing diffusers installation if it exists
    if run_command("pip uninstall diffusers -y", "Removing existing diffusers installation"):
        print("✅ Removed existing diffusers installation")
    
    # Clone diffusers repository
    if not run_command("git clone https://github.com/huggingface/diffusers", "Cloning diffusers repository"):
        return False
    
    # Change to diffusers directory and install in editable mode
    if not run_command("cd diffusers && pip install -e .", "Installing diffusers in editable mode"):
        return False
    
    # Install DreamBooth Flux requirements
    if not run_command("cd diffusers/examples/dreambooth && pip install -r requirements_flux.txt", "Installing DreamBooth Flux requirements"):
        return False
    
    print("✅ Diffusers installed from source successfully")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    print("=" * 30)
    
    # Install from requirements.txt
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return False
    
    # Install diffusers from source
    if not install_diffusers_from_source():
        return False
    
    return True

def main():
    """Main setup function"""
    print("🎯 DreamBooth Flux Environment Setup")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found. Please run this script from the project root.")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return False
    
    # Setup accelerate
    if not setup_accelerate():
        print("❌ Failed to setup accelerate")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Build Docker image: docker build -t dreambooth-flux .")
    print("2. Deploy to RunPod serverless")
    print("3. Test with: python test_api.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
