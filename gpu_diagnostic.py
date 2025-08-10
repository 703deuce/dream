#!/usr/bin/env python3
"""
Simple GPU test script to diagnose GPU issues in RunPod.
This will help identify what's wrong with the nvidia-smi command.
"""

import subprocess
import os
import sys

def test_nvidia_smi():
    """Test basic nvidia-smi functionality"""
    print("🔍 Testing nvidia-smi...")
    
    try:
        # Test 1: Basic nvidia-smi
        print("📝 Test 1: Basic nvidia-smi")
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        print(f"Return code: {result.returncode}")
        print(f"STDOUT: {result.stdout[:500]}...")
        print(f"STDERR: {result.stderr}")
        
        # Test 2: nvidia-smi with specific query
        print("\n📝 Test 2: nvidia-smi --query-gpu=name")
        result2 = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                               capture_output=True, text=True, timeout=10)
        print(f"Return code: {result2.returncode}")
        print(f"STDOUT: {result2.stdout}")
        print(f"STDERR: {result2.stderr}")
        
        # Test 3: nvidia-smi with memory query
        print("\n📝 Test 3: nvidia-smi --query-gpu=memory.total")
        result3 = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True, timeout=10)
        print(f"Return code: {result3.returncode}")
        print(f"STDOUT: {result3.stdout}")
        print(f"STDERR: {result3.stderr}")
        
        # Test 4: Check if CUDA is available
        print("\n📝 Test 4: Check CUDA availability")
        try:
            import torch
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA version: {torch.version.cuda}")
                print(f"GPU count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        except ImportError:
            print("PyTorch not available")
        except Exception as e:
            print(f"Error checking PyTorch CUDA: {e}")
            
    except Exception as e:
        print(f"❌ Error running nvidia-smi tests: {e}")

def test_environment():
    """Test environment variables and paths"""
    print("\n🔍 Testing environment...")
    
    # Check PATH
    print(f"PATH: {os.environ.get('PATH', 'Not set')}")
    
    # Check CUDA environment
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
    print(f"CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
    
    # Check if nvidia-smi exists
    nvidia_smi_path = None
    for path in os.environ.get('PATH', '').split(':'):
        if os.path.exists(os.path.join(path, 'nvidia-smi')):
            nvidia_smi_path = os.path.join(path, 'nvidia-smi')
            break
    
    print(f"nvidia-smi found at: {nvidia_smi_path or 'Not found in PATH'}")
    
    # Check /usr/bin
    usr_bin_nvidia = "/usr/bin/nvidia-smi"
    if os.path.exists(usr_bin_nvidia):
        print(f"nvidia-smi exists at {usr_bin_nvidia}")
        print(f"Permissions: {oct(os.stat(usr_bin_nvidia).st_mode)[-3:]}")
    else:
        print(f"nvidia-smi not found at {usr_bin_nvidia}")

def test_gpu_devices():
    """Test GPU device files"""
    print("\n🔍 Testing GPU device files...")
    
    gpu_devices = [
        "/dev/nvidia0",
        "/dev/nvidia1", 
        "/dev/nvidia2",
        "/dev/nvidia3"
    ]
    
    for device in gpu_devices:
        if os.path.exists(device):
            print(f"✅ {device} exists")
            try:
                stat = os.stat(device)
                print(f"   Permissions: {oct(stat.st_mode)[-3:]}")
                print(f"   Owner: {stat.st_uid}")
                print(f"   Group: {stat.st_gid}")
            except Exception as e:
                print(f"   Error getting stats: {e}")
        else:
            print(f"❌ {device} not found")

if __name__ == "__main__":
    print("🚀 GPU Diagnostic Test Suite")
    print("=" * 50)
    
    test_nvidia_smi()
    test_environment()
    test_gpu_devices()
    
    print("\n🎉 GPU diagnostic tests completed!")
    print("Check the output above for any errors or issues.")
