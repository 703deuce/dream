#!/usr/bin/env python3
"""
Test script for RunPod DreamBooth training setup.
This script tests the key components needed for training on RunPod.
"""

import os
import sys
import json
from pathlib import Path

def test_basic_setup():
    """Test basic environment setup."""
    print("🔍 Testing basic environment setup...")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check working directory
    print(f"Working directory: {os.getcwd()}")
    
    # Check if we're in the right place
    if os.path.exists("/workspace"):
        print("✅ Running in /workspace directory")
    else:
        print("⚠️  Not in /workspace directory")
    
    return True

def test_pytorch():
    """Test PyTorch installation."""
    print("\n🔍 Testing PyTorch...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            print(f"   Current device: {torch.cuda.current_device()}")
            print(f"   Device name: {torch.cuda.get_device_name()}")
            
            # Test GPU memory
            if torch.cuda.device_count() > 0:
                device = torch.device("cuda:0")
                print(f"   GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️  CUDA not available")
            
        return True
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False

def test_training_dependencies():
    """Test training dependencies."""
    print("\n🔍 Testing training dependencies...")
    
    dependencies = [
        ("torchvision", "TorchVision"),
        ("torchaudio", "TorchAudio"),
        ("numpy", "NumPy"),
        ("transformers", "Transformers"),
        ("diffusers", "Diffusers"),
        ("accelerate", "Accelerate"),
        ("bitsandbytes", "BitsAndBytes"),
        ("xformers", "XFormers"),
        ("flash_attn", "Flash Attention"),
        ("triton", "Triton"),
    ]
    
    all_good = True
    for module_name, display_name in dependencies:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name}: {version}")
        except ImportError as e:
            print(f"❌ {display_name}: {e}")
            all_good = False
    
    return all_good

def test_clip_modules():
    """Test CLIP modules."""
    print("\n🔍 Testing CLIP modules...")
    
    try:
        from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
        print("✅ All CLIP modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ CLIP modules import failed: {e}")
        return False

def test_diffusers():
    """Test diffusers installation."""
    print("\n🔍 Testing Diffusers...")
    
    try:
        from diffusers import StableDiffusionPipeline
        print("✅ StableDiffusionPipeline imported successfully")
        
        # Check diffusers version
        import diffusers
        print(f"   Diffusers version: {diffusers.__version__}")
        
        return True
    except ImportError as e:
        print(f"❌ Diffusers import failed: {e}")
        return False

def test_accelerate():
    """Test accelerate setup."""
    print("\n🔍 Testing Accelerate...")
    
    try:
        from accelerate.utils import write_basic_config
        write_basic_config()
        print("✅ Accelerate config written")
        
        from accelerate import Accelerator
        accelerator = Accelerator()
        print("✅ Accelerator initialized")
        
        return True
    except Exception as e:
        print(f"❌ Accelerate setup failed: {e}")
        return False

def test_training_files():
    """Test training files exist."""
    print("\n🔍 Testing training files...")
    
    files_to_check = [
        "train_dreambooth.py",
        "handler.py",
        "/workspace/diffusers/examples/dreambooth/requirements_flux.txt"
    ]
    
    all_exist = True
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} not found")
            all_exist = False
    
    return all_exist

def test_dataset():
    """Test dataset access."""
    print("\n🔍 Testing dataset...")
    
    dataset_path = "/workspace/dog"
    if os.path.exists(dataset_path):
        files = os.listdir(dataset_path)
        print(f"✅ Dog dataset found with {len(files)} files")
        
        # Show some file examples
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            print(f"   Image files: {image_files[:3]}{'...' if len(image_files) > 3 else ''}")
        
        return True
    else:
        print("⚠️  Dog dataset not found at /workspace/dog")
        return True  # Not critical

def test_memory():
    """Test available memory."""
    print("\n🔍 Testing memory...")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"✅ System memory: {memory.total / 1024**3:.1f} GB total, {memory.available / 1024**3:.1f} GB available")
        
        if hasattr(psutil, 'gpu_memory'):
            gpu_memory = psutil.gpu_memory()
            print(f"   GPU memory: {gpu_memory.total / 1024**3:.1f} GB total, {gpu_memory.available / 1024**3:.1f} GB available")
        
        return True
    except ImportError:
        print("⚠️  psutil not available for memory info")
        return True

def main():
    """Run all tests."""
    print("🚀 Starting RunPod DreamBooth training tests...\n")
    
    tests = [
        ("Basic Setup", test_basic_setup),
        ("PyTorch", test_pytorch),
        ("Training Dependencies", test_training_dependencies),
        ("CLIP Modules", test_clip_modules),
        ("Diffusers", test_diffusers),
        ("Accelerate", test_accelerate),
        ("Training Files", test_training_files),
        ("Dataset", test_dataset),
        ("Memory", test_memory),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    print("\n" + "="*60)
    print("📊 RUNPOD TRAINING SETUP TEST RESULTS")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Training setup is ready for RunPod.")
        print("\nNext steps:")
        print("1. Upload your training images to /tmp/instance_data/")
        print("2. Set your environment variables (HF_TOKEN, etc.)")
        print("3. Run: python train_dreambooth.py")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
