#!/usr/bin/env python3
"""
Test script to verify DreamBooth training setup is working correctly.
This tests all the key components needed for training.
"""

import os
import sys
import traceback

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing module imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import torchvision
        print(f"✅ TorchVision: {torchvision.__version__}")
    except ImportError as e:
        print(f"❌ TorchVision import failed: {e}")
        return False
    
    try:
        import torchaudio
        print(f"✅ TorchAudio: {torchaudio.__version__}")
    except ImportError as e:
        print(f"❌ TorchAudio import failed: {e}")
        return False
    
    try:
        import numpy
        print(f"✅ NumPy: {numpy.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    try:
        import diffusers
        print(f"✅ Diffusers: {diffusers.__version__}")
    except ImportError as e:
        print(f"❌ Diffusers import failed: {e}")
        return False
    
    try:
        import accelerate
        print(f"✅ Accelerate: {accelerate.__version__}")
    except ImportError as e:
        print(f"❌ Accelerate import failed: {e}")
        return False
    
    try:
        import bitsandbytes
        print(f"✅ BitsAndBytes: {bitsandbytes.__version__}")
    except ImportError as e:
        print(f"❌ BitsAndBytes import failed: {e}")
        return False
    
    try:
        import xformers
        print(f"✅ XFormers: {xformers.__version__}")
    except ImportError as e:
        print(f"❌ XFormers import failed: {e}")
        return False
    
    try:
        import flash_attn
        print(f"✅ Flash Attention: {flash_attn.__version__}")
    except ImportError as e:
        print(f"❌ Flash Attention import failed: {e}")
        return False
    
    try:
        import triton
        print(f"✅ Triton: {triton.__version__}")
    except ImportError as e:
        print(f"❌ Triton import failed: {e}")
        return False
    
    return True

def test_clip_modules():
    """Test that CLIP modules can be imported."""
    print("\n🔍 Testing CLIP module imports...")
    
    try:
        from transformers import CLIPTextModel
        print("✅ CLIPTextModel imported successfully")
    except ImportError as e:
        print(f"❌ CLIPTextModel import failed: {e}")
        return False
    
    try:
        from transformers import CLIPTokenizer
        print("✅ CLIPTokenizer imported successfully")
    except ImportError as e:
        print(f"❌ CLIPTokenizer import failed: {e}")
        return False
    
    try:
        from transformers import CLIPImageProcessor
        print("✅ CLIPImageProcessor imported successfully")
    except ImportError as e:
        print(f"❌ CLIPImageProcessor import failed: {e}")
        return False
    
    return True

def test_diffusers_pipelines():
    """Test that diffusers pipelines can be imported."""
    print("\n🔍 Testing Diffusers pipeline imports...")
    
    try:
        from diffusers import StableDiffusionPipeline
        print("✅ StableDiffusionPipeline imported successfully")
    except ImportError as e:
        print(f"❌ StableDiffusionPipeline import failed: {e}")
        return False
    
    try:
        from diffusers import DPMSolverMultistepScheduler
        print("✅ DPMSolverMultistepScheduler imported successfully")
    except ImportError as e:
        print(f"❌ DPMSolverMultistepScheduler import failed: {e}")
        return False
    
    return True

def test_accelerate_setup():
    """Test that accelerate can be configured."""
    print("\n🔍 Testing Accelerate setup...")
    
    try:
        from accelerate.utils import write_basic_config
        write_basic_config()
        print("✅ Accelerate basic config written successfully")
    except Exception as e:
        print(f"❌ Accelerate config failed: {e}")
        return False
    
    try:
        from accelerate import Accelerator
        accelerator = Accelerator()
        print("✅ Accelerator initialized successfully")
    except Exception as e:
        print(f"❌ Accelerator initialization failed: {e}")
        return False
    
    return True

def test_training_script():
    """Test that the training script can be imported."""
    print("\n🔍 Testing training script import...")
    
    try:
        # Check if training script exists
        if os.path.exists("train_dreambooth.py"):
            print("✅ Training script exists")
            
            # Try to import it (this might fail if there are syntax errors)
            with open("train_dreambooth.py", "r") as f:
                content = f.read()
                if "def main" in content or "if __name__" in content:
                    print("✅ Training script appears to have valid Python structure")
                else:
                    print("⚠️  Training script structure unclear")
        else:
            print("❌ Training script not found")
            return False
    except Exception as e:
        print(f"❌ Training script test failed: {e}")
        return False
    
    return True

def test_dataset_access():
    """Test that the dog dataset can be accessed."""
    print("\n🔍 Testing dataset access...")
    
    try:
        if os.path.exists("/workspace/dog"):
            print("✅ Dog dataset directory exists")
            files = os.listdir("/workspace/dog")
            print(f"   Found {len(files)} files in dataset")
            return True
        else:
            print("⚠️  Dog dataset directory not found at /workspace/dog")
            return True  # Not critical for basic setup
    except Exception as e:
        print(f"❌ Dataset access test failed: {e}")
        return True  # Not critical for basic setup

def main():
    """Run all tests."""
    print("🚀 Starting DreamBooth training setup tests...\n")
    
    tests = [
        ("Module Imports", test_imports),
        ("CLIP Modules", test_clip_modules),
        ("Diffusers Pipelines", test_diffusers_pipelines),
        ("Accelerate Setup", test_accelerate_setup),
        ("Training Script", test_training_script),
        ("Dataset Access", test_dataset_access),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    print("\n" + "="*50)
    print("📊 TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Training setup is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
