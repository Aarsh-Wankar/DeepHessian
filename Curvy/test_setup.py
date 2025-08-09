#!/usr/bin/env python3
"""
Test script to verify W&B sweeps setup
"""

import sys
import os
import subprocess

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not found")
        return False
    
    try:
        import torchvision
        print(f"✅ Torchvision: {torchvision.__version__}")
    except ImportError:
        print("❌ Torchvision not found")
        return False
    
    try:
        import wandb
        print(f"✅ W&B: {wandb.__version__}")
    except ImportError:
        print("❌ W&B not found")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")
    except ImportError:
        print("❌ Pandas not found")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("❌ Matplotlib not found")
        return False
    
    return True

def test_script_syntax():
    """Test if the main script has valid syntax"""
    print("\n🔍 Testing script syntax...")
    
    script_path = "resnet-18-hyperparam-tune.py"
    if not os.path.exists(script_path):
        print(f"❌ {script_path} not found")
        return False
    
    try:
        result = subprocess.run([sys.executable, "-m", "py_compile", script_path], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Script syntax is valid")
            return True
        else:
            print(f"❌ Syntax errors: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error checking syntax: {e}")
        return False

def test_wandb_status():
    """Test W&B authentication status"""
    print("\n🔑 Testing W&B authentication...")
    
    try:
        import wandb
        # Try to get current user (this will fail if not logged in)
        api = wandb.Api()
        user = api.viewer
        print(f"✅ W&B authenticated as: {user.get('username', 'unknown')}")
        return True
    except Exception as e:
        print(f"❌ W&B not authenticated: {e}")
        print("💡 Run 'wandb login' to authenticate")
        return False

def test_cuda_availability():
    """Test CUDA availability"""
    print("\n🖥️  Testing CUDA availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            print(f"✅ CUDA available: {device_count} device(s)")
            print(f"   Primary device: {device_name}")
            return True
        else:
            print("⚠️  CUDA not available - will use CPU")
            return True
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False

def test_file_structure():
    """Test if all required files exist"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        "resnet-18-hyperparam-tune.py",
        "requirements.txt", 
        "run_sweeps.sh",
        "README.md",
        "sweep_config.yaml"
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} not found")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("🚀 Curvy Optimizer W&B Sweeps - Setup Test")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_imports,
        test_script_syntax,
        test_cuda_availability,
        test_wandb_status
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! You're ready to run sweeps!")
        print("\nNext steps:")
        print("1. ./run_sweeps.sh create    # Create a sweep")
        print("2. ./run_sweeps.sh agent <ID> # Run sweep agent")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above before running sweeps.")
        
        if passed < 3:  # If major issues
            print("\n🔧 Setup recommendations:")
            print("1. pip install -r requirements.txt")
            print("2. wandb login")
            print("3. Check that all files are present")

if __name__ == "__main__":
    main()
