"""
System Setup Checker
Verifies that all requirements are met before running the project
"""

import sys
import importlib
import torch
import os
from pathlib import Path


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}", end="")
    if version.major >= 3 and version.minor >= 7:
        print(" ✓")
        return True
    else:
        print(" ✗ (require 3.7+)")
        return False


def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"{package_name:25} ✓ ({version})")
        return True
    except ImportError:
        print(f"{package_name:25} ✗ (not installed)")
        return False


def check_torch():
    """Check PyTorch and CUDA"""
    print(f"PyTorch:                  ✓ ({torch.__version__})")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"  GPU:                      ✓ {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version:             ✓ {torch.version.cuda}")
    else:
        print(f"  GPU:                      ✗ (CPU mode)")
    return True


def check_directories():
    """Check required directories"""
    dirs_to_check = [
        './data/raw',
        './data/processed',
        './data/augmented',
        './models',
        './results/plots',
        './results/metrics',
        './documentation'
    ]
    
    print("\nDirectories:")
    all_ok = True
    for directory in dirs_to_check:
        path = Path(directory)
        if path.exists():
            print(f"  {directory:30} ✓")
        else:
            print(f"  {directory:30} ✗ (missing)")
            all_ok = False
    
    return all_ok


def main():
    """Main execution"""
    
    print("\n" + "="*70)
    print("DEEP LEARNING PROJECT - SETUP CHECK")
    print("="*70 + "\n")
    
    # Check Python
    print("Python:")
    py_ok = check_python_version()
    
    # Check packages
    print("\nRequired Packages:")
    packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('scikit-learn', 'sklearn'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('opencv-python', 'cv2'),
        ('Pillow', 'PIL'),
        ('tqdm', 'tqdm'),
    ]
    
    packages_ok = all(check_package(pkg, imp) for pkg, imp in packages)
    
    # Check PyTorch
    print("\nPyTorch Setup:")
    torch_ok = check_torch()
    
    # Check directories
    print("\nRequired Directories:")
    dirs_ok = check_directories()
    
    # Summary
    print("\n" + "="*70)
    if py_ok and packages_ok and torch_ok:
        print("✓ SETUP CHECK PASSED - Ready to run!")
        print("\nNext steps:")
        print("  1. Download datasets: python download_data.py")
        print("  2. Run project:       python main.py")
    else:
        print("✗ SETUP CHECK FAILED")
        print("\nPlease install missing packages:")
        print("  pip install -r requirements.txt")
    
    if not dirs_ok:
        print("\nCreating missing directories...")
        for directory in ['./data/raw', './data/processed', './data/augmented', 
                         './models', './results/plots', './results/metrics', './documentation']:
            Path(directory).mkdir(parents=True, exist_ok=True)
        print("✓ Directories created")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
