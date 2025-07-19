#!/usr/bin/env python3
"""
Dependency installation script for Desi Debate
"""

import subprocess
import sys
from pathlib import Path

def install_package(package):
    """Install a Python package"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("Desi Debate - Dependency Installation")
    print("=" * 40)
    
    # Core dependencies
    core_packages = [
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "PyYAML>=6.0",
        "python-dotenv>=1.0.0",
        "flask>=3.0.0",
        "flask-cors>=4.0.0",
        "openai>=1.0.0",
        "tqdm>=4.65.0"
    ]
    
    # Optional packages for full functionality
    optional_packages = [
        "torch-geometric>=2.3.0",
        "scikit-learn>=1.3.0", 
        "matplotlib>=3.7.0",
        "transformers>=4.30.0",
        "langchain>=0.1.0",
        "langchain-community>=0.0.20",
        "langchain-openai>=0.0.5",
        "chromadb>=0.4.0",
        "faiss-cpu>=1.7.0"
    ]
    
    print("Installing core dependencies...")
    core_success = 0
    for package in core_packages:
        print(f"Installing {package}...")
        if install_package(package):
            print(f"✓ {package}")
            core_success += 1
        else:
            print(f"❌ {package}")
    
    print(f"\nCore dependencies: {core_success}/{len(core_packages)} installed")
    
    print("\nInstalling optional dependencies...")
    optional_success = 0
    for package in optional_packages:
        print(f"Installing {package}...")
        if install_package(package):
            print(f"✓ {package}")
            optional_success += 1
        else:
            print(f"❌ {package} (optional)")
    
    print(f"\nOptional dependencies: {optional_success}/{len(optional_packages)} installed")
    
    total_success = core_success + optional_success
    total_packages = len(core_packages) + len(optional_packages)
    
    print(f"\nTotal: {total_success}/{total_packages} packages installed")
    
    if core_success == len(core_packages):
        print("✓ Core functionality should work!")
    else:
        print("❌ Some core dependencies failed to install")
    
    if optional_success < len(optional_packages):
        print("\nNote: Some optional features may not work without all dependencies")
        print("You can install them manually with:")
        print("  pip install -r requirements.txt")
    
    print("\nNext steps:")
    print("1. Copy .env.example to .env")
    print("2. Add your OpenAI API key to .env")
    print("3. Run: python test_basic_functionality.py")
    print("4. Run: python run_flask.py")

if __name__ == "__main__":
    main()