#!/usr/bin/env python3
"""
Environment setup script for Desi Debate
"""

import os
import sys
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/models",
        "data/processed", 
        "data/raw",
        "data/chroma",
        "logs",
        "cache",
        "src/rag/data/rag"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def check_environment():
    """Check if environment is properly set up"""
    print("Checking environment setup...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check if .env exists
    if not Path(".env").exists():
        print("❌ .env file not found")
        print("   Please copy .env.example to .env and add your OpenAI API key")
        return False
    else:
        print("✓ .env file exists")
    
    # Check API key
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-openai-api-key-here":
        print("❌ OpenAI API key not configured")
        print("   Please set OPENAI_API_KEY in your .env file")
        return False
    else:
        print("✓ OpenAI API key configured")
    
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("Installing dependencies...")
    os.system(f"{sys.executable} -m pip install -r requirements.txt")
    print("✓ Dependencies installed")

def main():
    print("Desi Debate - Environment Setup")
    print("=" * 40)
    
    # Create directories
    create_directories()
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment setup incomplete")
        print("\nNext steps:")
        print("1. Copy .env.example to .env")
        print("2. Add your OpenAI API key to .env")
        print("3. Run this script again")
        return
    
    print("\n✓ Environment setup complete!")
    print("\nYou can now run:")
    print("  python run_flask.py")
    print("\nOr use the startup scripts:")
    print("  scripts/start_flask.bat (Windows)")
    print("  scripts/start_flask.sh (Linux/Mac)")

if __name__ == "__main__":
    main()