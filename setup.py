#!/usr/bin/env python3
"""
Quick setup script for LLM Finance Leaderboard.

This script performs initial setup and demonstrates the system.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"📋 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("🏦 LLM Finance Leaderboard - Quick Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ is required")
        return 1
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create .env file if it doesn't exist
    if not Path(".env").exists():
        print("📝 Creating .env file from template...")
        try:
            with open(".env.example", "r") as src, open(".env", "w") as dst:
                dst.write(src.read())
            print("✅ .env file created")
            print("⚠️  Please edit .env file with your API keys before running the application")
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("⚠️  Some dependencies may have failed to install")
    
    # Run environment setup
    if not run_command("python scripts/setup_environment.py --skip-api-check --skip-pinecone", "Setting up environment"):
        print("⚠️  Environment setup had some issues")
    
    # Create sample data directories
    print("📁 Creating data directories...")
    directories = [
        "data/raw/sec_filings",
        "data/raw/earnings_transcripts", 
        "data/raw/market_data",
        "data/raw/news_data",
        "data/processed/embeddings",
        "data/processed/ground_truth",
        "data/results/benchmark_runs",
        "data/results/model_outputs",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Data directories created")
    
    print("\n🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your API keys:")
    print("   - PINECONE_API_KEY (required)")
    print("   - PINECONE_ENVIRONMENT (required)")
    print("   - OPENAI_API_KEY (optional)")
    print("   - ANTHROPIC_API_KEY (optional)")
    print("   - HUGGINGFACE_API_TOKEN (optional)")
    print()
    print("2. Run the Streamlit dashboard:")
    print("   streamlit run streamlit_app/main.py")
    print()
    print("3. Or run a demo benchmark:")
    print("   python scripts/run_benchmark.py --verbose")
    print()
    print("4. View the architecture documentation:")
    print("   cat LLM_Finance_Leaderboard_Architecture.md")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())