#!/usr/bin/env python3
"""
Startup script for Lift-os-LLM microservice.

Provides easy commands to run, test, and manage the microservice.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

def run_development():
    """Run the microservice in development mode."""
    print("🚀 Starting Lift-os-LLM in development mode...")
    
    # Set environment variables for development
    env = os.environ.copy()
    env.update({
        "ENVIRONMENT": "development",
        "DEBUG": "true",
        "LOG_LEVEL": "INFO",
        "API_HOST": "0.0.0.0",
        "API_PORT": "8000"
    })
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "src.main:app",
            "--reload",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--log-level", "info"
        ], env=env, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down microservice...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start microservice: {e}")
        return False
    
    return True

def run_production():
    """Run the microservice in production mode."""
    print("🚀 Starting Lift-os-LLM in production mode...")
    
    # Set environment variables for production
    env = os.environ.copy()
    env.update({
        "ENVIRONMENT": "production",
        "DEBUG": "false",
        "LOG_LEVEL": "WARNING",
        "API_HOST": "0.0.0.0",
        "API_PORT": "8000"
    })
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "src.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--workers", "4",
            "--log-level", "warning"
        ], env=env, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down microservice...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start microservice: {e}")
        return False
    
    return True

def run_docker():
    """Run the microservice using Docker."""
    print("🐳 Starting Lift-os-LLM with Docker...")
    
    try:
        # Build the Docker image
        print("📦 Building Docker image...")
        subprocess.run(["docker", "build", "-t", "lift-os-llm", "."], check=True)
        
        # Run the container
        print("🚀 Starting Docker container...")
        subprocess.run([
            "docker", "run",
            "--rm",
            "-p", "8000:8000",
            "--env-file", ".env",
            "lift-os-llm"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Docker container...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run Docker container: {e}")
        return False
    
    return True

def run_docker_compose():
    """Run the microservice using Docker Compose."""
    print("🐳 Starting Lift-os-LLM with Docker Compose...")
    
    try:
        subprocess.run(["docker-compose", "up", "--build"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Docker Compose...")
        subprocess.run(["docker-compose", "down"])
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run Docker Compose: {e}")
        return False
    
    return True

def run_tests():
    """Run the microservice tests."""
    print("🧪 Running Lift-os-LLM tests...")
    
    try:
        result = subprocess.run([sys.executable, "test_microservice.py"], check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def install_dependencies():
    """Install Python dependencies."""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_environment():
    """Setup development environment."""
    print("🔧 Setting up development environment...")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        env_example = Path(".env.example")
        if env_example.exists():
            env_file.write_text(env_example.read_text())
            print("📝 Created .env file from .env.example")
        else:
            # Create basic .env file
            env_content = """# Lift-os-LLM Environment Configuration
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Database Configuration
DATABASE_URL=sqlite:///./lift_os_llm.db
REDIS_URL=redis://localhost:6379/0

# LLM Provider API Keys (add your keys here)
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
HUGGINGFACE_API_KEY=your-huggingface-api-key-here

# Security
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# CORS
CORS_ORIGINS=["http://localhost:3000","http://localhost:8080"]
ALLOWED_HOSTS=["localhost","127.0.0.1"]

# Features
ENABLE_METRICS=true
ENABLE_RATE_LIMITING=true
"""
            env_file.write_text(env_content)
            print("📝 Created basic .env file")
    
    # Create necessary directories
    directories = ["logs", "data", "uploads"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    print("✅ Environment setup complete")
    return True

def check_health():
    """Check if the microservice is running and healthy."""
    print("🏥 Checking microservice health...")
    
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Service is healthy: {data}")
            return True
        else:
            print(f"❌ Service returned status {response.status_code}")
            return False
    except ImportError:
        print("❌ requests library not installed. Run: pip install requests")
        return False
    except Exception as e:
        print(f"❌ Service is not responding: {e}")
        return False

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Lift-os-LLM Microservice Manager")
    parser.add_argument("command", choices=[
        "dev", "prod", "docker", "compose", "test", "install", "setup", "health"
    ], help="Command to execute")
    
    args = parser.parse_args()
    
    print("🎯 Lift-os-LLM Microservice Manager")
    print("=" * 50)
    
    success = False
    
    if args.command == "dev":
        success = run_development()
    elif args.command == "prod":
        success = run_production()
    elif args.command == "docker":
        success = run_docker()
    elif args.command == "compose":
        success = run_docker_compose()
    elif args.command == "test":
        success = run_tests()
    elif args.command == "install":
        success = install_dependencies()
    elif args.command == "setup":
        success = setup_environment()
    elif args.command == "health":
        success = check_health()
    
    if success:
        print("\n✅ Command completed successfully")
    else:
        print("\n❌ Command failed")
        sys.exit(1)

if __name__ == "__main__":
    main()