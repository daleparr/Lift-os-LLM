#!/usr/bin/env python3
"""
Environment setup script for LLM Finance Leaderboard.

This script initializes the environment, creates necessary directories,
and performs initial setup tasks.
"""

import os
import sys
import argparse
from pathlib import Path
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.settings import settings


def create_directories():
    """Create all necessary directories."""
    logger.info("Creating directory structure...")
    
    try:
        settings.create_directories()
        logger.success("Directory structure created successfully")
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        return False
    
    return True


def check_api_keys():
    """Check if required API keys are configured."""
    logger.info("Checking API key configuration...")
    
    required_keys = {
        "Pinecone API Key": settings.pinecone_api_key,
        "Pinecone Environment": settings.pinecone_environment,
    }
    
    optional_keys = {
        "OpenAI API Key": settings.openai_api_key,
        "Anthropic API Key": settings.anthropic_api_key,
        "HuggingFace Token": settings.huggingface_api_token,
        "FRED API Key": settings.fred_api_key,
    }
    
    missing_required = []
    for name, value in required_keys.items():
        if not value:
            missing_required.append(name)
    
    if missing_required:
        logger.error(f"Missing required API keys: {', '.join(missing_required)}")
        logger.info("Please update your .env file with the required API keys")
        return False
    
    missing_optional = []
    for name, value in optional_keys.items():
        if not value:
            missing_optional.append(name)
    
    if missing_optional:
        logger.warning(f"Missing optional API keys: {', '.join(missing_optional)}")
        logger.info("Some features may be limited without these keys")
    
    logger.success("API key configuration check completed")
    return True


def test_imports():
    """Test that all required packages can be imported."""
    logger.info("Testing package imports...")
    
    required_packages = [
        "streamlit",
        "pandas",
        "numpy",
        "torch",
        "transformers",
        "sentence_transformers",
        "pinecone",
        "sqlalchemy",
        "plotly",
        "pydantic",
        "requests",
        "loguru",
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.debug(f"✓ {package}")
        except ImportError as e:
            failed_imports.append(package)
            logger.error(f"✗ {package}: {e}")
    
    if failed_imports:
        logger.error(f"Failed to import: {', '.join(failed_imports)}")
        logger.info("Please install missing packages: pip install -r requirements.txt")
        return False
    
    logger.success("All required packages imported successfully")
    return True


def initialize_database():
    """Initialize the SQLite database."""
    logger.info("Initializing database...")
    
    try:
        from src.utils.database import init_database
        init_database()
        logger.success("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False


def test_pinecone_connection():
    """Test connection to Pinecone."""
    if not settings.pinecone_api_key:
        logger.warning("Pinecone API key not configured, skipping connection test")
        return True
    
    logger.info("Testing Pinecone connection...")
    
    try:
        import pinecone
        
        pinecone.init(
            api_key=settings.pinecone_api_key,
            environment=settings.pinecone_environment
        )
        
        # List indexes to test connection
        indexes = pinecone.list_indexes()
        logger.success(f"Pinecone connection successful. Available indexes: {indexes}")
        return True
        
    except Exception as e:
        logger.error(f"Pinecone connection failed: {e}")
        return False


def create_sample_env():
    """Create a sample .env file if it doesn't exist."""
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    
    if env_path.exists():
        logger.info(".env file already exists")
        return True
    
    if not env_example_path.exists():
        logger.error(".env.example file not found")
        return False
    
    try:
        # Copy .env.example to .env
        with open(env_example_path, 'r') as src, open(env_path, 'w') as dst:
            dst.write(src.read())
        
        logger.success("Created .env file from .env.example")
        logger.info("Please edit .env file with your actual API keys")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create .env file: {e}")
        return False


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup LLM Finance Leaderboard environment")
    parser.add_argument("--skip-api-check", action="store_true", help="Skip API key validation")
    parser.add_argument("--skip-pinecone", action="store_true", help="Skip Pinecone connection test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    logger.info("🚀 Starting LLM Finance Leaderboard setup...")
    
    setup_steps = [
        ("Creating sample .env file", create_sample_env),
        ("Creating directories", create_directories),
        ("Testing package imports", test_imports),
        ("Initializing database", initialize_database),
    ]
    
    if not args.skip_api_check:
        setup_steps.append(("Checking API keys", check_api_keys))
    
    if not args.skip_pinecone:
        setup_steps.append(("Testing Pinecone connection", test_pinecone_connection))
    
    # Execute setup steps
    failed_steps = []
    
    for step_name, step_func in setup_steps:
        logger.info(f"📋 {step_name}...")
        try:
            if not step_func():
                failed_steps.append(step_name)
        except Exception as e:
            logger.error(f"Error in {step_name}: {e}")
            failed_steps.append(step_name)
    
    # Summary
    if failed_steps:
        logger.error(f"❌ Setup completed with errors. Failed steps: {', '.join(failed_steps)}")
        logger.info("Please resolve the issues above and run setup again")
        return 1
    else:
        logger.success("✅ Setup completed successfully!")
        logger.info("You can now run the application with: streamlit run streamlit_app/main.py")
        return 0


if __name__ == "__main__":
    sys.exit(main())