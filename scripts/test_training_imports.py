#!/usr/bin/env python3
"""
Test training component imports without settings dependency.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def test_training_imports():
    """Test training component imports."""
    print("Testing Training Component Imports...")
    
    try:
        # Test individual imports without triggering settings
        print("Testing LoRA trainer...")
        from src.training.lora_trainer import LoRATrainer
        print("✅ LoRA trainer imported")
        
        print("Testing resource manager...")
        from src.training.resource_manager import LocalResourceManager
        print("✅ Resource manager imported")
        
        print("Testing comparison engine...")
        from src.training.comparison_engine import ModelComparisonEngine
        print("✅ Comparison engine imported")
        
        # Test data models
        print("Testing data models...")
        from src.data.schemas.data_models import TrainingConfig, TrainingJob, ModelComparison
        print("✅ Data models imported")
        
        # Test creating instances
        print("Testing component initialization...")
        
        resource_manager = LocalResourceManager()
        print("✅ Resource manager initialized")
        
        comparison_engine = ModelComparisonEngine()
        print("✅ Comparison engine initialized")
        
        # Test training config
        config = TrainingConfig(
            model_name="test-model",
            dataset_path="test-path",
            output_dir="test-output",
            gpu_type="test-gpu"
        )
        print("✅ Training config created")
        
        print("\n🎉 All training components imported and initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

if __name__ == "__main__":
    success = test_training_imports()
    exit(0 if success else 1)