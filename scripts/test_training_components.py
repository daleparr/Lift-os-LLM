#!/usr/bin/env python3
"""
Simple test script for training components without full settings dependency.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def test_dataset_loading():
    """Test dataset loading."""
    print("Testing Dataset Loading...")
    
    dataset_path = "data/training/synthetic_finance_v2.jsonl"
    
    if not os.path.exists(dataset_path):
        print(f"X Dataset not found: {dataset_path}")
        return False
    
    # Count lines in dataset
    with open(dataset_path, 'r') as f:
        lines = f.readlines()
    
    print(f"OK Dataset has {len(lines)} samples")
    
    # Test loading first sample
    if lines:
        try:
            sample = json.loads(lines[0])
            print(f"OK Sample format: {list(sample.keys())}")
            print(f"OK Sample instruction: {sample.get('instruction', '')[:100]}...")
            return True
        except json.JSONDecodeError as e:
            print(f"X JSON parsing error: {e}")
            return False
    
    return False


def test_training_config():
    """Test training configuration loading."""
    print("\nTesting Training Configuration...")
    
    config_path = "src/config/training_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"X Training config not found: {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"OK Training config loaded")
        print(f"OK LoRA config: {config.get('lora_config', {})}")
        print(f"OK Supported models: {len(config.get('supported_models', []))}")
        return True
        
    except Exception as e:
        print(f"X Config loading error: {e}")
        return False


def test_gpu_detection():
    """Test GPU detection."""
    print("\nTesting GPU Detection...")
    
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        
        if gpu_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"OK GPU available: {gpu_name}")
            print(f"OK GPU count: {gpu_count}")
            
            # Test memory
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"OK GPU memory: {memory_total:.1f} GB")
            return True
        else:
            print(" No GPU detected (CPU mode)")
            return True
            
    except ImportError:
        print(" PyTorch not available")
        return True
    except Exception as e:
        print(f"X GPU detection error: {e}")
        return False


def test_resource_monitoring():
    """Test resource monitoring libraries."""
    print("\nTesting Resource Monitoring...")
    
    try:
        import psutil
        
        # Test CPU info
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / 1024**3
        print(f"OK CPU cores: {cpu_count}")
        print(f"OK System memory: {memory_gb:.1f} GB")
        
        # Test GPU monitoring
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            print(f"OK GPUtil detected {len(gpus)} GPUs")
            
            if gpus:
                gpu = gpus[0]
                print(f"OK GPU 0: {gpu.name}, {gpu.memoryTotal}MB")
        except ImportError:
            print(" GPUtil not available")
        
        return True
        
    except ImportError as e:
        print(f" Resource monitoring libraries not available: {e}")
        return True
    except Exception as e:
        print(f"X Resource monitoring error: {e}")
        return False


def test_training_dependencies():
    """Test training dependencies."""
    print("\nTesting Training Dependencies...")
    
    dependencies = [
        ("transformers", "Transformers library"),
        ("peft", "PEFT library for LoRA"),
        ("datasets", "HuggingFace datasets"),
        ("torch", "PyTorch"),
        ("bitsandbytes", "Quantization library")
    ]
    
    available = 0
    for module, description in dependencies:
        try:
            __import__(module)
            print(f"OK {description}")
            available += 1
        except ImportError:
            print(f" {description} not available")
    
    print(f"OK {available}/{len(dependencies)} training dependencies available")
    return available > 0


def test_streamlit_components():
    """Test Streamlit components."""
    print("\nTesting Streamlit Components...")
    
    components = [
        "streamlit_app/components/model_selector.py",
        "streamlit_app/components/comparison_results.py",
        "streamlit_app/pages/training_dashboard.py"
    ]
    
    available = 0
    for component in components:
        if os.path.exists(component):
            print(f"OK {component}")
            available += 1
        else:
            print(f"X {component} not found")
    
    print(f"OK {available}/{len(components)} Streamlit components available")
    return available == len(components)


def test_directory_structure():
    """Test directory structure."""
    print("\nTesting Directory Structure...")
    
    directories = [
        "src/training",
        "data/training", 
        "streamlit_app/components",
        "streamlit_app/pages"
    ]
    
    available = 0
    for directory in directories:
        if os.path.exists(directory):
            print(f"OK {directory}")
            available += 1
        else:
            print(f"X {directory} not found")
    
    print(f"OK {available}/{len(directories)} directories available")
    return available == len(directories)


def main():
    """Run all tests."""
    print("Auto Fine-tuning Component Tests")
    print("=" * 50)
    
    tests = [
        test_directory_structure,
        test_dataset_loading,
        test_training_config,
        test_gpu_detection,
        test_resource_monitoring,
        test_training_dependencies,
        test_streamlit_components
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"X Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f" Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print(" All tests passed! Auto fine-tuning system is ready.")
        return 0
    elif passed >= total * 0.7:
        print(" Most tests passed. System should work with some limitations.")
        return 0
    else:
        print("X Many tests failed. Please check dependencies and setup.")
        return 1


if __name__ == "__main__":
    exit(main())