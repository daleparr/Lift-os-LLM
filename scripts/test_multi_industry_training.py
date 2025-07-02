#!/usr/bin/env python3
"""
Test script for multi-industry training system.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_dataset_files():
    """Test that all industry dataset files exist and are valid."""
    
    datasets = {
        "finance": "data/training/synthetic_finance_gsib_v3.jsonl",
        "education": "data/training/synthetic_education_v1.jsonl",
        "retail": "data/training/synthetic_retail_v1.jsonl",
        "healthcare": "data/training/synthetic_healthcare_v1.jsonl",
        "combined": "data/training/combined_multi_industry_corpus.jsonl"
    }
    
    print("Testing dataset files...")
    
    for industry, path in datasets.items():
        if os.path.exists(path):
            print(f"[OK] {industry.title()}: {path}")
            
            # Test file format
            try:
                with open(path, 'r') as f:
                    lines = f.readlines()
                    sample_count = len(lines)
                    
                    # Test first line is valid JSON
                    first_line = json.loads(lines[0])
                    required_fields = ["instruction", "input", "output"]
                    
                    if all(field in first_line for field in required_fields):
                        print(f"   [INFO] {sample_count} samples, valid format")
                    else:
                        print(f"   [ERROR] Invalid format - missing required fields")
                        
            except Exception as e:
                print(f"   [ERROR] Error reading file: {e}")
        else:
            print(f"[ERROR] {industry.title()}: {path} - FILE NOT FOUND")
    
    print()


def test_training_config():
    """Test training configuration."""
    
    config_path = "src/config/training_config.yaml"
    
    print("Testing training configuration...")
    
    if os.path.exists(config_path):
        print(f"[OK] Training config: {config_path}")
        
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check for industry configuration
            if 'industries' in config:
                industries = config['industries']
                print(f"   [INFO] {len(industries)} industries configured:")
                for industry, info in industries.items():
                    print(f"      - {industry}: {info.get('name', 'Unknown')}")
            else:
                print("   [ERROR] No industries section found")
                
            # Check datasets
            if 'datasets' in config:
                datasets = config['datasets']
                print(f"   [INFO] {len(datasets)} datasets configured")
            else:
                print("   [ERROR] No datasets section found")
                
        except Exception as e:
            print(f"   [ERROR] Error reading config: {e}")
    else:
        print(f"[ERROR] Training config not found: {config_path}")
    
    print()


def test_data_models():
    """Test data models import."""
    
    print("Testing data models...")
    
    try:
        from src.data.schemas.data_models import TrainingJob, Industry
        print("[OK] Data models imported successfully")
        
        # Test TrainingJob with industries
        job = TrainingJob(
            job_id="test-123",
            model_name="test-model",
            dataset_name="test-dataset",
            industries=["finance", "education"],
            status="pending",
            priority="Normal"
        )
        print(f"   [OK] TrainingJob with industries: {job.industries}")
        
        # Test Industry enum
        print(f"   [OK] Industry enum values: {[e.value for e in Industry]}")
        
    except Exception as e:
        print(f"[ERROR] Error importing data models: {e}")
    
    print()


def main():
    """Run all tests."""
    
    print("Multi-Industry Training System Test")
    print("=" * 50)
    
    test_dataset_files()
    test_training_config()
    test_data_models()
    
    print("[OK] Multi-industry training system test completed!")
    print("\nNext Steps:")
    print("1. Start the Streamlit app: streamlit run streamlit_app/main.py")
    print("2. Navigate to Training Dashboard")
    print("3. Select industries and submit a fine-tuning job")
    print("4. Monitor progress in the job monitoring section")


if __name__ == "__main__":
    main()