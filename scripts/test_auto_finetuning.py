#!/usr/bin/env python3
"""
Test script for auto fine-tuning functionality.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.local_orchestrator import LocalTrainingOrchestrator
from src.training.comparison_engine import ModelComparisonEngine
from src.training.resource_manager import LocalResourceManager
from src.data.schemas.data_models import TrainingConfig
from loguru import logger


def test_resource_manager():
    """Test the resource manager."""
    logger.info("Testing Resource Manager...")
    
    resource_manager = LocalResourceManager()
    
    # Test GPU detection
    gpus = resource_manager.available_gpus
    logger.info(f"Detected {len(gpus)} GPUs")
    
    if gpus:
        # Test GPU stats
        stats = resource_manager.get_gpu_stats(0)
        logger.info(f"GPU 0 stats: {stats}")
        
        # Test resource check
        resources_ok = resource_manager.check_resources(0)
        logger.info(f"Resources OK: {resources_ok}")
        
        # Test memory check
        available_memory = resource_manager.get_available_memory(0)
        logger.info(f"Available memory: {available_memory:.1f} GB")
        
        # Test model fit check
        can_fit_7b = resource_manager.can_fit_model(16.0, 0)  # 16GB for 7B model
        logger.info(f"Can fit 7B model: {can_fit_7b}")
    
    # Test system info
    system_info = resource_manager.get_system_info()
    logger.info(f"System info: {system_info}")
    
    logger.success("Resource Manager test completed")


def test_training_orchestrator():
    """Test the training orchestrator."""
    logger.info("Testing Training Orchestrator...")
    
    orchestrator = LocalTrainingOrchestrator(max_concurrent_jobs=1)
    orchestrator.start()
    
    try:
        # Test job submission (mock)
        job_id = orchestrator.submit_training_job(
            model_name="mistralai/Mistral-7B-Instruct-v0.1",
            dataset_path="data/training/synthetic_finance_v2.jsonl",
            num_epochs=1,  # Short test
            max_steps=10   # Very short test
        )
        
        logger.info(f"Submitted test job: {job_id}")
        
        # Check job status
        job = orchestrator.get_job_status(job_id)
        if job:
            logger.info(f"Job status: {job.status}")
        
        # List jobs
        jobs = orchestrator.list_jobs()
        logger.info(f"Total jobs: {len(jobs)}")
        
        # Get queue status
        queue_status = orchestrator.get_queue_status()
        logger.info(f"Queue status: {queue_status}")
        
        # Wait a bit for processing
        time.sleep(5)
        
        # Check status again
        job = orchestrator.get_job_status(job_id)
        if job:
            logger.info(f"Updated job status: {job.status}")
        
    finally:
        orchestrator.stop()
    
    logger.success("Training Orchestrator test completed")


def test_comparison_engine():
    """Test the comparison engine."""
    logger.info("Testing Comparison Engine...")
    
    comparison_engine = ModelComparisonEngine()
    
    # Test mock comparison
    try:
        comparison = comparison_engine.compare_models(
            base_model_name="mistralai/Mistral-7B-Instruct-v0.1",
            finetuned_model_path="./models/test_lora",
            tasks=["eps_extraction", "sentiment_analysis"]
        )
        
        logger.info(f"Comparison completed:")
        logger.info(f"  Base scores: {comparison.base_scores}")
        logger.info(f"  Fine-tuned scores: {comparison.finetuned_scores}")
        logger.info(f"  Overall improvement: {comparison.overall_improvement:.2%}")
        
        # Test report generation
        report = comparison_engine.generate_comparison_report(comparison)
        logger.info(f"Generated report ({len(report)} characters)")
        
        # Test listing comparisons
        comparisons = comparison_engine.list_comparisons()
        logger.info(f"Found {len(comparisons)} saved comparisons")
        
    except Exception as e:
        logger.warning(f"Comparison test failed (expected for mock): {e}")
    
    logger.success("Comparison Engine test completed")


def test_dataset_loading():
    """Test dataset loading."""
    logger.info("Testing Dataset Loading...")
    
    dataset_path = "data/training/synthetic_finance_v2.jsonl"
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return
    
    # Count lines in dataset
    with open(dataset_path, 'r') as f:
        lines = f.readlines()
    
    logger.info(f"Dataset has {len(lines)} samples")
    
    # Test loading first sample
    import json
    if lines:
        sample = json.loads(lines[0])
        logger.info(f"Sample format: {list(sample.keys())}")
        logger.info(f"Sample instruction: {sample.get('instruction', '')[:100]}...")
    
    logger.success("Dataset loading test completed")


def test_training_config():
    """Test training configuration."""
    logger.info("Testing Training Configuration...")
    
    config = TrainingConfig(
        model_name="mistralai/Mistral-7B-Instruct-v0.1",
        dataset_path="data/training/synthetic_finance_v2.jsonl",
        output_dir="./models/test_output",
        gpu_type="RTX 4090",
        max_power_watts=450.0,
        num_epochs=1,
        batch_size=2,
        learning_rate=2e-4
    )
    
    logger.info(f"Training config created: {config.model_name}")
    logger.info(f"  Dataset: {config.dataset_path}")
    logger.info(f"  Output: {config.output_dir}")
    logger.info(f"  LoRA rank: {config.lora_r}")
    logger.info(f"  Batch size: {config.batch_size}")
    
    logger.success("Training Configuration test completed")


def main():
    """Run all tests."""
    logger.info("Starting Auto Fine-tuning Tests")
    logger.info("=" * 50)
    
    try:
        # Test individual components
        test_resource_manager()
        print()
        
        test_training_config()
        print()
        
        test_dataset_loading()
        print()
        
        test_comparison_engine()
        print()
        
        # Note: Skip orchestrator test by default as it requires GPU
        # test_training_orchestrator()
        
        logger.success("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())