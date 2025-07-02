"""
Training Service for Lift-os-LLM microservice.

Integrates with the existing LocalTrainingOrchestrator and provides
API-friendly interfaces for model training, evaluation, and comparison.
"""

import asyncio
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from ..core.logging import logger
from ..core.config import settings

# Import existing training components
try:
    from ..training.local_orchestrator import LocalTrainingOrchestrator
    from ..training.comparison_engine import ModelComparisonEngine
    from ..data.schemas.data_models import TrainingConfig, TrainingJob
    TRAINING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Training components not available: {e}")
    TRAINING_AVAILABLE = False


class TrainingService:
    """Service for managing model training, evaluation, and comparison."""
    
    def __init__(self):
        self.orchestrator = None
        self.comparison_engine = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize training components if available."""
        if TRAINING_AVAILABLE:
            try:
                self.orchestrator = LocalTrainingOrchestrator(max_concurrent_jobs=1)
                self.comparison_engine = ModelComparisonEngine()
                logger.info("Training service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize training components: {e}")
                self.orchestrator = None
                self.comparison_engine = None
        else:
            logger.warning("Training service not available - missing dependencies")
    
    async def submit_training_job(
        self,
        job_id: str,
        user_id: int,
        model_name: str,
        industry: str,
        training_config: Dict[str, Any],
        dataset_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Submit a new training job."""
        if not self.orchestrator:
            raise RuntimeError("Training orchestrator not available")
        
        try:
            # Get dataset path based on industry
            if not dataset_path:
                dataset_path = self._get_industry_dataset_path(industry)
            
            # Create training configuration
            config = TrainingConfig(
                model_name=model_name,
                dataset_path=dataset_path,
                output_dir=f"models/fine-tuned/{job_id}",
                learning_rate=training_config.get("learning_rate", 0.0001),
                num_epochs=training_config.get("epochs", 20),
                batch_size=training_config.get("batch_size", 4),
                max_seq_length=training_config.get("max_seq_length", 512),
                save_steps=training_config.get("save_steps", 100),
                eval_steps=training_config.get("eval_steps", 100),
                warmup_steps=training_config.get("warmup_steps", 10),
                logging_steps=training_config.get("logging_steps", 10)
            )
            
            # Create training job
            training_job = TrainingJob(
                job_id=job_id,
                user_id=user_id,
                config=config,
                status="queued",
                created_at=datetime.utcnow()
            )
            
            # Submit to orchestrator
            await asyncio.get_event_loop().run_in_executor(
                None, self.orchestrator.submit_job, training_job
            )
            
            logger.info(f"Training job {job_id} submitted successfully")
            
            return {
                "job_id": job_id,
                "status": "queued",
                "model_name": model_name,
                "industry": industry,
                "dataset_path": dataset_path,
                "config": training_config
            }
            
        except Exception as e:
            logger.error(f"Failed to submit training job {job_id}: {e}")
            raise
    
    async def get_training_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of a training job."""
        if not self.orchestrator:
            raise RuntimeError("Training orchestrator not available")
        
        try:
            # Check active jobs
            if job_id in self.orchestrator.active_jobs:
                job = self.orchestrator.active_jobs[job_id]
                return {
                    "job_id": job_id,
                    "status": "running",
                    "progress": getattr(job, 'progress', 0.0),
                    "current_epoch": getattr(job, 'current_epoch', 0),
                    "total_epochs": job.config.num_epochs if hasattr(job, 'config') else 20
                }
            
            # Check completed jobs
            if job_id in self.orchestrator.completed_jobs:
                job = self.orchestrator.completed_jobs[job_id]
                return {
                    "job_id": job_id,
                    "status": "completed",
                    "completed_at": job.completed_at.isoformat() if hasattr(job, 'completed_at') else None,
                    "metrics": getattr(job, 'final_metrics', {}),
                    "model_path": f"models/fine-tuned/{job_id}"
                }
            
            # Check failed jobs
            if job_id in self.orchestrator.failed_jobs:
                job = self.orchestrator.failed_jobs[job_id]
                return {
                    "job_id": job_id,
                    "status": "failed",
                    "error": getattr(job, 'error_message', 'Unknown error'),
                    "failed_at": job.failed_at.isoformat() if hasattr(job, 'failed_at') else None
                }
            
            # Job not found
            return {
                "job_id": job_id,
                "status": "not_found",
                "error": "Job not found"
            }
            
        except Exception as e:
            logger.error(f"Failed to get training job status for {job_id}: {e}")
            raise
    
    async def list_training_jobs(self, user_id: int) -> List[Dict[str, Any]]:
        """List training jobs for a user."""
        if not self.orchestrator:
            return []
        
        try:
            jobs = []
            
            # Add active jobs
            for job_id, job in self.orchestrator.active_jobs.items():
                if hasattr(job, 'user_id') and job.user_id == user_id:
                    jobs.append({
                        "job_id": job_id,
                        "status": "running",
                        "model_name": job.config.model_name if hasattr(job, 'config') else "unknown",
                        "created_at": job.created_at.isoformat() if hasattr(job, 'created_at') else None,
                        "progress": getattr(job, 'progress', 0.0)
                    })
            
            # Add completed jobs
            for job_id, job in self.orchestrator.completed_jobs.items():
                if hasattr(job, 'user_id') and job.user_id == user_id:
                    jobs.append({
                        "job_id": job_id,
                        "status": "completed",
                        "model_name": job.config.model_name if hasattr(job, 'config') else "unknown",
                        "created_at": job.created_at.isoformat() if hasattr(job, 'created_at') else None,
                        "completed_at": job.completed_at.isoformat() if hasattr(job, 'completed_at') else None,
                        "metrics": getattr(job, 'final_metrics', {})
                    })
            
            # Add failed jobs
            for job_id, job in self.orchestrator.failed_jobs.items():
                if hasattr(job, 'user_id') and job.user_id == user_id:
                    jobs.append({
                        "job_id": job_id,
                        "status": "failed",
                        "model_name": job.config.model_name if hasattr(job, 'config') else "unknown",
                        "created_at": job.created_at.isoformat() if hasattr(job, 'created_at') else None,
                        "error": getattr(job, 'error_message', 'Unknown error')
                    })
            
            return jobs
            
        except Exception as e:
            logger.error(f"Failed to list training jobs for user {user_id}: {e}")
            return []
    
    async def evaluate_model(
        self,
        model_path: str,
        industry: str,
        evaluation_dataset: Optional[str] = None,
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """Evaluate a trained model."""
        if not self.comparison_engine:
            raise RuntimeError("Comparison engine not available")
        
        try:
            # Get evaluation dataset
            if not evaluation_dataset:
                evaluation_dataset = self._get_industry_dataset_path(industry)
            
            # Default metrics
            if not metrics:
                metrics = ["accuracy", "precision", "recall", "f1_score", "perplexity"]
            
            # Run evaluation (this would integrate with the existing evaluation logic)
            evaluation_results = await asyncio.get_event_loop().run_in_executor(
                None, self._run_model_evaluation, model_path, evaluation_dataset, metrics
            )
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Failed to evaluate model {model_path}: {e}")
            raise
    
    async def compare_models(
        self,
        model_paths: List[str],
        industry: str,
        metrics: List[str] = None,
        test_dataset: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compare multiple models."""
        if not self.comparison_engine:
            raise RuntimeError("Comparison engine not available")
        
        try:
            # Get test dataset
            if not test_dataset:
                test_dataset = self._get_industry_dataset_path(industry)
            
            # Default metrics
            if not metrics:
                metrics = ["accuracy", "latency", "cost", "throughput"]
            
            # Run comparison (this would integrate with the existing comparison logic)
            comparison_results = await asyncio.get_event_loop().run_in_executor(
                None, self._run_model_comparison, model_paths, test_dataset, metrics
            )
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            raise
    
    def _get_industry_dataset_path(self, industry: str) -> str:
        """Get the dataset path for a specific industry."""
        dataset_paths = {
            "finance": "data/training/synthetic_finance_gsib_v3.jsonl",
            "education": "data/training/synthetic_education_v1.jsonl",
            "retail": "data/training/synthetic_retail_v1.jsonl",
            "healthcare": "data/training/synthetic_healthcare_v1.jsonl",
            "multi-industry": "data/training/combined_multi_industry_corpus.jsonl"
        }
        
        dataset_path = dataset_paths.get(industry)
        if not dataset_path:
            raise ValueError(f"Unknown industry: {industry}")
        
        # Check if dataset exists
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset not found: {dataset_path}")
            # Could create a sample dataset or raise an error
        
        return dataset_path
    
    def _run_model_evaluation(self, model_path: str, dataset_path: str, metrics: List[str]) -> Dict[str, Any]:
        """Run model evaluation (placeholder for actual evaluation logic)."""
        # This would integrate with the existing evaluation engine
        # For now, return mock results
        return {
            "model_path": model_path,
            "dataset_path": dataset_path,
            "metrics": {
                "accuracy": 0.92,
                "precision": 0.89,
                "recall": 0.94,
                "f1_score": 0.91,
                "perplexity": 1.8
            },
            "evaluation_date": datetime.utcnow().isoformat()
        }
    
    def _run_model_comparison(self, model_paths: List[str], dataset_path: str, metrics: List[str]) -> Dict[str, Any]:
        """Run model comparison (placeholder for actual comparison logic)."""
        # This would integrate with the existing comparison engine
        # For now, return mock results
        return {
            "models": [
                {
                    "model_path": path,
                    "metrics": {
                        "accuracy": 0.85 + (i * 0.02),
                        "latency_ms": 120 + (i * 10),
                        "cost_per_1k_tokens": 0.002
                    },
                    "rank": i + 1
                }
                for i, path in enumerate(model_paths)
            ],
            "comparison_date": datetime.utcnow().isoformat(),
            "dataset_path": dataset_path
        }
    
    def get_available_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available training datasets."""
        datasets = {
            "finance": {
                "name": "G-SIB Banking Corpus",
                "path": "data/training/synthetic_finance_gsib_v3.jsonl",
                "samples": 30,
                "complexity": "high",
                "description": "Basel III compliance scenarios and financial analysis"
            },
            "education": {
                "name": "Educational Analytics Corpus",
                "path": "data/training/synthetic_education_v1.jsonl",
                "samples": 20,
                "complexity": "medium",
                "description": "K-12 and higher education analytics"
            },
            "retail": {
                "name": "Retail Business Analytics",
                "path": "data/training/synthetic_retail_v1.jsonl",
                "samples": 20,
                "complexity": "medium",
                "description": "Sales, inventory, and customer analysis"
            },
            "healthcare": {
                "name": "Healthcare Analytics Corpus",
                "path": "data/training/synthetic_healthcare_v1.jsonl",
                "samples": 20,
                "complexity": "medium",
                "description": "Patient care and quality metrics"
            },
            "multi-industry": {
                "name": "Combined Multi-Industry Corpus",
                "path": "data/training/combined_multi_industry_corpus.jsonl",
                "samples": 80,
                "complexity": "mixed",
                "description": "Integrated corpus across all industries"
            }
        }
        
        # Check which datasets actually exist
        for industry, info in datasets.items():
            info["exists"] = os.path.exists(info["path"])
        
        return datasets
    
    def is_available(self) -> bool:
        """Check if training service is available."""
        return TRAINING_AVAILABLE and self.orchestrator is not None