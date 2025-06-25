"""
Local training orchestrator for managing fine-tuning jobs.
"""

import os
import uuid
import json
import threading
from typing import Dict, List, Optional, Callable
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty

from loguru import logger

from .lora_trainer import LoRATrainer
from .resource_manager import LocalResourceManager
from ..data.schemas.data_models import TrainingConfig, TrainingJob


class LocalTrainingOrchestrator:
    """Orchestrates local training jobs with queue management."""
    
    def __init__(self, max_concurrent_jobs: int = 1):
        """Initialize training orchestrator."""
        self.max_concurrent_jobs = max_concurrent_jobs
        self.job_queue = Queue()
        self.active_jobs = {}
        self.completed_jobs = {}
        self.failed_jobs = {}
        
        # Resource manager
        self.resource_manager = LocalResourceManager()
        
        # Worker threads
        self.workers = []
        self.running = False
        
        # Job status callbacks
        self.status_callbacks = []
        
        logger.info(f"Initialized LocalTrainingOrchestrator with {max_concurrent_jobs} max concurrent jobs")
    
    def start(self) -> None:
        """Start the orchestrator workers."""
        if self.running:
            logger.warning("Orchestrator already running")
            return
        
        self.running = True
        self.resource_manager.start_monitoring()
        
        # Start worker threads
        for i in range(self.max_concurrent_jobs):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Started {len(self.workers)} worker threads")
    
    def stop(self) -> None:
        """Stop the orchestrator."""
        self.running = False
        self.resource_manager.stop_monitoring()
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=10)
        
        logger.info("Stopped training orchestrator")
    
    def submit_training_job(
        self,
        model_name: str,
        dataset_path: str,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> str:
        """Submit a new training job."""
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Create output directory
        if output_dir is None:
            output_dir = f"./models/auto_finetuned/{job_id}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create training configuration
        config = TrainingConfig(
            model_name=model_name,
            dataset_path=dataset_path,
            output_dir=output_dir,
            gpu_type="auto",
            **kwargs
        )
        
        # Create training job
        job = TrainingJob(
            job_id=job_id,
            model_name=model_name,
            dataset_name=os.path.basename(dataset_path),
            status="pending",
            created_at=datetime.now()
        )
        
        # Validate resources
        if not self._validate_resources(config):
            job.status = "failed"
            job.error_message = "Insufficient resources for training"
            self.failed_jobs[job_id] = job
            logger.error(f"Job {job_id} failed: Insufficient resources")
            return job_id
        
        # Add to queue
        self.job_queue.put((job, config))
        logger.info(f"Submitted training job {job_id} for model {model_name}")
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[TrainingJob]:
        """Get status of a training job."""
        # Check active jobs
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        
        # Check completed jobs
        if job_id in self.completed_jobs:
            return self.completed_jobs[job_id]
        
        # Check failed jobs
        if job_id in self.failed_jobs:
            return self.failed_jobs[job_id]
        
        return None
    
    def list_jobs(self, status: Optional[str] = None) -> List[TrainingJob]:
        """List all jobs, optionally filtered by status."""
        all_jobs = []
        
        # Collect all jobs
        all_jobs.extend(self.active_jobs.values())
        all_jobs.extend(self.completed_jobs.values())
        all_jobs.extend(self.failed_jobs.values())
        
        # Filter by status if specified
        if status:
            all_jobs = [job for job in all_jobs if job.status == status]
        
        # Sort by creation time
        all_jobs.sort(key=lambda x: x.created_at, reverse=True)
        
        return all_jobs
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job."""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = "cancelled"
            job.error_message = "Job cancelled by user"
            
            # Move to failed jobs
            self.failed_jobs[job_id] = job
            del self.active_jobs[job_id]
            
            logger.info(f"Cancelled job {job_id}")
            return True
        
        return False
    
    def add_status_callback(self, callback: Callable[[TrainingJob], None]) -> None:
        """Add a callback for job status updates."""
        self.status_callbacks.append(callback)
    
    def _validate_resources(self, config: TrainingConfig) -> bool:
        """Validate that resources are available for training."""
        # Check GPU availability
        if not self.resource_manager.available_gpus:
            logger.error("No GPUs available")
            return False
        
        # Check memory requirements
        model_size_gb = self._estimate_model_memory(config.model_name)
        if not self.resource_manager.can_fit_model(model_size_gb):
            logger.error(f"Insufficient GPU memory for model {config.model_name}")
            return False
        
        # Check thermal and power limits
        if not self.resource_manager.check_resources():
            logger.error("GPU resources not within safe limits")
            return False
        
        return True
    
    def _estimate_model_memory(self, model_name: str) -> float:
        """Estimate model memory requirements in GB."""
        # Simple estimation based on model name
        if "7B" in model_name or "7b" in model_name:
            return 16.0  # GB for 7B model with LoRA
        elif "13B" in model_name or "13b" in model_name:
            return 24.0  # GB for 13B model with LoRA
        elif "3B" in model_name or "3b" in model_name:
            return 8.0   # GB for 3B model with LoRA
        else:
            return 16.0  # Default assumption
    
    def _worker_loop(self, worker_id: int) -> None:
        """Worker loop for processing training jobs."""
        logger.info(f"Started worker {worker_id}")
        
        while self.running:
            try:
                # Get job from queue (with timeout)
                job, config = self.job_queue.get(timeout=1.0)
                
                # Move to active jobs
                job.status = "running"
                job.started_at = datetime.now()
                self.active_jobs[job.job_id] = job
                
                logger.info(f"Worker {worker_id} starting job {job.job_id}")
                self._notify_status_change(job)
                
                # Run training
                try:
                    output_path = self._run_training(job, config)
                    
                    # Training completed successfully
                    job.status = "completed"
                    job.completed_at = datetime.now()
                    job.output_model_path = output_path
                    
                    # Move to completed jobs
                    self.completed_jobs[job.job_id] = job
                    del self.active_jobs[job.job_id]
                    
                    logger.info(f"Job {job.job_id} completed successfully")
                    
                except Exception as e:
                    # Training failed
                    job.status = "failed"
                    job.completed_at = datetime.now()
                    job.error_message = str(e)
                    
                    # Move to failed jobs
                    self.failed_jobs[job.job_id] = job
                    del self.active_jobs[job.job_id]
                    
                    logger.error(f"Job {job.job_id} failed: {e}")
                
                self._notify_status_change(job)
                
            except Empty:
                # No jobs in queue, continue
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.info(f"Worker {worker_id} stopped")
    
    def _run_training(self, job: TrainingJob, config: TrainingConfig) -> str:
        """Run the actual training."""
        # Create trainer
        trainer = LoRATrainer(config)
        
        # Update job progress callback
        def update_progress(step: int, total_steps: int, loss: float):
            job.current_step = step
            job.total_steps = total_steps
            job.loss = loss
            job.progress = step / total_steps if total_steps > 0 else 0.0
            
            # Update GPU stats
            stats = self.resource_manager.get_gpu_stats()
            job.gpu_utilization = stats.get("utilization", 0.0)
            job.power_consumption = stats.get("power_usage", 0.0)
            job.temperature = stats.get("temperature", 0.0)
            
            self._notify_status_change(job)
        
        # Run training
        output_path = trainer.train(job)
        
        # Validate model
        validation_results = trainer.validate_model(output_path)
        logger.info(f"Validation results: {validation_results}")
        
        return output_path
    
    def _notify_status_change(self, job: TrainingJob) -> None:
        """Notify all callbacks of job status change."""
        for callback in self.status_callbacks:
            try:
                callback(job)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")
    
    def get_queue_status(self) -> Dict:
        """Get current queue status."""
        return {
            "pending_jobs": self.job_queue.qsize(),
            "active_jobs": len(self.active_jobs),
            "completed_jobs": len(self.completed_jobs),
            "failed_jobs": len(self.failed_jobs),
            "total_jobs": len(self.active_jobs) + len(self.completed_jobs) + len(self.failed_jobs)
        }
    
    def get_resource_status(self) -> Dict:
        """Get current resource status."""
        return {
            "gpu_stats": self.resource_manager.get_current_stats(),
            "system_info": self.resource_manager.get_system_info()
        }