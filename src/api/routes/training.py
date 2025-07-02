"""
Training API routes for Lift-os-LLM microservice.

Provides endpoints for model fine-tuning, evaluation, and comparison
across multiple industries (Finance, Education, Retail, Healthcare).
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.security import HTTPBearer
import uuid
from datetime import datetime

from ...core.security import get_current_user, verify_api_key
from ...core.logging import logger
from ...models.entities import User
from ...models.requests import (
    TrainingJobRequest, ModelEvaluationRequest, ModelComparisonRequest
)
from ...models.responses import (
    TrainingJobResponse, ModelEvaluationResponse, ModelComparisonResponse,
    TrainingJobListResponse
)

router = APIRouter(prefix="/training", tags=["Model Training & Evaluation"])
security = HTTPBearer()


@router.post(
    "/jobs/submit",
    response_model=TrainingJobResponse,
    summary="Submit fine-tuning job",
    description="Submit a model fine-tuning job for specific industry or multi-industry training."
)
async def submit_training_job(
    request: TrainingJobRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
) -> TrainingJobResponse:
    """
    Submit a new fine-tuning job.
    
    - **model_name**: Base model to fine-tune (e.g., "gpt-3.5-turbo", "llama-2-7b")
    - **industry**: Target industry (finance, education, retail, healthcare, multi-industry)
    - **training_config**: Training parameters (learning_rate, epochs, batch_size, etc.)
    - **dataset_path**: Optional custom dataset path
    """
    try:
        job_id = str(uuid.uuid4())
        logger.info(f"Submitting training job {job_id} for user {current_user.id}")
        
        # Validate industry selection
        valid_industries = ["finance", "education", "retail", "healthcare", "multi-industry"]
        if request.industry not in valid_industries:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid industry. Must be one of: {valid_industries}"
            )
        
        # Validate model availability
        supported_models = [
            "gpt-3.5-turbo", "gpt-4", "llama-2-7b", "llama-2-13b", 
            "mistral-7b", "phi-2", "gemma-7b"
        ]
        if request.model_name not in supported_models:
            raise HTTPException(
                status_code=400,
                detail=f"Model {request.model_name} not supported. Available: {supported_models}"
            )
        
        # Create training job
        training_job = {
            "job_id": job_id,
            "user_id": current_user.id,
            "model_name": request.model_name,
            "industry": request.industry,
            "status": "queued",
            "training_config": request.training_config,
            "dataset_path": request.dataset_path,
            "created_at": datetime.utcnow(),
            "estimated_duration_minutes": estimate_training_duration(
                request.model_name, request.industry, request.training_config
            )
        }
        
        # Add to background processing queue
        background_tasks.add_task(
            process_training_job,
            job_id=job_id,
            user_id=current_user.id,
            training_job=training_job
        )
        
        logger.info(
            f"Training job {job_id} submitted successfully",
            extra={
                "job_id": job_id,
                "user_id": current_user.id,
                "model_name": request.model_name,
                "industry": request.industry
            }
        )
        
        return TrainingJobResponse(
            success=True,
            data={
                "job_id": job_id,
                "status": "queued",
                "model_name": request.model_name,
                "industry": request.industry,
                "estimated_duration_minutes": training_job["estimated_duration_minutes"],
                "created_at": training_job["created_at"].isoformat()
            },
            message=f"Training job {job_id} submitted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit training job for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit training job: {str(e)}"
        )


@router.get(
    "/jobs",
    response_model=TrainingJobListResponse,
    summary="List training jobs",
    description="Get a list of training jobs for the current user with optional filtering."
)
async def list_training_jobs(
    status: Optional[str] = Query(None, description="Filter by job status"),
    industry: Optional[str] = Query(None, description="Filter by industry"),
    limit: int = Query(50, le=100, description="Maximum number of jobs to return"),
    offset: int = Query(0, description="Number of jobs to skip"),
    current_user: User = Depends(get_current_user)
) -> TrainingJobListResponse:
    """
    List training jobs for the current user.
    
    - **status**: Optional filter by job status (queued, running, completed, failed)
    - **industry**: Optional filter by industry
    - **limit**: Maximum number of jobs to return (max 100)
    - **offset**: Number of jobs to skip for pagination
    """
    try:
        logger.info(f"Listing training jobs for user {current_user.id}")
        
        # This would integrate with the existing training orchestrator
        # For now, return sample data
        sample_jobs = [
            {
                "job_id": "train-123",
                "model_name": "gpt-3.5-turbo",
                "industry": "finance",
                "status": "completed",
                "created_at": "2024-01-01T10:00:00Z",
                "completed_at": "2024-01-01T11:30:00Z",
                "duration_minutes": 90,
                "metrics": {
                    "final_loss": 0.15,
                    "accuracy": 0.92,
                    "perplexity": 1.8
                }
            },
            {
                "job_id": "train-124",
                "model_name": "llama-2-7b",
                "industry": "multi-industry",
                "status": "running",
                "created_at": "2024-01-01T12:00:00Z",
                "progress": 0.65,
                "current_epoch": 13,
                "total_epochs": 20
            }
        ]
        
        # Apply filters
        if status:
            sample_jobs = [job for job in sample_jobs if job["status"] == status]
        if industry:
            sample_jobs = [job for job in sample_jobs if job["industry"] == industry]
        
        # Apply pagination
        total_count = len(sample_jobs)
        jobs = sample_jobs[offset:offset + limit]
        
        return TrainingJobListResponse(
            success=True,
            data={
                "jobs": jobs,
                "total_count": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count
            },
            message="Training jobs retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to list training jobs for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve training jobs: {str(e)}"
        )


@router.get(
    "/jobs/{job_id}",
    response_model=TrainingJobResponse,
    summary="Get training job details",
    description="Get detailed information about a specific training job."
)
async def get_training_job(
    job_id: str,
    current_user: User = Depends(get_current_user)
) -> TrainingJobResponse:
    """
    Get detailed information about a training job.
    
    - **job_id**: The ID of the training job to retrieve
    """
    try:
        logger.info(f"Getting training job {job_id} for user {current_user.id}")
        
        # This would integrate with the existing training orchestrator
        # For now, return sample data
        job_details = {
            "job_id": job_id,
            "user_id": current_user.id,
            "model_name": "gpt-3.5-turbo",
            "industry": "finance",
            "status": "completed",
            "created_at": "2024-01-01T10:00:00Z",
            "started_at": "2024-01-01T10:05:00Z",
            "completed_at": "2024-01-01T11:30:00Z",
            "duration_minutes": 85,
            "training_config": {
                "learning_rate": 0.0001,
                "epochs": 20,
                "batch_size": 4,
                "max_seq_length": 512
            },
            "metrics": {
                "final_loss": 0.15,
                "accuracy": 0.92,
                "perplexity": 1.8,
                "training_samples": 30,
                "validation_samples": 10
            },
            "model_path": f"models/fine-tuned/{job_id}",
            "logs": [
                {"timestamp": "2024-01-01T10:05:00Z", "message": "Training started"},
                {"timestamp": "2024-01-01T10:30:00Z", "message": "Epoch 10/20 completed, loss: 0.25"},
                {"timestamp": "2024-01-01T11:00:00Z", "message": "Epoch 20/20 completed, loss: 0.15"},
                {"timestamp": "2024-01-01T11:30:00Z", "message": "Training completed successfully"}
            ]
        }
        
        return TrainingJobResponse(
            success=True,
            data=job_details,
            message=f"Training job {job_id} details retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to get training job {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve training job: {str(e)}"
        )


@router.post(
    "/evaluate",
    response_model=ModelEvaluationResponse,
    summary="Evaluate model performance",
    description="Evaluate a fine-tuned model against baseline and industry benchmarks."
)
async def evaluate_model(
    request: ModelEvaluationRequest,
    current_user: User = Depends(get_current_user)
) -> ModelEvaluationResponse:
    """
    Evaluate model performance against benchmarks.
    
    - **model_path**: Path to the fine-tuned model
    - **industry**: Industry to evaluate against
    - **evaluation_dataset**: Optional custom evaluation dataset
    - **metrics**: Specific metrics to evaluate (accuracy, latency, cost, etc.)
    """
    try:
        logger.info(f"Evaluating model for user {current_user.id}")
        
        # This would integrate with the existing evaluation engine
        evaluation_results = {
            "model_path": request.model_path,
            "industry": request.industry,
            "evaluation_date": datetime.utcnow().isoformat(),
            "metrics": {
                "accuracy": 0.92,
                "precision": 0.89,
                "recall": 0.94,
                "f1_score": 0.91,
                "perplexity": 1.8,
                "latency_ms": 150,
                "cost_per_1k_tokens": 0.002,
                "throughput_tokens_per_second": 45
            },
            "benchmark_comparison": {
                "baseline_model": "gpt-3.5-turbo",
                "accuracy_improvement": 0.08,
                "latency_improvement": -0.02,  # Negative means slower
                "cost_improvement": 0.15,  # Positive means cheaper
                "overall_score": 8.5
            },
            "industry_ranking": {
                "rank": 3,
                "total_models": 15,
                "percentile": 85
            },
            "detailed_results": [
                {
                    "test_case": "Financial calculation accuracy",
                    "score": 0.95,
                    "baseline_score": 0.87,
                    "improvement": 0.08
                },
                {
                    "test_case": "Regulatory compliance understanding",
                    "score": 0.89,
                    "baseline_score": 0.82,
                    "improvement": 0.07
                }
            ]
        }
        
        return ModelEvaluationResponse(
            success=True,
            data=evaluation_results,
            message="Model evaluation completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Model evaluation failed for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Model evaluation failed: {str(e)}"
        )


@router.post(
    "/compare",
    response_model=ModelComparisonResponse,
    summary="Compare multiple models",
    description="Compare performance of multiple models across accuracy, cost, and latency metrics."
)
async def compare_models(
    request: ModelComparisonRequest,
    current_user: User = Depends(get_current_user)
) -> ModelComparisonResponse:
    """
    Compare multiple models across various metrics.
    
    - **model_paths**: List of model paths to compare
    - **industry**: Industry context for comparison
    - **metrics**: Specific metrics to compare
    - **test_dataset**: Optional test dataset for comparison
    """
    try:
        logger.info(f"Comparing models for user {current_user.id}")
        
        # This would integrate with the existing comparison engine
        comparison_results = {
            "comparison_id": str(uuid.uuid4()),
            "industry": request.industry,
            "comparison_date": datetime.utcnow().isoformat(),
            "models": [
                {
                    "model_path": "models/gpt-3.5-turbo-baseline",
                    "model_name": "GPT-3.5 Turbo (Baseline)",
                    "metrics": {
                        "accuracy": 0.84,
                        "latency_ms": 120,
                        "cost_per_1k_tokens": 0.002,
                        "throughput": 50
                    },
                    "rank": 2
                },
                {
                    "model_path": "models/fine-tuned/train-123",
                    "model_name": "GPT-3.5 Turbo (Fine-tuned)",
                    "metrics": {
                        "accuracy": 0.92,
                        "latency_ms": 150,
                        "cost_per_1k_tokens": 0.002,
                        "throughput": 45
                    },
                    "rank": 1
                }
            ],
            "summary": {
                "best_accuracy": "GPT-3.5 Turbo (Fine-tuned)",
                "best_latency": "GPT-3.5 Turbo (Baseline)",
                "best_cost": "Tie",
                "recommended_model": "GPT-3.5 Turbo (Fine-tuned)",
                "recommendation_reason": "Best overall accuracy with acceptable latency trade-off"
            },
            "detailed_comparison": {
                "accuracy_scores": [0.84, 0.92],
                "latency_scores": [120, 150],
                "cost_scores": [0.002, 0.002],
                "overall_scores": [7.8, 8.5]
            }
        }
        
        return ModelComparisonResponse(
            success=True,
            data=comparison_results,
            message="Model comparison completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Model comparison failed for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Model comparison failed: {str(e)}"
        )


@router.get(
    "/datasets",
    summary="List available training datasets",
    description="Get a list of available training datasets for different industries."
)
async def list_datasets(
    industry: Optional[str] = Query(None, description="Filter by industry"),
    current_user: User = Depends(get_current_user)
):
    """List available training datasets."""
    try:
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
        
        if industry:
            if industry in datasets:
                datasets = {industry: datasets[industry]}
            else:
                datasets = {}
        
        return {
            "success": True,
            "data": {
                "datasets": datasets,
                "total_count": len(datasets)
            },
            "message": "Datasets retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to list datasets: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve datasets: {str(e)}"
        )


# Helper functions
def estimate_training_duration(model_name: str, industry: str, config: Dict[str, Any]) -> int:
    """Estimate training duration in minutes."""
    base_times = {
        "gpt-3.5-turbo": 60,
        "gpt-4": 120,
        "llama-2-7b": 90,
        "llama-2-13b": 150,
        "mistral-7b": 75,
        "phi-2": 45,
        "gemma-7b": 80
    }
    
    industry_multipliers = {
        "finance": 1.2,
        "education": 1.0,
        "retail": 1.0,
        "healthcare": 1.1,
        "multi-industry": 1.5
    }
    
    base_time = base_times.get(model_name, 90)
    multiplier = industry_multipliers.get(industry, 1.0)
    epochs = config.get("epochs", 20)
    
    return int(base_time * multiplier * (epochs / 20))


async def process_training_job(job_id: str, user_id: int, training_job: Dict[str, Any]):
    """Process training job in background."""
    try:
        logger.info(f"Processing training job {job_id} for user {user_id}")
        
        # This would integrate with the existing LocalTrainingOrchestrator
        # For now, just log the completion
        logger.info(f"Training job {job_id} completed for user {user_id}")
        
    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}", exc_info=True)