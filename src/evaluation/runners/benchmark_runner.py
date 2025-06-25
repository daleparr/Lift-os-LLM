"""
Benchmark runner for LLM Finance Leaderboard.

Orchestrates the complete evaluation pipeline including data retrieval, 
model execution, and scoring.
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

from ...data.schemas.data_models import (
    BenchmarkResult, TaskResult, EvaluationMetrics, 
    TaskComplexity, ModelProvider
)
from ...agents.base_agent import create_agent_pipeline
from ...agents.retriever_agent import create_retriever_agent
from ...data.processors.vector_store import create_vector_store
from ...models.model_loader import load_model, get_model_info
from ...tasks.base_task import BaseTask
from ...tasks.low_complexity.eps_extraction import create_eps_extraction_task
from ..metrics.quality_metrics import calculate_quality_metrics, aggregate_quality_scores
from ...utils.database import save_benchmark_run, save_model_result, save_task_result
from ...utils.logging_config import log_benchmark_start, log_benchmark_end, log_model_evaluation
from ...config.settings import settings


class BenchmarkRunner:
    """Runs comprehensive benchmarks on LLM models."""
    
    def __init__(
        self,
        vector_store=None,
        max_concurrent_evaluations: int = None,
        timeout_minutes: int = None
    ):
        """Initialize benchmark runner."""
        self.vector_store = vector_store or create_vector_store(use_mock=False)
        self.max_concurrent = max_concurrent_evaluations or settings.max_concurrent_evaluations
        self.timeout_minutes = timeout_minutes or settings.evaluation_timeout_minutes
        
        # Initialize available tasks
        self.available_tasks = self._initialize_tasks()
        
        logger.info(f"Initialized BenchmarkRunner with {len(self.available_tasks)} tasks")
    
    def _initialize_tasks(self) -> Dict[str, BaseTask]:
        """Initialize available evaluation tasks."""
        tasks = {}
        
        # Low complexity tasks
        tasks["eps_extraction"] = create_eps_extraction_task()
        
        # TODO: Add more tasks as they are implemented
        # tasks["ratio_identification"] = create_ratio_identification_task()
        # tasks["revenue_analysis"] = create_revenue_analysis_task()
        # tasks["sentiment_classification"] = create_sentiment_classification_task()
        # tasks["target_price_generation"] = create_target_price_generation_task()
        
        return tasks
    
    async def run_benchmark(
        self,
        models: List[str],
        tasks: List[str] = None,
        run_name: str = None,
        description: str = None,
        evaluation_seeds: List[int] = None
    ) -> BenchmarkResult:
        """
        Run complete benchmark evaluation.
        
        Args:
            models: List of model names to evaluate
            tasks: List of task names to run (if None, runs all available)
            run_name: Name for this benchmark run
            description: Description of the benchmark run
            evaluation_seeds: Random seeds for reproducibility
            
        Returns:
            BenchmarkResult with complete evaluation results
        """
        run_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        # Set defaults
        run_name = run_name or f"Benchmark Run {start_time.strftime('%Y%m%d_%H%M%S')}"
        description = description or "Automated benchmark evaluation"
        tasks = tasks or list(self.available_tasks.keys())
        evaluation_seeds = evaluation_seeds or settings.benchmark_seeds
        
        logger.info(f"Starting benchmark run: {run_name}")
        log_benchmark_start(run_id, models, tasks)
        
        # Initialize benchmark result
        benchmark_result = BenchmarkResult(
            run_id=run_id,
            run_name=run_name,
            description=description,
            models_evaluated=models,
            tasks_included=tasks,
            evaluation_seeds=evaluation_seeds,
            start_time=start_time,
            status="running"
        )
        
        # Save initial benchmark run
        save_benchmark_run({
            "run_id": run_id,
            "run_name": run_name,
            "description": description,
            "status": "running",
            "start_time": start_time,
        })
        
        try:
            # Prepare sample documents for tasks
            sample_documents = await self._prepare_sample_documents()
            
            # Run evaluations
            all_task_results = []
            model_metrics = {}
            
            for model_name in models:
                logger.info(f"Evaluating model: {model_name}")
                
                try:
                    # Load model
                    model_llm = self._load_model_safely(model_name)
                    
                    # Create agent pipeline
                    agent_pipeline = create_agent_pipeline(
                        model_name=model_name,
                        model_provider=self._detect_model_provider(model_name)
                    )
                    
                    # Run tasks for this model
                    model_task_results = await self._run_model_tasks(
                        model_name=model_name,
                        agent_pipeline=agent_pipeline,
                        tasks=tasks,
                        sample_documents=sample_documents,
                        evaluation_seeds=evaluation_seeds,
                        run_id=run_id
                    )
                    
                    all_task_results.extend(model_task_results)
                    
                    # Calculate model metrics
                    model_metrics[model_name] = self._calculate_model_metrics(
                        model_name, model_task_results
                    )
                    
                    # Save model results
                    save_model_result({
                        "run_id": run_id,
                        "model_name": model_name,
                        **model_metrics[model_name].__dict__,
                        "created_at": datetime.utcnow()
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate model {model_name}: {e}")
                    # Create failed model metrics
                    model_metrics[model_name] = self._create_failed_model_metrics(model_name)
            
            # Finalize benchmark result
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds() / 60
            
            benchmark_result.task_results = all_task_results
            benchmark_result.model_metrics = model_metrics
            benchmark_result.end_time = end_time
            benchmark_result.total_duration_minutes = duration
            benchmark_result.status = "completed"
            
            # Log completion
            log_benchmark_end(run_id, duration, True)
            logger.info(f"Benchmark completed successfully in {duration:.2f} minutes")
            
            return benchmark_result
            
        except Exception as e:
            # Handle benchmark failure
            benchmark_result.status = "failed"
            benchmark_result.error_message = str(e)
            benchmark_result.end_time = datetime.utcnow()
            
            log_benchmark_end(run_id, 0, False)
            logger.error(f"Benchmark failed: {e}")
            raise
    
    async def _prepare_sample_documents(self) -> List[Dict[str, Any]]:
        """Prepare sample documents for evaluation."""
        # In a real implementation, this would retrieve actual documents
        # For now, we'll create sample financial documents
        
        sample_documents = [
            {
                "id": "sample_10q_jpm_q1_2024",
                "title": "JPMorgan Chase Q1 2024 10-Q Filing",
                "content": """
                JPMORGAN CHASE & CO.
                FORM 10-Q
                For the quarterly period ended March 31, 2024
                
                CONSOLIDATED STATEMENTS OF INCOME
                
                Net interest income: $22.9 billion
                Noninterest income: $15.2 billion
                Total net revenue: $38.1 billion
                
                Net income: $13.4 billion
                Earnings per share:
                Basic: $4.44
                Diluted: $4.44
                
                The Firm reported net income of $13.4 billion, or $4.44 per share, 
                compared with net income of $12.6 billion, or $4.10 per share, 
                in the prior year quarter.
                
                Return on equity was 17% and return on tangible common equity was 22%.
                The Firm's CET1 ratio was 15.0%.
                """,
                "document_type": "sec_10q",
                "ticker": "JPM",
                "score": 0.95
            },
            {
                "id": "sample_earnings_transcript_jpm_q1_2024",
                "title": "JPMorgan Chase Q1 2024 Earnings Call Transcript",
                "content": """
                JPMorgan Chase Q1 2024 Earnings Call
                April 12, 2024
                
                Jamie Dimon, CEO: Good morning everyone. We delivered strong results 
                this quarter with earnings per share of $4.44, driven by robust 
                performance across our businesses.
                
                Our net interest income of $22.9 billion reflects the benefit of 
                higher rates, while credit costs remain well-controlled at $1.9 billion.
                
                Looking ahead, we remain cautious about the economic environment 
                but are well-positioned with strong capital and liquidity.
                
                Jeremy Barnum, CFO: Our CET1 ratio of 15.0% remains well above 
                regulatory requirements. We continue to see healthy loan growth 
                and strong deposit levels.
                """,
                "document_type": "earnings_transcript",
                "ticker": "JPM",
                "score": 0.90
            }
        ]
        
        return sample_documents
    
    def _load_model_safely(self, model_name: str):
        """Load model with error handling."""
        try:
            # Import model loader
            from src.models.model_loader import ModelLoader
            
            # Initialize model loader
            model_loader = ModelLoader()
            
            # Load the model using the real model loader
            logger.info(f"Loading model: {model_name}")
            model = model_loader.load_model(model_name)
            
            if model is None:
                logger.warning(f"Model {model_name} could not be loaded, using fallback")
                return None
            
            logger.info(f"Successfully loaded model: {model_name}")
            return model
            
        except ImportError as e:
            logger.error(f"Model loader not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None
    
    def _detect_model_provider(self, model_name: str) -> ModelProvider:
        """Detect model provider from name."""
        model_name_lower = model_name.lower()
        
        if "gpt" in model_name_lower or "openai" in model_name_lower:
            return ModelProvider.OPENAI
        elif "claude" in model_name_lower or "anthropic" in model_name_lower:
            return ModelProvider.ANTHROPIC
        else:
            return ModelProvider.HUGGINGFACE
    
    async def _run_model_tasks(
        self,
        model_name: str,
        agent_pipeline,
        tasks: List[str],
        sample_documents: List[Dict[str, Any]],
        evaluation_seeds: List[int],
        run_id: str
    ) -> List[TaskResult]:
        """Run all tasks for a specific model."""
        task_results = []
        
        for task_name in tasks:
            if task_name not in self.available_tasks:
                logger.warning(f"Task {task_name} not available, skipping")
                continue
            
            task = self.available_tasks[task_name]
            
            for seed in evaluation_seeds:
                try:
                    logger.info(f"Running {task_name} for {model_name} (seed: {seed})")
                    
                    # Prepare task context
                    context = task.prepare_context(
                        documents=sample_documents,
                        expected_eps="4.44" if task_name == "eps_extraction" else None
                    )
                    
                    # Execute task
                    start_time = time.time()
                    task_result = task.execute(
                        agent_pipeline=agent_pipeline,
                        context=context,
                        evaluation_seed=seed
                    )
                    execution_time = (time.time() - start_time) * 1000
                    
                    # Log execution
                    log_model_evaluation(
                        run_id=run_id,
                        model_name=model_name,
                        task_id=task_result.task_id,
                        latency_ms=execution_time,
                        tokens_used=task_result.total_tokens_used,
                        cost_usd=task_result.cost_usd,
                        success=task_result.task_completed
                    )
                    
                    # Save task result
                    save_task_result({
                        "run_id": run_id,
                        "task_id": task_result.task_id,
                        "model_name": model_name,
                        "task_complexity": task_result.complexity.value,
                        "prompt": task_result.prompt,
                        "expected_answer": task_result.expected_answer,
                        "model_response": task_result.model_response,
                        "exact_match_score": task_result.exact_match_score,
                        "f1_score": task_result.f1_score,
                        "rouge_1_score": task_result.rouge_1_score,
                        "rouge_2_score": task_result.rouge_2_score,
                        "bleu_score": task_result.bleu_score,
                        "human_rating": task_result.human_rating,
                        "latency_ms": task_result.total_latency_ms,
                        "tokens_used": task_result.total_tokens_used,
                        "cost_usd": task_result.cost_usd,
                        "task_completed": task_result.task_completed,
                        "evaluation_seed": seed,
                        "created_at": datetime.utcnow()
                    })
                    
                    task_results.append(task_result)
                    
                except Exception as e:
                    logger.error(f"Task {task_name} failed for {model_name}: {e}")
                    # Create failed task result
                    failed_result = self._create_failed_task_result(
                        task_name, model_name, seed, str(e)
                    )
                    task_results.append(failed_result)
        
        return task_results
    
    def _calculate_model_metrics(
        self,
        model_name: str,
        task_results: List[TaskResult]
    ) -> EvaluationMetrics:
        """Calculate aggregated metrics for a model."""
        if not task_results:
            return self._create_failed_model_metrics(model_name)
        
        # Separate by complexity
        low_results = [r for r in task_results if r.complexity == TaskComplexity.LOW]
        medium_results = [r for r in task_results if r.complexity == TaskComplexity.MEDIUM]
        high_results = [r for r in task_results if r.complexity == TaskComplexity.HIGH]
        
        # Calculate scores by complexity
        low_score = self._calculate_complexity_score(low_results) if low_results else 0.0
        medium_score = self._calculate_complexity_score(medium_results) if medium_results else 0.0
        high_score = self._calculate_complexity_score(high_results) if high_results else 0.0
        
        # Calculate overall quality score (weighted)
        overall_quality = (
            0.3 * low_score +
            0.4 * medium_score +
            0.3 * high_score
        )
        
        # Calculate efficiency metrics
        avg_latency = sum(r.total_latency_ms for r in task_results) / len(task_results)
        avg_tokens = sum(r.total_tokens_used for r in task_results) / len(task_results)
        avg_cost = sum(r.cost_usd for r in task_results) / len(task_results)
        
        # Efficiency score (inverse of normalized latency)
        efficiency_score = max(0.0, 1.0 - (avg_latency / 10000))  # Normalize by 10s
        
        # Success metrics
        completion_rate = sum(1 for r in task_results if r.task_completed) / len(task_results)
        avg_tool_success = sum(r.tool_call_success_rate for r in task_results) / len(task_results)
        hallucination_rate = sum(1 for r in task_results if r.hallucination_detected) / len(task_results)
        
        # Final composite score
        final_score = 0.9 * overall_quality + 0.1 * efficiency_score
        
        return EvaluationMetrics(
            model_name=model_name,
            total_tasks=len(task_results),
            completed_tasks=sum(1 for r in task_results if r.task_completed),
            low_complexity_score=low_score,
            medium_complexity_score=medium_score,
            high_complexity_score=high_score,
            overall_quality_score=overall_quality,
            avg_latency_ms=avg_latency,
            avg_tokens_per_task=avg_tokens,
            avg_cost_per_task=avg_cost,
            efficiency_score=efficiency_score,
            completion_rate=completion_rate,
            avg_tool_success_rate=avg_tool_success,
            hallucination_rate=hallucination_rate,
            final_score=final_score,
            evaluation_date=datetime.utcnow(),
            evaluation_seeds=settings.benchmark_seeds
        )
    
    def _calculate_complexity_score(self, results: List[TaskResult]) -> float:
        """Calculate average score for a complexity level."""
        if not results:
            return 0.0
        
        # Use F1 score as primary metric, fall back to exact match
        scores = []
        for result in results:
            if result.f1_score is not None:
                scores.append(result.f1_score)
            elif result.exact_match_score is not None:
                scores.append(result.exact_match_score)
            else:
                scores.append(0.0)
        
        return sum(scores) / len(scores)
    
    def _create_failed_task_result(
        self,
        task_name: str,
        model_name: str,
        seed: int,
        error_message: str
    ) -> TaskResult:
        """Create a failed task result."""
        return TaskResult(
            task_id=str(uuid.uuid4()),
            task_name=task_name,
            complexity=TaskComplexity.LOW,  # Default
            model_name=model_name,
            prompt="Failed to generate prompt",
            expected_answer="Failed to get expected answer",
            model_response=f"Task failed: {error_message}",
            agent_responses=[],
            exact_match_score=0.0,
            f1_score=0.0,
            total_latency_ms=0.0,
            total_tokens_used=0,
            cost_usd=0.0,
            task_completed=False,
            evaluation_seed=seed,
            created_at=datetime.utcnow()
        )
    
    def _create_failed_model_metrics(self, model_name: str) -> EvaluationMetrics:
        """Create failed model metrics."""
        return EvaluationMetrics(
            model_name=model_name,
            total_tasks=0,
            completed_tasks=0,
            low_complexity_score=0.0,
            medium_complexity_score=0.0,
            high_complexity_score=0.0,
            overall_quality_score=0.0,
            avg_latency_ms=0.0,
            avg_tokens_per_task=0.0,
            avg_cost_per_task=0.0,
            efficiency_score=0.0,
            completion_rate=0.0,
            avg_tool_success_rate=0.0,
            hallucination_rate=1.0,
            final_score=0.0,
            evaluation_date=datetime.utcnow(),
            evaluation_seeds=[]
        )


def create_benchmark_runner(**kwargs) -> BenchmarkRunner:
    """Factory function to create a benchmark runner."""
    return BenchmarkRunner(**kwargs)


async def run_quick_benchmark(
    models: List[str] = None,
    tasks: List[str] = None
) -> BenchmarkResult:
    """Run a quick benchmark with default settings."""
    models = models or ["mistral-7b", "finma-7b"]
    tasks = tasks or ["eps_extraction"]
    
    runner = create_benchmark_runner()
    return await runner.run_benchmark(
        models=models,
        tasks=tasks,
        run_name="Quick Benchmark",
        description="Quick evaluation run"
    )