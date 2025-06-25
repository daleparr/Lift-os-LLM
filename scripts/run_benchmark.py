#!/usr/bin/env python3
"""
Benchmark runner script for LLM Finance Leaderboard.

This script demonstrates how to run benchmarks and collect results.
"""

import sys
import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import uuid

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.settings import settings
from src.utils.logging_config import setup_logging, ContextualLogger, log_benchmark_start, log_benchmark_end
from src.utils.database import init_database, save_benchmark_run, save_model_result
from src.data.schemas.data_models import BenchmarkResult, TaskComplexity, ModelProvider
from src.agents.base_agent import create_agent_pipeline


class MockBenchmarkRunner:
    """Mock benchmark runner for demonstration purposes."""
    
    def __init__(self):
        self.run_id = str(uuid.uuid4())
        self.logger = ContextualLogger(run_id=self.run_id)
        
        # Sample models for demonstration
        self.available_models = {
            "mistral-7b": {
                "name": "mistralai/Mistral-7B-Instruct-v0.1",
                "display_name": "Mistral 7B Instruct",
                "provider": ModelProvider.HUGGINGFACE,
                "parameters": "7B"
            },
            "llama2-13b": {
                "name": "meta-llama/Llama-2-13b-chat-hf", 
                "display_name": "Llama 2 13B Chat",
                "provider": ModelProvider.HUGGINGFACE,
                "parameters": "13B"
            },
            "finma-7b": {
                "name": "FinMA-7B",
                "display_name": "FinMA 7B",
                "provider": ModelProvider.HUGGINGFACE,
                "parameters": "7B"
            }
        }
        
        # Sample tasks
        self.available_tasks = {
            "eps_extraction": {
                "name": "EPS Extraction",
                "complexity": TaskComplexity.LOW,
                "description": "Extract earnings per share from 10-Q filings"
            },
            "revenue_analysis": {
                "name": "Revenue Analysis", 
                "complexity": TaskComplexity.MEDIUM,
                "description": "Analyze revenue drivers across quarters"
            },
            "target_price": {
                "name": "Target Price Generation",
                "complexity": TaskComplexity.HIGH,
                "description": "Generate bull/bear cases with target prices"
            }
        }
    
    async def run_benchmark(
        self,
        models: List[str],
        tasks: List[str],
        run_name: str = "Demo Benchmark",
        description: str = "Demonstration benchmark run"
    ) -> BenchmarkResult:
        """Run benchmark evaluation."""
        
        start_time = datetime.utcnow()
        self.logger.info(f"Starting benchmark run: {run_name}")
        
        # Log benchmark start
        log_benchmark_start(self.run_id, models, tasks)
        
        # Initialize benchmark result
        benchmark_result = BenchmarkResult(
            run_id=self.run_id,
            run_name=run_name,
            description=description,
            models_evaluated=models,
            tasks_included=tasks,
            evaluation_seeds=settings.benchmark_seeds,
            start_time=start_time,
            status="running"
        )
        
        # Save initial benchmark run to database
        save_benchmark_run({
            "run_id": self.run_id,
            "run_name": run_name,
            "description": description,
            "status": "running",
            "start_time": start_time,
        })
        
        try:
            # Run evaluations for each model
            for model_key in models:
                if model_key not in self.available_models:
                    self.logger.warning(f"Unknown model: {model_key}")
                    continue
                
                model_info = self.available_models[model_key]
                self.logger.info(f"Evaluating model: {model_info['display_name']}")
                
                # Simulate model evaluation
                model_results = await self._evaluate_model(model_info, tasks)
                
                # Save model results
                save_model_result({
                    "run_id": self.run_id,
                    "model_name": model_info["display_name"],
                    "final_score": model_results["final_score"],
                    "quality_score": model_results["quality_score"],
                    "efficiency_score": model_results["efficiency_score"],
                    "low_complexity_score": model_results["low_score"],
                    "medium_complexity_score": model_results["medium_score"],
                    "high_complexity_score": model_results["high_score"],
                    "avg_latency_ms": model_results["avg_latency_ms"],
                    "avg_cost_per_task": model_results["avg_cost_per_task"],
                    "completion_rate": model_results["completion_rate"],
                    "created_at": datetime.utcnow()
                })
            
            # Mark as completed
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds() / 60
            
            benchmark_result.end_time = end_time
            benchmark_result.total_duration_minutes = duration
            benchmark_result.status = "completed"
            
            # Log completion
            log_benchmark_end(self.run_id, duration, True)
            self.logger.info(f"Benchmark completed successfully in {duration:.2f} minutes")
            
            return benchmark_result
            
        except Exception as e:
            # Handle errors
            benchmark_result.status = "failed"
            benchmark_result.error_message = str(e)
            
            log_benchmark_end(self.run_id, 0, False)
            self.logger.error(f"Benchmark failed: {e}")
            raise
    
    async def _evaluate_model(self, model_info: Dict[str, Any], tasks: List[str]) -> Dict[str, float]:
        """Simulate model evaluation."""
        import random
        import asyncio
        
        # Simulate evaluation time
        await asyncio.sleep(1)
        
        # Generate mock results based on model type
        is_finance_tuned = "fin" in model_info["name"].lower()
        base_quality = 0.8 if is_finance_tuned else 0.7
        
        # Add some randomness
        quality_variance = random.uniform(-0.1, 0.1)
        efficiency_variance = random.uniform(-0.1, 0.1)
        
        results = {
            "final_score": min(1.0, max(0.0, base_quality + quality_variance * 0.5)),
            "quality_score": min(1.0, max(0.0, base_quality + quality_variance)),
            "efficiency_score": min(1.0, max(0.0, 0.7 + efficiency_variance)),
            "low_score": min(1.0, max(0.0, base_quality + 0.1 + quality_variance)),
            "medium_score": min(1.0, max(0.0, base_quality + quality_variance)),
            "high_score": min(1.0, max(0.0, base_quality - 0.1 + quality_variance)),
            "avg_latency_ms": random.uniform(1500, 4000),
            "avg_cost_per_task": random.uniform(0.001, 0.01),
            "completion_rate": random.uniform(0.85, 1.0),
        }
        
        self.logger.info(f"Model {model_info['display_name']} - Final Score: {results['final_score']:.3f}")
        return results


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run LLM Finance Leaderboard benchmark")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mistral-7b", "finma-7b"],
        help="Models to evaluate"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["eps_extraction"],
        help="Tasks to run"
    )
    parser.add_argument(
        "--name",
        default=f"Benchmark Run {datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Benchmark run name"
    )
    parser.add_argument(
        "--description",
        default="Automated benchmark run",
        help="Benchmark description"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--use-real-runner",
        action="store_true",
        help="Use real benchmark runner instead of mock"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level=log_level)
    
    # Initialize database
    init_database()
    
    print("🚀 LLM Finance Leaderboard - Benchmark Runner")
    print(f"📊 Models: {', '.join(args.models)}")
    print(f"📋 Tasks: {', '.join(args.tasks)}")
    print(f"📝 Run Name: {args.name}")
    print()
    
    try:
        if args.use_real_runner:
            # Use the new real benchmark runner
            from src.evaluation.runners.benchmark_runner import create_benchmark_runner
            
            runner = create_benchmark_runner()
            result = await runner.run_benchmark(
                models=args.models,
                tasks=args.tasks,
                run_name=args.name,
                description=args.description
            )
        else:
            # Use mock runner for compatibility
            runner = MockBenchmarkRunner()
            result = await runner.run_benchmark(
                models=args.models,
                tasks=args.tasks,
                run_name=args.name,
                description=args.description
            )
        
        print("✅ Benchmark completed successfully!")
        print(f"🆔 Run ID: {result.run_id}")
        print(f"⏱️  Duration: {result.total_duration_minutes:.2f} minutes")
        print(f"📈 Models Evaluated: {len(result.models_evaluated)}")
        print(f"📋 Tasks Completed: {len(result.tasks_included)}")
        print()
        print("🎯 View results in the Streamlit dashboard:")
        print("   streamlit run streamlit_app/main.py")
        
        return 0
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))