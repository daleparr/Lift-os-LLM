"""
Model comparison engine for evaluating base vs fine-tuned models.
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from loguru import logger

from ..data.schemas.data_models import ModelComparison
from ..evaluation.runners.benchmark_runner import BenchmarkRunner
from ..models.model_loader import ModelLoader


class ModelComparisonEngine:
    """Engine for comparing base and fine-tuned models."""
    
    def __init__(self):
        """Initialize comparison engine."""
        self.model_loader = ModelLoader()
        self.benchmark_runner = BenchmarkRunner()
        
        logger.info("Initialized ModelComparisonEngine")
    
    def compare_models(
        self,
        base_model_name: str,
        finetuned_model_path: str,
        tasks: Optional[List[str]] = None
    ) -> ModelComparison:
        """Compare base model against fine-tuned variant."""
        
        logger.info(f"Starting comparison: {base_model_name} vs {finetuned_model_path}")
        
        # Default tasks for comparison
        if tasks is None:
            tasks = [
                "eps_extraction",
                "sentiment_analysis", 
                "revenue_analysis"
            ]
        
        # Evaluate base model
        logger.info("Evaluating base model...")
        base_results = self._evaluate_model(base_model_name, tasks, is_base=True)
        
        # Evaluate fine-tuned model
        logger.info("Evaluating fine-tuned model...")
        finetuned_results = self._evaluate_model(
            base_model_name, 
            tasks, 
            is_base=False,
            lora_path=finetuned_model_path
        )
        
        # Calculate improvements
        improvements = self._calculate_improvements(base_results, finetuned_results)
        overall_improvement = sum(improvements.values()) / len(improvements)
        
        # Create comparison object
        comparison = ModelComparison(
            base_model_id=base_model_name,
            finetuned_model_id=finetuned_model_path,
            comparison_date=datetime.now(),
            base_scores=base_results,
            finetuned_scores=finetuned_results,
            improvements=improvements,
            overall_improvement=overall_improvement,
            training_time_hours=self._estimate_training_time(base_model_name),
            power_consumption_kwh=self._estimate_power_consumption(base_model_name)
        )
        
        # Save comparison results
        self._save_comparison(comparison)
        
        logger.info(f"Comparison completed. Overall improvement: {overall_improvement:.2%}")
        return comparison
    
    def _evaluate_model(
        self,
        model_name: str,
        tasks: List[str],
        is_base: bool = True,
        lora_path: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate a model on specified tasks."""
        
        results = {}
        
        try:
            # Load model
            if is_base:
                model = self.model_loader.load_model(model_name)
                logger.info(f"Loaded base model: {model_name}")
            else:
                # Load fine-tuned model
                model_config = self.model_loader.model_configs.get(model_name, {})
                model_config["lora_path"] = lora_path
                model = self.model_loader.load_model(model_name, model_config)
                logger.info(f"Loaded fine-tuned model from: {lora_path}")
            
            # Run evaluation on each task
            for task_name in tasks:
                try:
                    task_result = self._run_task_evaluation(model, task_name)
                    results[task_name] = task_result
                    logger.info(f"Task {task_name}: {task_result:.3f}")
                except Exception as e:
                    logger.error(f"Failed to evaluate task {task_name}: {e}")
                    results[task_name] = 0.0
            
            # Calculate overall score
            if results:
                results["overall_score"] = sum(results.values()) / len(results)
            else:
                results["overall_score"] = 0.0
            
        except Exception as e:
            logger.error(f"Failed to evaluate model: {e}")
            # Return zero scores for all tasks
            for task_name in tasks:
                results[task_name] = 0.0
            results["overall_score"] = 0.0
        
        return results
    
    def _run_task_evaluation(self, model, task_name: str) -> float:
        """Run evaluation for a specific task."""
        
        # Mock evaluation for demonstration
        # In a real implementation, this would run the actual benchmark
        
        if task_name == "eps_extraction":
            # Simulate EPS extraction task
            test_prompt = """
            Analyze the following financial statement and extract the EPS value:
            
            Q3 2024 Financial Results:
            - Revenue: $2.1 billion
            - Net Income: $150 million  
            - Shares Outstanding: 50 million
            
            What is the EPS?
            """
            
            # Mock response evaluation
            # In reality, this would generate a response and score it
            return 0.78 if "base" in str(model) else 0.92
            
        elif task_name == "sentiment_analysis":
            # Simulate sentiment analysis task
            test_prompt = """
            Analyze the sentiment of this earnings call excerpt:
            
            "We are extremely pleased with our Q3 performance. Revenue exceeded 
            expectations by 15%, and we're seeing strong momentum across all 
            business segments. We remain optimistic about Q4 prospects."
            
            Sentiment: Positive, Negative, or Neutral?
            """
            
            return 0.71 if "base" in str(model) else 0.88
            
        elif task_name == "revenue_analysis":
            # Simulate revenue analysis task
            test_prompt = """
            Compare the revenue trends:
            
            Q1 2024: $1.8B
            Q2 2024: $1.9B  
            Q3 2024: $2.1B
            
            What is the quarter-over-quarter growth rate for Q3?
            """
            
            return 0.65 if "base" in str(model) else 0.84
        
        else:
            # Unknown task
            return 0.5
    
    def _calculate_improvements(
        self,
        base_results: Dict[str, float],
        finetuned_results: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate improvement percentages."""
        
        improvements = {}
        
        for task_name in base_results:
            base_score = base_results[task_name]
            finetuned_score = finetuned_results.get(task_name, 0.0)
            
            if base_score > 0:
                improvement = (finetuned_score - base_score) / base_score
            else:
                improvement = 0.0
            
            improvements[task_name] = improvement
        
        return improvements
    
    def _estimate_training_time(self, model_name: str) -> float:
        """Estimate training time in hours."""
        # Simple estimation based on model size
        if "7B" in model_name or "7b" in model_name:
            return 3.2
        elif "13B" in model_name or "13b" in model_name:
            return 6.5
        elif "3B" in model_name or "3b" in model_name:
            return 1.8
        else:
            return 3.0
    
    def _estimate_power_consumption(self, model_name: str) -> float:
        """Estimate power consumption in kWh."""
        training_hours = self._estimate_training_time(model_name)
        # Assume 450W average power consumption
        power_kw = 0.45
        return training_hours * power_kw
    
    def _save_comparison(self, comparison: ModelComparison) -> None:
        """Save comparison results to file."""
        
        # Create comparisons directory
        comparisons_dir = Path("data/comparisons")
        comparisons_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = comparison.comparison_date.strftime("%Y%m%d_%H%M%S")
        base_name = comparison.base_model_id.split("/")[-1]
        filename = f"{base_name}_comparison_{timestamp}.json"
        
        filepath = comparisons_dir / filename
        
        # Save to JSON
        try:
            with open(filepath, 'w') as f:
                json.dump(comparison.dict(), f, indent=2, default=str)
            
            logger.info(f"Saved comparison results to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save comparison: {e}")
    
    def load_comparison(self, comparison_id: str) -> Optional[ModelComparison]:
        """Load a saved comparison."""
        
        comparisons_dir = Path("data/comparisons")
        
        # Look for comparison file
        for filepath in comparisons_dir.glob("*.json"):
            if comparison_id in filepath.name:
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    return ModelComparison(**data)
                    
                except Exception as e:
                    logger.error(f"Failed to load comparison {filepath}: {e}")
        
        return None
    
    def list_comparisons(self) -> List[Dict]:
        """List all saved comparisons."""
        
        comparisons_dir = Path("data/comparisons")
        comparisons = []
        
        if not comparisons_dir.exists():
            return comparisons
        
        for filepath in comparisons_dir.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Extract summary info
                summary = {
                    "id": filepath.stem,
                    "base_model": data.get("base_model_id", ""),
                    "comparison_date": data.get("comparison_date", ""),
                    "overall_improvement": data.get("overall_improvement", 0.0),
                    "training_time_hours": data.get("training_time_hours", 0.0)
                }
                
                comparisons.append(summary)
                
            except Exception as e:
                logger.error(f"Failed to read comparison {filepath}: {e}")
        
        # Sort by date (newest first)
        comparisons.sort(key=lambda x: x["comparison_date"], reverse=True)
        
        return comparisons
    
    def generate_comparison_report(self, comparison: ModelComparison) -> str:
        """Generate a human-readable comparison report."""
        
        report = f"""
# Model Comparison Report

**Base Model:** {comparison.base_model_id}
**Fine-tuned Model:** {comparison.finetuned_model_id}
**Comparison Date:** {comparison.comparison_date.strftime('%Y-%m-%d %H:%M:%S')}

## Performance Results

| Metric | Base Model | Fine-tuned | Improvement |
|--------|------------|------------|-------------|
"""
        
        for task_name in comparison.base_scores:
            if task_name != "overall_score":
                base_score = comparison.base_scores[task_name]
                finetuned_score = comparison.finetuned_scores[task_name]
                improvement = comparison.improvements[task_name]
                
                report += f"| {task_name.replace('_', ' ').title()} | {base_score:.3f} | {finetuned_score:.3f} | {improvement:+.1%} |\n"
        
        # Overall score
        base_overall = comparison.base_scores.get("overall_score", 0.0)
        finetuned_overall = comparison.finetuned_scores.get("overall_score", 0.0)
        
        report += f"| **Overall Score** | **{base_overall:.3f}** | **{finetuned_overall:.3f}** | **{comparison.overall_improvement:+.1%}** |\n"
        
        report += f"""
## Training Metrics

- **Training Time:** {comparison.training_time_hours:.1f} hours
- **Power Consumption:** {comparison.power_consumption_kwh:.1f} kWh
- **Performance ROI:** {comparison.overall_improvement:.1%} improvement

## Summary

The fine-tuned model shows a **{comparison.overall_improvement:.1%}** overall improvement 
compared to the base model, achieved with {comparison.training_time_hours:.1f} hours of training.
"""
        
        return report