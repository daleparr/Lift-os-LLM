"""
Base task interface for LLM Finance Leaderboard.

Defines the common interface for all evaluation tasks.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import uuid
from loguru import logger

from ..data.schemas.data_models import TaskComplexity, TaskResult, AgentResponse
from ..config.settings import settings


class BaseTask(ABC):
    """Base class for all evaluation tasks."""
    
    def __init__(
        self,
        task_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        complexity: TaskComplexity = TaskComplexity.LOW,
        timeout_minutes: int = 10,
        max_retries: int = 2
    ):
        """Initialize base task."""
        self.task_id = task_id or str(uuid.uuid4())
        self.name = name or self.__class__.__name__
        self.description = description or "No description provided"
        self.complexity = complexity
        self.timeout_minutes = timeout_minutes
        self.max_retries = max_retries
        
        logger.info(f"Initialized task: {self.name} ({self.complexity})")
    
    @abstractmethod
    def generate_prompt(self, context: Dict[str, Any]) -> str:
        """
        Generate the prompt for this task.
        
        Args:
            context: Context information including documents, parameters, etc.
            
        Returns:
            Formatted prompt string
        """
        pass
    
    @abstractmethod
    def validate_response(self, response: str, expected_answer: str) -> Dict[str, float]:
        """
        Validate and score a model response.
        
        Args:
            response: Model's response
            expected_answer: Expected/ground truth answer
            
        Returns:
            Dictionary of metric scores
        """
        pass
    
    @abstractmethod
    def get_expected_answer(self, context: Dict[str, Any]) -> str:
        """
        Get the expected answer for the given context.
        
        Args:
            context: Context information
            
        Returns:
            Expected answer string
        """
        pass
    
    def execute(
        self,
        agent_pipeline,
        context: Dict[str, Any],
        evaluation_seed: int = 42
    ) -> TaskResult:
        """
        Execute the task using the provided agent pipeline.
        
        Args:
            agent_pipeline: Agent pipeline to use for execution
            context: Task context and data
            evaluation_seed: Random seed for reproducibility
            
        Returns:
            TaskResult with execution details and scores
        """
        start_time = datetime.utcnow()
        
        try:
            # Generate prompt
            prompt = self.generate_prompt(context)
            
            # Get expected answer
            expected_answer = self.get_expected_answer(context)
            
            # Execute with agent pipeline
            agent_state = agent_pipeline.run(
                query=prompt,
                task_id=self.task_id,
                task_complexity=self.complexity.value,
                config={"configurable": {"thread_id": f"{self.task_id}_{evaluation_seed}"}}
            )
            
            # Extract response and metrics
            model_response = agent_state.get("final_response", "")
            agent_responses = []  # Would be populated from agent_state in real implementation
            
            # Calculate execution metrics
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Validate response
            scores = self.validate_response(model_response, expected_answer)
            
            # Create task result
            task_result = TaskResult(
                task_id=self.task_id,
                task_name=self.name,
                complexity=self.complexity,
                model_name=getattr(agent_pipeline, 'model_name', 'unknown'),
                prompt=prompt,
                expected_answer=expected_answer,
                model_response=model_response,
                agent_responses=agent_responses,
                
                # Quality metrics
                exact_match_score=scores.get("exact_match", None),
                f1_score=scores.get("f1_score", None),
                rouge_1_score=scores.get("rouge_1", None),
                rouge_2_score=scores.get("rouge_2", None),
                rouge_l_score=scores.get("rouge_l", None),
                bleu_score=scores.get("bleu", None),
                fact_score=scores.get("fact_score", None),
                human_rating=scores.get("human_rating", None),
                
                # Efficiency metrics
                total_latency_ms=execution_time,
                total_tokens_used=agent_state.get("execution_metrics", {}).get("total_tokens", 0),
                cost_usd=agent_state.get("execution_metrics", {}).get("cost_usd", 0.0),
                
                # Success metrics
                task_completed=bool(model_response and not agent_state.get("error_message")),
                tool_call_success_rate=agent_state.get("execution_metrics", {}).get("tool_success_rate", 1.0),
                hallucination_detected=scores.get("hallucination_detected", False),
                
                # Metadata
                evaluation_seed=evaluation_seed,
                created_at=datetime.utcnow()
            )
            
            logger.info(f"Task {self.name} completed - Score: {scores.get('f1_score', 0):.3f}")
            return task_result
            
        except Exception as e:
            logger.error(f"Task {self.name} failed: {e}")
            
            # Create failed task result
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return TaskResult(
                task_id=self.task_id,
                task_name=self.name,
                complexity=self.complexity,
                model_name=getattr(agent_pipeline, 'model_name', 'unknown'),
                prompt=self.generate_prompt(context) if context else "Error generating prompt",
                expected_answer="Error getting expected answer",
                model_response=f"Task execution failed: {str(e)}",
                agent_responses=[],
                
                # Set all scores to 0 for failed task
                exact_match_score=0.0,
                f1_score=0.0,
                total_latency_ms=execution_time,
                total_tokens_used=0,
                cost_usd=0.0,
                task_completed=False,
                evaluation_seed=evaluation_seed,
                created_at=datetime.utcnow()
            )
    
    def get_task_config(self) -> Dict[str, Any]:
        """Get task configuration."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "complexity": self.complexity.value,
            "timeout_minutes": self.timeout_minutes,
            "max_retries": self.max_retries
        }
    
    def prepare_context(self, documents: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Prepare context for task execution.
        
        Args:
            documents: Retrieved documents
            **kwargs: Additional context parameters
            
        Returns:
            Prepared context dictionary
        """
        return {
            "documents": documents,
            "task_config": self.get_task_config(),
            **kwargs
        }


class FinancialExtractionTask(BaseTask):
    """Base class for financial data extraction tasks."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def extract_financial_value(self, text: str, patterns: List[str]) -> Optional[str]:
        """Extract financial value using regex patterns."""
        import re
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1) if match.groups() else match.group(0)
        
        return None
    
    def normalize_financial_value(self, value: str) -> Optional[float]:
        """Normalize financial value to float."""
        if not value:
            return None
        
        try:
            # Remove common formatting
            cleaned = re.sub(r'[,$\s]', '', value)
            
            # Handle millions/billions
            if 'M' in cleaned.upper() or 'MILLION' in cleaned.upper():
                cleaned = re.sub(r'[MB]|MILLION', '', cleaned, flags=re.IGNORECASE)
                return float(cleaned) * 1_000_000
            elif 'B' in cleaned.upper() or 'BILLION' in cleaned.upper():
                cleaned = re.sub(r'[B]|BILLION', '', cleaned, flags=re.IGNORECASE)
                return float(cleaned) * 1_000_000_000
            
            return float(cleaned)
            
        except (ValueError, TypeError):
            return None


class FinancialAnalysisTask(BaseTask):
    """Base class for financial analysis tasks."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def extract_key_insights(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Extract key insights from documents."""
        insights = []
        
        for doc in documents:
            content = doc.get("content", "")
            
            # Look for key phrases that indicate insights
            insight_patterns = [
                r"(?i)(?:we believe|we expect|outlook|guidance|forecast).*?[.!?]",
                r"(?i)(?:growth|increase|decrease|improvement|decline).*?[.!?]",
                r"(?i)(?:risk|challenge|opportunity|trend).*?[.!?]"
            ]
            
            for pattern in insight_patterns:
                matches = re.findall(pattern, content)
                insights.extend(matches[:2])  # Limit per document
        
        return insights[:5]  # Limit total insights


def create_task_from_config(task_config: Dict[str, Any]) -> BaseTask:
    """Create a task instance from configuration."""
    task_type = task_config.get("type", "base")
    
    # This would be expanded with actual task implementations
    if task_type == "eps_extraction":
        from .low_complexity.eps_extraction import EPSExtractionTask
        return EPSExtractionTask(**task_config)
    else:
        raise ValueError(f"Unknown task type: {task_type}")