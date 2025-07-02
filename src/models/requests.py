"""
API request models for Lift-os-LLM microservice.

Defines Pydantic models for all API request payloads with validation.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator, HttpUrl

from .entities import (
    AnalysisType, OptimizationType, Priority, ContentInput,
    ModelProvider
)


# Analysis Requests
class AnalysisRequest(BaseModel):
    """Request for content analysis."""
    content: ContentInput
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)
    analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE
    include_embeddings: bool = True
    include_knowledge_graph: bool = True
    target_audience: Optional[str] = None
    
    @validator('options')
    def validate_options(cls, v):
        """Validate analysis options."""
        allowed_options = {
            'max_tokens', 'temperature', 'model_override', 
            'embedding_model', 'analysis_depth', 'language'
        }
        invalid_options = set(v.keys()) - allowed_options
        if invalid_options:
            raise ValueError(f"Invalid options: {invalid_options}")
        return v


class EmbeddingsRequest(BaseModel):
    """Request for vector embeddings analysis."""
    content: ContentInput
    embedding_model: Optional[str] = None
    include_quality_metrics: bool = True
    similarity_threshold: float = Field(0.8, ge=0.0, le=1.0)
    options: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeGraphRequest(BaseModel):
    """Request for knowledge graph analysis."""
    content: ContentInput
    include_relationships: bool = True
    max_depth: int = Field(3, ge=1, le=5)
    entity_types: Optional[List[str]] = None
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)


# Optimization Requests
class OptimizationRequest(BaseModel):
    """Request for content optimization."""
    original_content: ContentInput
    optimization_targets: List[OptimizationType]
    platform: Optional[str] = None
    include_implementation_guide: bool = True
    target_score_improvement: Optional[float] = Field(None, ge=0, le=100)
    
    @validator('optimization_targets')
    def validate_targets(cls, v):
        """Ensure at least one optimization target."""
        if not v:
            raise ValueError("At least one optimization target must be specified")
        return v


class SchemaOptimizationRequest(BaseModel):
    """Request for schema.org markup optimization."""
    content: ContentInput
    schema_type: Optional[str] = None
    include_validation: bool = True
    structured_data_format: str = Field("json-ld", regex="^(json-ld|microdata|rdfa)$")


class MetaTagsOptimizationRequest(BaseModel):
    """Request for meta tags optimization."""
    content: ContentInput
    include_open_graph: bool = True
    include_twitter_cards: bool = True
    target_platforms: List[str] = Field(default_factory=lambda: ["google", "facebook", "twitter"])


# Model Management Requests
class ModelEvaluationRequest(BaseModel):
    """Request for model evaluation."""
    models: List[str] = Field(..., min_items=1)
    evaluation_tasks: List[str] = Field(..., min_items=1)
    test_content: List[ContentInput] = Field(..., min_items=1)
    comparison_metrics: List[str] = Field(default_factory=lambda: ["accuracy", "speed", "cost"])


class ModelComparisonRequest(BaseModel):
    """Request for model comparison."""
    base_model: str
    comparison_models: List[str] = Field(..., min_items=1)
    test_scenarios: List[Dict[str, Any]] = Field(..., min_items=1)
    include_cost_analysis: bool = True


class FineTuningRequest(BaseModel):
    """Request for model fine-tuning."""
    base_model: str
    training_data: List[Dict[str, str]] = Field(..., min_items=10)
    validation_data: Optional[List[Dict[str, str]]] = None
    training_config: Dict[str, Any] = Field(default_factory=dict)
    job_name: Optional[str] = None
    
    @validator('training_data')
    def validate_training_data(cls, v):
        """Validate training data format."""
        for item in v:
            if 'input' not in item or 'output' not in item:
                raise ValueError("Training data must contain 'input' and 'output' fields")
        return v


# Batch Processing Requests
class BatchAnalysisRequest(BaseModel):
    """Request for batch content analysis."""
    urls: List[HttpUrl] = Field(..., min_items=1, max_items=100)
    analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE
    priority: Priority = Priority.NORMAL
    webhook_url: Optional[HttpUrl] = None
    batch_options: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('urls')
    def validate_urls(cls, v):
        """Validate URL list."""
        if len(set(v)) != len(v):
            raise ValueError("Duplicate URLs are not allowed")
        return v


class BatchOptimizationRequest(BaseModel):
    """Request for batch content optimization."""
    content_items: List[ContentInput] = Field(..., min_items=1, max_items=50)
    optimization_targets: List[OptimizationType]
    priority: Priority = Priority.NORMAL
    webhook_url: Optional[HttpUrl] = None
    batch_options: Dict[str, Any] = Field(default_factory=dict)


class BatchJobStatusRequest(BaseModel):
    """Request for batch job status."""
    job_id: str
    include_results: bool = False
    include_errors: bool = False


# AI-SERP Integration Requests
class SERPTestRequest(BaseModel):
    """Request for AI search engine testing."""
    content: ContentInput
    search_engines: List[str] = Field(default_factory=lambda: ["perplexity", "sge", "amazon_ai"])
    test_queries: List[str] = Field(..., min_items=1)
    include_ranking_prediction: bool = True


class SERPRankingRequest(BaseModel):
    """Request for AI search ranking check."""
    urls: List[HttpUrl] = Field(..., min_items=1, max_items=10)
    queries: List[str] = Field(..., min_items=1)
    search_engines: List[str] = Field(default_factory=lambda: ["perplexity", "sge"])
    track_changes: bool = True


# Similarity Search Requests
class SimilaritySearchRequest(BaseModel):
    """Request for content similarity search."""
    query_content: ContentInput
    search_corpus: Optional[List[ContentInput]] = None
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0)
    max_results: int = Field(10, ge=1, le=100)
    include_scores: bool = True


class HybridSearchRequest(BaseModel):
    """Request for hybrid similarity search."""
    query_content: ContentInput
    search_types: List[str] = Field(default_factory=lambda: ["semantic", "keyword", "structural"])
    weights: Dict[str, float] = Field(default_factory=lambda: {"semantic": 0.6, "keyword": 0.3, "structural": 0.1})
    max_results: int = Field(10, ge=1, le=100)
    
    @validator('weights')
    def validate_weights(cls, v):
        """Validate search weights sum to 1.0."""
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError("Search weights must sum to 1.0")
        return v


# Authentication Requests
class LoginRequest(BaseModel):
    """User login request."""
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    password: str = Field(..., min_length=8)


class RegisterRequest(BaseModel):
    """User registration request."""
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password strength."""
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class APIKeyCreateRequest(BaseModel):
    """API key creation request."""
    name: str = Field(..., min_length=1, max_length=100)
    permissions: List[str] = Field(default_factory=list)
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)


class PasswordResetRequest(BaseModel):
    """Password reset request."""
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')


class PasswordChangeRequest(BaseModel):
    """Password change request."""
    current_password: str
    new_password: str = Field(..., min_length=8)
    
    @validator('new_password')
    def validate_new_password(cls, v):
        """Validate new password strength."""
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


# Configuration Requests
class ConfigUpdateRequest(BaseModel):
    """Configuration update request."""
    settings: Dict[str, Any]
    
    @validator('settings')
    def validate_settings(cls, v):
        """Validate configuration settings."""
        allowed_settings = {
            'rate_limit_requests', 'rate_limit_window', 'max_concurrent_analyses',
            'cache_ttl_seconds', 'default_llm_model', 'default_embedding_model'
        }
        invalid_settings = set(v.keys()) - allowed_settings
        if invalid_settings:
            raise ValueError(f"Invalid settings: {invalid_settings}")
        return v


# Health Check Requests
class HealthCheckRequest(BaseModel):
    """Health check request."""
    include_detailed_status: bool = False
    check_external_services: bool = True


# Webhook Requests
class WebhookRequest(BaseModel):
    """Webhook configuration request."""
    url: HttpUrl
    events: List[str] = Field(..., min_items=1)
    secret: Optional[str] = None
    active: bool = True
    
    @validator('events')
    def validate_events(cls, v):
        """Validate webhook events."""
        allowed_events = {
            'analysis.completed', 'optimization.completed', 'batch.completed',
            'batch.failed', 'model.evaluation.completed'
        }
        invalid_events = set(v) - allowed_events
        if invalid_events:
            raise ValueError(f"Invalid events: {invalid_events}")
        return v


# Training and Evaluation Requests
class TrainingJobRequest(BaseModel):
    """Request model for submitting a training job."""
    model_name: str = Field(..., description="Base model to fine-tune")
    industry: str = Field(..., description="Target industry (finance, education, retail, healthcare, multi-industry)")
    training_config: Dict[str, Any] = Field(default_factory=dict, description="Training configuration parameters")
    dataset_path: Optional[str] = Field(None, description="Optional custom dataset path")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "gpt-3.5-turbo",
                "industry": "finance",
                "training_config": {
                    "learning_rate": 0.0001,
                    "epochs": 20,
                    "batch_size": 4,
                    "max_seq_length": 512
                },
                "dataset_path": "data/training/custom_finance_dataset.jsonl"
            }
        }


class ModelEvaluationRequest(BaseModel):
    """Request model for evaluating a model."""
    model_path: str = Field(..., description="Path to the model to evaluate")
    industry: str = Field(..., description="Industry context for evaluation")
    evaluation_dataset: Optional[str] = Field(None, description="Optional custom evaluation dataset")
    metrics: List[str] = Field(
        default=["accuracy", "latency", "cost", "throughput"],
        description="Metrics to evaluate"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_path": "models/fine-tuned/train-123",
                "industry": "finance",
                "evaluation_dataset": "data/evaluation/finance_test.jsonl",
                "metrics": ["accuracy", "precision", "recall", "latency", "cost"]
            }
        }


class ModelComparisonRequest(BaseModel):
    """Request model for comparing multiple models."""
    model_paths: List[str] = Field(..., description="List of model paths to compare")
    industry: str = Field(..., description="Industry context for comparison")
    metrics: List[str] = Field(
        default=["accuracy", "latency", "cost"],
        description="Metrics to compare"
    )
    test_dataset: Optional[str] = Field(None, description="Optional test dataset for comparison")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_paths": [
                    "models/gpt-3.5-turbo-baseline",
                    "models/fine-tuned/train-123",
                    "models/fine-tuned/train-124"
                ],
                "industry": "finance",
                "metrics": ["accuracy", "latency", "cost", "throughput"],
                "test_dataset": "data/evaluation/finance_benchmark.jsonl"
            }
        }