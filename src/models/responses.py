"""
API response models for Lift-os-LLM microservice.

Defines Pydantic models for all API response payloads.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field

from .entities import (
    ContentAnalysisResult, ContentOptimizationResult, ModelEvaluationResult,
    BatchJob, HealthStatus, ServiceMetrics, ModelInfo, JobStatus
)


# Base Response Models
class BaseResponse(BaseModel):
    """Base response model with common fields."""
    success: bool = True
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseResponse):
    """Error response model."""
    success: bool = False
    error: Dict[str, Any]
    
    @classmethod
    def create(
        cls,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        """Create error response."""
        return cls(
            error={
                "code": error_code,
                "message": message,
                "details": details or {},
                "timestamp": datetime.utcnow().isoformat()
            },
            request_id=request_id
        )


class SuccessResponse(BaseResponse):
    """Generic success response."""
    message: str
    data: Optional[Dict[str, Any]] = None


# Analysis Response Models
class AnalysisResponse(BaseResponse):
    """Content analysis response."""
    data: ContentAnalysisResult
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingsResponse(BaseResponse):
    """Vector embeddings response."""
    data: Dict[str, Any]
    quality_metrics: Dict[str, float]
    recommendations: List[str] = Field(default_factory=list)


class KnowledgeGraphResponse(BaseResponse):
    """Knowledge graph response."""
    data: Dict[str, Any]
    metrics: Dict[str, float]
    insights: List[str] = Field(default_factory=list)


# Optimization Response Models
class OptimizationResponse(BaseResponse):
    """Content optimization response."""
    data: ContentOptimizationResult
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SchemaOptimizationResponse(BaseResponse):
    """Schema optimization response."""
    data: Dict[str, Any]
    validation_results: Dict[str, Any]
    implementation_guide: Dict[str, Any]


class MetaTagsOptimizationResponse(BaseResponse):
    """Meta tags optimization response."""
    data: Dict[str, str]
    improvements: Dict[str, Any]
    platform_specific: Dict[str, Dict[str, str]] = Field(default_factory=dict)


# Model Management Response Models
class ModelListResponse(BaseResponse):
    """Available models list response."""
    data: List[ModelInfo]
    total_count: int
    available_providers: List[str]


class ModelEvaluationResponse(BaseResponse):
    """Model evaluation response."""
    data: List[ModelEvaluationResult]
    comparison_summary: Dict[str, Any]
    recommendations: List[str] = Field(default_factory=list)


class ModelComparisonResponse(BaseResponse):
    """Model comparison response."""
    data: Dict[str, Any]
    winner: str
    performance_breakdown: Dict[str, Dict[str, float]]
    cost_analysis: Dict[str, float]


class FineTuningResponse(BaseResponse):
    """Fine-tuning job response."""
    data: Dict[str, Any]
    job_id: str
    estimated_completion: Optional[datetime] = None
    cost_estimate: Optional[float] = None


# Batch Processing Response Models
class BatchJobResponse(BaseResponse):
    """Batch job submission response."""
    data: BatchJob
    estimated_completion: Optional[datetime] = None


class BatchStatusResponse(BaseResponse):
    """Batch job status response."""
    data: BatchJob
    progress_details: Dict[str, Any] = Field(default_factory=dict)


class BatchResultsResponse(BaseResponse):
    """Batch job results response."""
    data: Dict[str, Any]
    summary: Dict[str, Any]
    failed_items: List[Dict[str, Any]] = Field(default_factory=list)


# AI-SERP Integration Response Models
class SERPTestResponse(BaseResponse):
    """AI search engine test response."""
    data: Dict[str, Any]
    visibility_predictions: Dict[str, Dict[str, float]]
    recommendations: List[str] = Field(default_factory=list)


class SERPRankingResponse(BaseResponse):
    """AI search ranking response."""
    data: Dict[str, Any]
    ranking_changes: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    trend_analysis: Dict[str, Any] = Field(default_factory=dict)


# Similarity Search Response Models
class SimilaritySearchResponse(BaseResponse):
    """Similarity search response."""
    data: List[Dict[str, Any]]
    search_metadata: Dict[str, Any]
    total_results: int


class HybridSearchResponse(BaseResponse):
    """Hybrid search response."""
    data: List[Dict[str, Any]]
    search_breakdown: Dict[str, List[Dict[str, Any]]]
    combined_scores: Dict[str, float]


# Authentication Response Models
class LoginResponse(BaseResponse):
    """Login response."""
    data: Dict[str, Any]
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class RegisterResponse(BaseResponse):
    """Registration response."""
    data: Dict[str, Any]
    message: str = "User registered successfully"


class APIKeyResponse(BaseResponse):
    """API key creation response."""
    data: Dict[str, Any]
    api_key: str
    warning: str = "Store this key securely. It will not be shown again."


class UserProfileResponse(BaseResponse):
    """User profile response."""
    data: Dict[str, Any]
    usage_statistics: Dict[str, Any]
    subscription_info: Dict[str, Any]


# Health and Monitoring Response Models
class HealthResponse(BaseResponse):
    """Health check response."""
    data: HealthStatus
    uptime_seconds: float
    version: str = "1.0.0"


class MetricsResponse(BaseResponse):
    """Service metrics response."""
    data: ServiceMetrics
    time_range: Dict[str, datetime]


class StatusResponse(BaseResponse):
    """Service status response."""
    data: Dict[str, Any]
    active_connections: int
    queue_status: Dict[str, int]


# Configuration Response Models
class ConfigResponse(BaseResponse):
    """Configuration response."""
    data: Dict[str, Any]
    last_updated: datetime
    updated_by: Optional[str] = None


# Webhook Response Models
class WebhookResponse(BaseResponse):
    """Webhook configuration response."""
    data: Dict[str, Any]
    webhook_id: str
    test_url: Optional[str] = None


class WebhookTestResponse(BaseResponse):
    """Webhook test response."""
    data: Dict[str, Any]
    delivery_status: str
    response_time_ms: float


# Pagination Response Models
class PaginatedResponse(BaseResponse):
    """Paginated response model."""
    data: List[Any]
    pagination: Dict[str, Any]
    
    @classmethod
    def create(
        cls,
        items: List[Any],
        page: int,
        page_size: int,
        total_items: int,
        **kwargs
    ):
        """Create paginated response."""
        total_pages = (total_items + page_size - 1) // page_size
        
        return cls(
            data=items,
            pagination={
                "page": page,
                "page_size": page_size,
                "total_items": total_items,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_previous": page > 1,
                "next_page": page + 1 if page < total_pages else None,
                "previous_page": page - 1 if page > 1 else None
            },
            **kwargs
        )


# Bulk Operation Response Models
class BulkOperationResponse(BaseResponse):
    """Bulk operation response."""
    data: Dict[str, Any]
    summary: Dict[str, int]
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    
    @classmethod
    def create(
        cls,
        successful_items: List[Any],
        failed_items: List[Dict[str, Any]],
        operation_type: str,
        **kwargs
    ):
        """Create bulk operation response."""
        return cls(
            data={
                "successful_items": successful_items,
                "failed_items": failed_items,
                "operation_type": operation_type
            },
            summary={
                "total_items": len(successful_items) + len(failed_items),
                "successful_count": len(successful_items),
                "failed_count": len(failed_items),
                "success_rate": len(successful_items) / (len(successful_items) + len(failed_items)) if (len(successful_items) + len(failed_items)) > 0 else 0
            },
            errors=failed_items,
            **kwargs
        )


# Export Response Models
class ExportResponse(BaseResponse):
    """Data export response."""
    data: Dict[str, Any]
    download_url: str
    expires_at: datetime
    file_format: str
    file_size_bytes: int


# Analytics Response Models
class AnalyticsResponse(BaseResponse):
    """Analytics data response."""
    data: Dict[str, Any]
    time_period: Dict[str, datetime]
    aggregation_level: str
    insights: List[str] = Field(default_factory=list)


class TrendAnalysisResponse(BaseResponse):
    """Trend analysis response."""
    data: Dict[str, Any]
    trends: List[Dict[str, Any]]
    predictions: Dict[str, Any] = Field(default_factory=dict)
    confidence_intervals: Dict[str, Dict[str, float]] = Field(default_factory=dict)


# Cache Response Models
class CacheStatsResponse(BaseResponse):
    """Cache statistics response."""
    data: Dict[str, Any]
    hit_rate: float
    total_keys: int
    memory_usage_mb: float


# Rate Limiting Response Models
class RateLimitResponse(BaseResponse):
    """Rate limit information response."""
    data: Dict[str, Any]
    current_usage: int
    limit: int
    reset_time: datetime
    retry_after_seconds: Optional[int] = None


# Validation Response Models
class ValidationResponse(BaseResponse):
    """Content validation response."""
    data: Dict[str, Any]
    validation_results: Dict[str, Any]
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


# Search Response Models
class SearchResponse(BaseResponse):
    """Search results response."""
    data: List[Dict[str, Any]]
    query: str
    total_results: int
    search_time_ms: float
    facets: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)


# Notification Response Models
class NotificationResponse(BaseResponse):
    """Notification response."""
    data: Dict[str, Any]
    notification_id: str
    delivery_status: str
    channels: List[str]


# Training and Evaluation Response Models
class TrainingJobResponse(BaseResponse):
    """Training job response."""
    data: Dict[str, Any]
    job_id: str
    status: str
    model_name: str
    industry: str


class TrainingJobListResponse(BaseResponse):
    """Training job list response."""
    data: Dict[str, Any]
    total_count: int
    limit: int
    offset: int
    has_more: bool


class ModelEvaluationResponse(BaseResponse):
    """Model evaluation response."""
    data: Dict[str, Any]
    model_path: str
    industry: str
    evaluation_date: str
    metrics: Dict[str, float]
    benchmark_comparison: Dict[str, Any]


class ModelComparisonResponse(BaseResponse):
    """Model comparison response."""
    data: Dict[str, Any]
    comparison_id: str
    industry: str
    models: List[Dict[str, Any]]
    summary: Dict[str, Any]