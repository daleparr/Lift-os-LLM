"""
Data models and schemas for the LLM Finance Leaderboard.

This module defines Pydantic models for all data structures used throughout
the application, ensuring type safety and data validation.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class TaskComplexity(str, Enum):
    """Task complexity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DocumentType(str, Enum):
    """Types of financial documents."""
    SEC_10Q = "sec_10q"
    SEC_10K = "sec_10k"
    EARNINGS_TRANSCRIPT = "earnings_transcript"
    MARKET_DATA = "market_data"
    NEWS_ARTICLE = "news_article"


class ModelProvider(str, Enum):
    """Model providers."""
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class Document(BaseModel):
    """Base document model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content: str
    document_type: DocumentType
    source_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True


class SECFiling(Document):
    """SEC filing document model."""
    cik: str = Field(..., description="Central Index Key")
    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str
    filing_type: str = Field(..., description="10-Q, 10-K, etc.")
    filing_date: datetime
    period_end_date: datetime
    fiscal_year: int
    fiscal_quarter: Optional[int] = None
    accession_number: str = Field(..., description="SEC accession number")
    
    @validator('document_type', pre=True, always=True)
    def set_document_type(cls, v, values):
        filing_type = values.get('filing_type', '').upper()
        if filing_type == '10-Q':
            return DocumentType.SEC_10Q
        elif filing_type == '10-K':
            return DocumentType.SEC_10K
        return v


class EarningsTranscript(Document):
    """Earnings call transcript model."""
    ticker: str
    company_name: str
    call_date: datetime
    fiscal_year: int
    fiscal_quarter: int
    participants: List[str] = Field(default_factory=list)
    transcript_sections: Dict[str, str] = Field(default_factory=dict)
    
    @validator('document_type', pre=True, always=True)
    def set_document_type(cls, v):
        return DocumentType.EARNINGS_TRANSCRIPT


class MarketData(BaseModel):
    """Market data model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ticker: str
    date: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    adjusted_close: Optional[float] = None
    dividend_amount: Optional[float] = None
    split_coefficient: Optional[float] = None
    
    class Config:
        use_enum_values = True


class NewsArticle(Document):
    """News article model."""
    headline: str
    summary: Optional[str] = None
    published_date: datetime
    source: str
    author: Optional[str] = None
    tickers_mentioned: List[str] = Field(default_factory=list)
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)
    
    @validator('document_type', pre=True, always=True)
    def set_document_type(cls, v):
        return DocumentType.NEWS_ARTICLE


class AgentResponse(BaseModel):
    """Response from an agent in the pipeline."""
    agent_name: str
    response_text: str
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    sources_used: List[str] = Field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    execution_time_ms: float
    token_usage: Dict[str, int] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TaskResult(BaseModel):
    """Result of a single task evaluation."""
    task_id: str
    task_name: str
    complexity: TaskComplexity
    model_name: str
    prompt: str
    expected_answer: str
    model_response: str
    agent_responses: List[AgentResponse] = Field(default_factory=list)
    
    # Quality metrics
    exact_match_score: Optional[float] = None
    f1_score: Optional[float] = None
    rouge_1_score: Optional[float] = None
    rouge_2_score: Optional[float] = None
    rouge_l_score: Optional[float] = None
    bleu_score: Optional[float] = None
    fact_score: Optional[float] = None
    human_rating: Optional[float] = Field(None, ge=1.0, le=5.0)
    
    # Efficiency metrics
    total_latency_ms: float
    total_tokens_used: int
    cost_usd: float
    
    # Success metrics
    task_completed: bool
    tool_call_success_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    hallucination_detected: bool = False
    
    # Metadata
    evaluation_seed: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True


class EvaluationMetrics(BaseModel):
    """Aggregated evaluation metrics for a model."""
    model_name: str
    total_tasks: int
    completed_tasks: int
    
    # Quality scores by complexity
    low_complexity_score: float = Field(ge=0.0, le=1.0)
    medium_complexity_score: float = Field(ge=0.0, le=1.0)
    high_complexity_score: float = Field(ge=0.0, le=1.0)
    overall_quality_score: float = Field(ge=0.0, le=1.0)
    
    # Efficiency metrics
    avg_latency_ms: float
    avg_tokens_per_task: float
    avg_cost_per_task: float
    efficiency_score: float = Field(ge=0.0, le=1.0)
    
    # Success metrics
    completion_rate: float = Field(ge=0.0, le=1.0)
    avg_tool_success_rate: float = Field(ge=0.0, le=1.0)
    hallucination_rate: float = Field(ge=0.0, le=1.0)
    
    # Final composite score
    final_score: float = Field(ge=0.0, le=1.0)
    
    # Metadata
    evaluation_date: datetime = Field(default_factory=datetime.utcnow)
    evaluation_seeds: List[int]
    
    class Config:
        use_enum_values = True


class BenchmarkResult(BaseModel):
    """Complete benchmark run result."""
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_name: str
    description: Optional[str] = None
    
    # Configuration
    models_evaluated: List[str]
    tasks_included: List[str]
    evaluation_seeds: List[int]
    
    # Results
    task_results: List[TaskResult] = Field(default_factory=list)
    model_metrics: Dict[str, EvaluationMetrics] = Field(default_factory=dict)
    
    # Timing
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration_minutes: Optional[float] = None
    
    # Status
    status: str = Field(default="running")  # running, completed, failed
    error_message: Optional[str] = None
    
    # Metadata
    created_by: Optional[str] = None
    hardware_info: Dict[str, Any] = Field(default_factory=dict)
    software_versions: Dict[str, str] = Field(default_factory=dict)
    
    @validator('total_duration_minutes', pre=True, always=True)
    def calculate_duration(cls, v, values):
        if v is not None:
            return v
        start_time = values.get('start_time')
        end_time = values.get('end_time')
        if start_time and end_time:
            return (end_time - start_time).total_seconds() / 60
        return None
    
    class Config:
        use_enum_values = True


class ModelConfig(BaseModel):
    """Model configuration."""
    name: str
    display_name: str
    provider: ModelProvider
    parameters: str
    context_length: int
    cost_per_1k_tokens: float
    quantization: Optional[str] = None
    base_model: Optional[str] = None
    lora_path: Optional[str] = None
    
    # Generation parameters
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 2048
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = Field(default_factory=list)
    
    class Config:
        use_enum_values = True


class TrainingConfig(BaseModel):
    """Training configuration for fine-tuning."""
    model_name: str
    dataset_path: str
    output_dir: str
    local_gpu_ids: List[int] = Field(default_factory=lambda: [0])
    gpu_type: str
    max_power_watts: float = 450.0
    max_temp_celsius: float = 85.0
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = Field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    
    # Training parameters
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    max_steps: int = 1000
    
    # Quantization
    use_4bit: bool = True
    use_fp16: bool = True
    
    class Config:
        use_enum_values = True


class TrainingJob(BaseModel):
    """Training job status and metadata."""
    job_id: str
    model_name: str
    dataset_name: str
    status: str  # "pending", "running", "completed", "failed"
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    current_step: int = 0
    total_steps: int = 0
    loss: Optional[float] = None
    gpu_utilization: Optional[float] = None
    power_consumption: Optional[float] = None
    temperature: Optional[float] = None
    error_message: Optional[str] = None
    output_model_path: Optional[str] = None
    
    class Config:
        use_enum_values = True


class ModelComparison(BaseModel):
    """Model comparison results."""
    base_model_id: str
    finetuned_model_id: str
    comparison_date: datetime
    
    # Performance metrics
    base_scores: Dict[str, float]
    finetuned_scores: Dict[str, float]
    improvements: Dict[str, float]
    overall_improvement: float
    
    # Training metadata
    training_time_hours: float
    power_consumption_kwh: float
    training_cost_estimate: Optional[float] = None
    
    class Config:
        use_enum_values = True


class TaskConfig(BaseModel):
    """Task configuration."""
    task_id: str
    name: str
    description: str
    complexity: TaskComplexity
    prompt_template: str
    expected_output_format: str
    scoring_method: str
    weight: float = Field(default=1.0, ge=0.0)
    timeout_minutes: int = Field(default=10, ge=1)
    
    class Config:
        use_enum_values = True