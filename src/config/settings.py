"""
Settings and configuration management for LLM Finance Leaderboard.
"""

import os
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    pinecone_api_key: Optional[str] = Field(None, env="PINECONE_API_KEY")
    pinecone_environment: Optional[str] = Field(None, env="PINECONE_ENVIRONMENT")
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    huggingface_api_token: Optional[str] = Field(None, env="HUGGINGFACE_API_TOKEN")
    
    # Financial Data APIs (Optional - for enhanced evaluation with real market data)
    fred_api_key: Optional[str] = Field(
        None,
        env="FRED_API_KEY",
        description="Optional: Federal Reserve Economic Data API key for accessing 800,000+ economic indicators (GDP, inflation, unemployment, etc.)"
    )
    alpha_vantage_api_key: Optional[str] = Field(
        None,
        env="ALPHA_VANTAGE_API_KEY",
        description="Optional: Alpha Vantage API key for real-time and historical stock market data (prices, earnings, technical indicators)"
    )
    
    # Database Configuration
    database_url: str = Field("sqlite:///data/leaderboard.db", env="DATABASE_URL")
    vector_index_name: str = Field("finance-leaderboard", env="VECTOR_INDEX_NAME")
    
    # Model Configuration
    default_embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2", 
        env="DEFAULT_EMBEDDING_MODEL"
    )
    default_temperature: float = Field(0.1, env="DEFAULT_TEMPERATURE")
    default_max_tokens: int = Field(2048, env="DEFAULT_MAX_TOKENS")
    default_top_p: float = Field(0.9, env="DEFAULT_TOP_P")
    
    # Evaluation Settings
    benchmark_seeds: List[int] = Field(default=[42, 123, 456], env="BENCHMARK_SEEDS")
    evaluation_timeout_minutes: int = Field(30, env="EVALUATION_TIMEOUT_MINUTES")
    max_concurrent_evaluations: int = Field(3, env="MAX_CONCURRENT_EVALUATIONS")
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    # Experiment Tracking (Optional - for ML experiment monitoring)
    wandb_project: Optional[str] = Field(
        None,
        env="WANDB_PROJECT",
        description="Optional: Weights & Biases project name for tracking training experiments and model performance metrics"
    )
    wandb_api_key: Optional[str] = Field(
        None,
        env="WANDB_API_KEY",
        description="Optional: Weights & Biases API key for logging training runs, hyperparameters, and model comparisons"
    )
    
    # Streamlit Configuration
    streamlit_server_port: int = Field(8501, env="STREAMLIT_SERVER_PORT")
    streamlit_server_address: str = Field("0.0.0.0", env="STREAMLIT_SERVER_ADDRESS")
    
    # Development Settings
    debug: bool = Field(False, env="DEBUG")
    testing: bool = Field(False, env="TESTING")
    
    # Data Paths
    data_dir: str = Field("data", env="DATA_DIR")
    raw_data_dir: str = Field("data/raw", env="RAW_DATA_DIR")
    processed_data_dir: str = Field("data/processed", env="PROCESSED_DATA_DIR")
    results_dir: str = Field("data/results", env="RESULTS_DIR")
    logs_dir: str = Field("logs", env="LOGS_DIR")
    
    @field_validator("benchmark_seeds", mode="before")
    @classmethod
    def parse_seeds(cls, v):
        """Parse comma-separated seeds from environment variable."""
        if isinstance(v, str):
            # Remove any whitespace and split by comma
            clean_str = v.strip()
            if clean_str:
                try:
                    # Handle JSON-like format
                    if clean_str.startswith('[') and clean_str.endswith(']'):
                        import json
                        return json.loads(clean_str)
                    else:
                        # Handle comma-separated format
                        parts = [x.strip() for x in clean_str.split(",") if x.strip()]
                        return [int(x) for x in parts]
                except (ValueError, json.JSONDecodeError):
                    pass
        elif isinstance(v, list):
            return v
        
        # Default fallback
        return [42, 123, 456]
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @property
    def sec_filings_dir(self) -> str:
        """Directory for SEC filings."""
        return os.path.join(self.raw_data_dir, "sec_filings")
    
    @property
    def earnings_transcripts_dir(self) -> str:
        """Directory for earnings transcripts."""
        return os.path.join(self.raw_data_dir, "earnings_transcripts")
    
    @property
    def market_data_dir(self) -> str:
        """Directory for market data."""
        return os.path.join(self.raw_data_dir, "market_data")
    
    @property
    def news_data_dir(self) -> str:
        """Directory for news data."""
        return os.path.join(self.raw_data_dir, "news_data")
    
    @property
    def embeddings_dir(self) -> str:
        """Directory for embeddings."""
        return os.path.join(self.processed_data_dir, "embeddings")
    
    @property
    def ground_truth_dir(self) -> str:
        """Directory for ground truth data."""
        return os.path.join(self.processed_data_dir, "ground_truth")
    
    @property
    def benchmark_runs_dir(self) -> str:
        """Directory for benchmark run results."""
        return os.path.join(self.results_dir, "benchmark_runs")
    
    @property
    def model_outputs_dir(self) -> str:
        """Directory for model outputs."""
        return os.path.join(self.results_dir, "model_outputs")
    
    def create_directories(self) -> None:
        """Create all necessary directories."""
        directories = [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.results_dir,
            self.logs_dir,
            self.sec_filings_dir,
            self.earnings_transcripts_dir,
            self.market_data_dir,
            self.news_data_dir,
            self.embeddings_dir,
            self.ground_truth_dir,
            self.benchmark_runs_dir,
            self.model_outputs_dir,
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()