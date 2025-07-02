"""
Configuration management for Lift-os-LLM microservice.

Handles environment variables, settings validation, and configuration
for different deployment environments.
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, validator
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    DEBUG: bool = False
    
    # Security
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALLOWED_HOSTS: List[str] = ["*"]
    CORS_ORIGINS: List[str] = ["*"]
    
    # LLM Providers
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    HUGGINGFACE_API_TOKEN: Optional[str] = None
    
    # Vector Database
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = None
    PINECONE_INDEX_NAME: str = "lift-os-llm"
    
    # Database
    DATABASE_URL: str = "sqlite:///./lift_os_llm.db"
    REDIS_URL: str = "redis://localhost:6379"
    
    # Monitoring & Logging
    LOG_LEVEL: str = "INFO"
    ENABLE_METRICS: bool = True
    SENTRY_DSN: Optional[str] = None
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 900  # 15 minutes
    
    # Processing
    MAX_CONCURRENT_ANALYSES: int = 10
    ANALYSIS_TIMEOUT_SECONDS: int = 300
    BATCH_SIZE_LIMIT: int = 100
    
    # Cache Settings
    CACHE_TTL_SECONDS: int = 3600
    CACHE_MAX_SIZE: int = 1000
    
    # Model Configuration
    DEFAULT_LLM_MODEL: str = "gpt-3.5-turbo"
    DEFAULT_EMBEDDING_MODEL: str = "text-embedding-ada-002"
    MAX_TOKENS: int = 4000
    TEMPERATURE: float = 0.1
    
    @validator("CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    @validator("ALLOWED_HOSTS", pre=True)
    def assemble_allowed_hosts(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    @validator("JWT_SECRET_KEY")
    def validate_jwt_secret(cls, v):
        if not v or len(v) < 32:
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters long")
        return v
    
    @property
    def database_url_sync(self) -> str:
        """Get synchronous database URL."""
        if self.DATABASE_URL.startswith("postgresql://"):
            return self.DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)
        return self.DATABASE_URL
    
    @property
    def database_url_async(self) -> str:
        """Get asynchronous database URL."""
        if self.DATABASE_URL.startswith("postgresql://"):
            return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
        return self.DATABASE_URL
    
    @property
    def has_openai(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.OPENAI_API_KEY)
    
    @property
    def has_anthropic(self) -> bool:
        """Check if Anthropic API key is configured."""
        return bool(self.ANTHROPIC_API_KEY)
    
    @property
    def has_huggingface(self) -> bool:
        """Check if HuggingFace token is configured."""
        return bool(self.HUGGINGFACE_API_TOKEN)
    
    @property
    def has_pinecone(self) -> bool:
        """Check if Pinecone is configured."""
        return bool(self.PINECONE_API_KEY and self.PINECONE_ENVIRONMENT)
    
    @property
    def available_llm_providers(self) -> List[str]:
        """Get list of available LLM providers."""
        providers = []
        if self.has_openai:
            providers.append("openai")
        if self.has_anthropic:
            providers.append("anthropic")
        if self.has_huggingface:
            providers.append("huggingface")
        return providers
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()


# Environment-specific configurations
class DevelopmentSettings(Settings):
    """Development environment settings."""
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    API_WORKERS: int = 1


class ProductionSettings(Settings):
    """Production environment settings."""
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    API_WORKERS: int = 4
    RATE_LIMIT_REQUESTS: int = 1000
    RATE_LIMIT_WINDOW: int = 900


class TestingSettings(Settings):
    """Testing environment settings."""
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    DATABASE_URL: str = "sqlite:///./test.db"
    REDIS_URL: str = "redis://localhost:6379/1"
    JWT_SECRET_KEY: str = "test-secret-key-for-testing-only-32-chars"


def get_environment_settings() -> Settings:
    """Get settings based on environment."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()