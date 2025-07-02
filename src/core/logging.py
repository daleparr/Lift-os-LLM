"""
Logging configuration for Lift-os-LLM microservice.

Provides structured logging with JSON formatting, request correlation,
and integration with monitoring systems.
"""

import logging
import sys
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from loguru import logger
from pythonjsonlogger import jsonlogger

from .config import settings


class StructuredLogger:
    """Structured logger with JSON formatting and correlation IDs."""
    
    def __init__(self):
        self.correlation_id: Optional[str] = None
        self.request_id: Optional[str] = None
        self.user_id: Optional[str] = None
    
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for request tracking."""
        self.correlation_id = correlation_id
    
    def set_request_id(self, request_id: str):
        """Set request ID for request tracking."""
        self.request_id = request_id
    
    def set_user_id(self, user_id: str):
        """Set user ID for user tracking."""
        self.user_id = user_id
    
    def _get_extra_fields(self) -> Dict[str, Any]:
        """Get extra fields for structured logging."""
        fields = {
            "service": "lift-os-llm",
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        if self.correlation_id:
            fields["correlation_id"] = self.correlation_id
        if self.request_id:
            fields["request_id"] = self.request_id
        if self.user_id:
            fields["user_id"] = self.user_id
        
        return fields
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data."""
        extra = self._get_extra_fields()
        extra.update(kwargs)
        logger.info(message, **extra)
    
    def error(self, message: str, **kwargs):
        """Log error message with structured data."""
        extra = self._get_extra_fields()
        extra.update(kwargs)
        logger.error(message, **extra)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data."""
        extra = self._get_extra_fields()
        extra.update(kwargs)
        logger.warning(message, **extra)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data."""
        extra = self._get_extra_fields()
        extra.update(kwargs)
        logger.debug(message, **extra)


class RequestLoggingFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for request logging."""
    
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp
        log_record['timestamp'] = datetime.utcnow().isoformat()
        
        # Add service information
        log_record['service'] = 'lift-os-llm'
        log_record['version'] = '1.0.0'
        
        # Add log level
        log_record['level'] = record.levelname
        
        # Add module information
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno


def setup_logging():
    """Setup logging configuration for the application."""
    
    # Remove default loguru handler
    logger.remove()
    
    # Configure loguru with JSON formatting
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
        level=settings.LOG_LEVEL,
        serialize=True if settings.LOG_LEVEL == "DEBUG" else False,
        backtrace=True,
        diagnose=True
    )
    
    # Add file logging for production
    if not settings.DEBUG:
        logger.add(
            "logs/lift-os-llm.log",
            rotation="100 MB",
            retention="30 days",
            compression="gzip",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
            level=settings.LOG_LEVEL,
            serialize=True
        )
    
    # Configure standard library logging to use loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            
            # Find caller from where originated the logged message
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            
            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )
    
    # Intercept standard library logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Configure specific loggers
    for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"]:
        logging.getLogger(logger_name).handlers = [InterceptHandler()]
    
    logger.info("🔧 Logging configuration initialized")


def log_api_request(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    request_id: str,
    user_id: Optional[str] = None,
    **kwargs
):
    """Log API request with structured data."""
    logger.info(
        "API Request",
        method=method,
        path=path,
        status_code=status_code,
        duration_ms=duration_ms,
        request_id=request_id,
        user_id=user_id,
        **kwargs
    )


def log_analysis_request(
    analysis_type: str,
    content_length: int,
    processing_time_ms: float,
    ai_surfacing_score: Optional[float] = None,
    request_id: Optional[str] = None,
    **kwargs
):
    """Log content analysis request."""
    logger.info(
        "Content Analysis",
        analysis_type=analysis_type,
        content_length=content_length,
        processing_time_ms=processing_time_ms,
        ai_surfacing_score=ai_surfacing_score,
        request_id=request_id,
        **kwargs
    )


def log_optimization_request(
    optimization_type: str,
    original_score: float,
    optimized_score: float,
    improvement: float,
    processing_time_ms: float,
    request_id: Optional[str] = None,
    **kwargs
):
    """Log content optimization request."""
    logger.info(
        "Content Optimization",
        optimization_type=optimization_type,
        original_score=original_score,
        optimized_score=optimized_score,
        improvement=improvement,
        processing_time_ms=processing_time_ms,
        request_id=request_id,
        **kwargs
    )


def log_model_evaluation(
    model_name: str,
    evaluation_type: str,
    score: float,
    processing_time_ms: float,
    token_usage: Optional[Dict[str, int]] = None,
    request_id: Optional[str] = None,
    **kwargs
):
    """Log model evaluation."""
    logger.info(
        "Model Evaluation",
        model_name=model_name,
        evaluation_type=evaluation_type,
        score=score,
        processing_time_ms=processing_time_ms,
        token_usage=token_usage,
        request_id=request_id,
        **kwargs
    )


def log_batch_job(
    job_id: str,
    job_type: str,
    status: str,
    total_items: int,
    processed_items: int,
    failed_items: int,
    processing_time_ms: Optional[float] = None,
    **kwargs
):
    """Log batch job status."""
    logger.info(
        "Batch Job",
        job_id=job_id,
        job_type=job_type,
        status=status,
        total_items=total_items,
        processed_items=processed_items,
        failed_items=failed_items,
        processing_time_ms=processing_time_ms,
        **kwargs
    )


def log_error(
    error_type: str,
    error_message: str,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **kwargs
):
    """Log error with structured data."""
    logger.error(
        "Application Error",
        error_type=error_type,
        error_message=error_message,
        request_id=request_id,
        user_id=user_id,
        **kwargs
    )


# Global structured logger instance
structured_logger = StructuredLogger()