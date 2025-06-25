"""
Logging configuration for LLM Finance Leaderboard.

Centralized logging setup with structured logging and multiple output formats.
"""

import sys
import os
from pathlib import Path
from loguru import logger
from typing import Optional

from ..config.settings import settings


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_json: bool = False
) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, uses default from settings)
        enable_json: Whether to enable JSON structured logging
    """
    # Remove default logger
    logger.remove()
    
    # Use provided level or fall back to settings
    level = log_level or settings.log_level
    
    # Console logging format
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # Add console handler
    logger.add(
        sys.stderr,
        format=console_format,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # File logging setup
    if log_file is None:
        # Create logs directory if it doesn't exist
        log_dir = Path(settings.logs_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "leaderboard.log"
    
    # File logging format
    if enable_json:
        file_format = "{time} | {level} | {name}:{function}:{line} | {message} | {extra}"
        logger.add(
            log_file,
            format=file_format,
            level=level,
            rotation="1 day",
            retention="30 days",
            compression="gz",
            serialize=True,  # JSON format
            backtrace=True,
            diagnose=True
        )
    else:
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )
        logger.add(
            log_file,
            format=file_format,
            level=level,
            rotation="1 day",
            retention="30 days",
            compression="gz",
            backtrace=True,
            diagnose=True
        )
    
    # Add error-specific log file
    error_log_file = Path(log_file).parent / "errors.log"
    logger.add(
        error_log_file,
        format=file_format,
        level="ERROR",
        rotation="1 week",
        retention="12 weeks",
        compression="gz",
        backtrace=True,
        diagnose=True
    )
    
    # Performance log for benchmark runs
    perf_log_file = Path(log_file).parent / "performance.log"
    logger.add(
        perf_log_file,
        format="{time} | {level} | {message}",
        level="INFO",
        rotation="1 day",
        retention="90 days",
        filter=lambda record: "PERF" in record["extra"]
    )
    
    logger.info(f"Logging initialized - Level: {level}, File: {log_file}")


def get_performance_logger():
    """Get a logger specifically for performance metrics."""
    return logger.bind(PERF=True)


def log_benchmark_start(run_id: str, models: list, tasks: list) -> None:
    """Log benchmark run start."""
    perf_logger = get_performance_logger()
    perf_logger.info(
        f"BENCHMARK_START | run_id={run_id} | models={len(models)} | tasks={len(tasks)} | "
        f"model_list={','.join(models)} | task_list={','.join(tasks)}"
    )


def log_benchmark_end(run_id: str, duration_minutes: float, success: bool) -> None:
    """Log benchmark run completion."""
    perf_logger = get_performance_logger()
    status = "SUCCESS" if success else "FAILED"
    perf_logger.info(
        f"BENCHMARK_END | run_id={run_id} | duration_minutes={duration_minutes:.2f} | status={status}"
    )


def log_model_evaluation(
    run_id: str,
    model_name: str,
    task_id: str,
    latency_ms: float,
    tokens_used: int,
    cost_usd: float,
    success: bool
) -> None:
    """Log individual model evaluation."""
    perf_logger = get_performance_logger()
    status = "SUCCESS" if success else "FAILED"
    perf_logger.info(
        f"MODEL_EVAL | run_id={run_id} | model={model_name} | task={task_id} | "
        f"latency_ms={latency_ms:.0f} | tokens={tokens_used} | cost_usd={cost_usd:.6f} | status={status}"
    )


def log_data_collection(
    source: str,
    operation: str,
    count: int,
    duration_seconds: float,
    success: bool
) -> None:
    """Log data collection operations."""
    perf_logger = get_performance_logger()
    status = "SUCCESS" if success else "FAILED"
    perf_logger.info(
        f"DATA_COLLECTION | source={source} | operation={operation} | "
        f"count={count} | duration_seconds={duration_seconds:.2f} | status={status}"
    )


def log_vector_store_operation(
    operation: str,
    document_count: int,
    duration_seconds: float,
    index_name: str,
    success: bool
) -> None:
    """Log vector store operations."""
    perf_logger = get_performance_logger()
    status = "SUCCESS" if success else "FAILED"
    perf_logger.info(
        f"VECTOR_STORE | operation={operation} | documents={document_count} | "
        f"duration_seconds={duration_seconds:.2f} | index={index_name} | status={status}"
    )


class ContextualLogger:
    """Contextual logger that adds run_id and other context to all log messages."""
    
    def __init__(self, run_id: str = None, model_name: str = None):
        self.context = {}
        if run_id:
            self.context["run_id"] = run_id
        if model_name:
            self.context["model"] = model_name
        
        self.logger = logger.bind(**self.context)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with context."""
        self.logger.bind(**kwargs).debug(message)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with context."""
        self.logger.bind(**kwargs).info(message)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with context."""
        self.logger.bind(**kwargs).warning(message)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message with context."""
        self.logger.bind(**kwargs).error(message)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message with context."""
        self.logger.bind(**kwargs).critical(message)


def setup_wandb_logging() -> None:
    """Setup Weights & Biases logging if configured."""
    if not settings.wandb_api_key or not settings.wandb_project:
        logger.info("W&B logging not configured, skipping")
        return
    
    try:
        import wandb
        
        wandb.init(
            project=settings.wandb_project,
            config={
                "framework": "llm-finance-leaderboard",
                "version": "0.1.0",
            }
        )
        
        logger.info("W&B logging initialized")
        
    except ImportError:
        logger.warning("wandb package not installed, skipping W&B logging")
    except Exception as e:
        logger.error(f"Failed to initialize W&B logging: {e}")


def log_system_info() -> None:
    """Log system information for debugging."""
    import platform
    import psutil
    
    logger.info(f"System: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"CPU: {psutil.cpu_count()} cores")
    logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # GPU info if available
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        else:
            logger.info("CUDA: Not available")
    except ImportError:
        logger.info("PyTorch not installed, GPU info unavailable")


# Initialize logging on module import
if not logger._core.handlers:
    setup_logging()