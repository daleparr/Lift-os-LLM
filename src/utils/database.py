"""
Database utilities for LLM Finance Leaderboard.

Handles SQLite database initialization and session management.
"""

import os
from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from loguru import logger

from ..config.settings import settings

# Database models
Base = declarative_base()


class BenchmarkRun(Base):
    """Database model for benchmark runs."""
    __tablename__ = "benchmark_runs"
    
    id = Column(Integer, primary_key=True)
    run_id = Column(String, unique=True, nullable=False)
    run_name = Column(String, nullable=False)
    description = Column(Text)
    status = Column(String, default="running")
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    total_duration_minutes = Column(Float)
    created_by = Column(String)
    error_message = Column(Text)


class ModelResult(Base):
    """Database model for model evaluation results."""
    __tablename__ = "model_results"
    
    id = Column(Integer, primary_key=True)
    run_id = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    final_score = Column(Float, nullable=False)
    quality_score = Column(Float, nullable=False)
    efficiency_score = Column(Float, nullable=False)
    low_complexity_score = Column(Float, nullable=False)
    medium_complexity_score = Column(Float, nullable=False)
    high_complexity_score = Column(Float, nullable=False)
    avg_latency_ms = Column(Float, nullable=False)
    avg_cost_per_task = Column(Float, nullable=False)
    completion_rate = Column(Float, nullable=False)
    created_at = Column(DateTime, nullable=False)


class TaskResult(Base):
    """Database model for individual task results."""
    __tablename__ = "task_results"
    
    id = Column(Integer, primary_key=True)
    run_id = Column(String, nullable=False)
    task_id = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    task_complexity = Column(String, nullable=False)
    prompt = Column(Text, nullable=False)
    expected_answer = Column(Text, nullable=False)
    model_response = Column(Text, nullable=False)
    exact_match_score = Column(Float)
    f1_score = Column(Float)
    rouge_1_score = Column(Float)
    rouge_2_score = Column(Float)
    bleu_score = Column(Float)
    human_rating = Column(Float)
    latency_ms = Column(Float, nullable=False)
    tokens_used = Column(Integer, nullable=False)
    cost_usd = Column(Float, nullable=False)
    task_completed = Column(Boolean, nullable=False)
    evaluation_seed = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False)


# Database engine and session
engine = None
SessionLocal = None


def get_engine():
    """Get database engine, initializing if necessary."""
    global engine
    if engine is None:
        init_database()
    return engine


def get_session_local():
    """Get session factory, initializing if necessary."""
    global SessionLocal
    if SessionLocal is None:
        init_database()
    return SessionLocal


def init_database() -> None:
    """Initialize the database and create tables."""
    global engine, SessionLocal
    
    try:
        # Ensure database directory exists
        db_path = Path(settings.database_url.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create engine
        engine = create_engine(
            settings.database_url,
            connect_args={"check_same_thread": False}  # For SQLite
        )
        
        # Create session factory
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # Create tables
        Base.metadata.create_all(bind=engine)
        
        logger.info(f"Database initialized at {settings.database_url}")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def get_database_session() -> Generator[Session, None, None]:
    """Get a database session."""
    session_local = get_session_local()
    db = session_local()
    try:
        yield db
    finally:
        db.close()


def get_db() -> Session:
    """Get a database session (synchronous version)."""
    session_local = get_session_local()
    return session_local()


def close_database() -> None:
    """Close database connections."""
    global engine
    if engine:
        engine.dispose()
        logger.info("Database connections closed")


# Database utility functions
def save_benchmark_run(run_data: dict) -> None:
    """Save benchmark run to database."""
    db = get_db()
    try:
        benchmark_run = BenchmarkRun(**run_data)
        db.add(benchmark_run)
        db.commit()
        logger.info(f"Saved benchmark run: {run_data['run_id']}")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save benchmark run: {e}")
        raise
    finally:
        db.close()


def save_model_result(result_data: dict) -> None:
    """Save model evaluation result to database."""
    db = get_db()
    try:
        model_result = ModelResult(**result_data)
        db.add(model_result)
        db.commit()
        logger.info(f"Saved model result: {result_data['model_name']}")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save model result: {e}")
        raise
    finally:
        db.close()


def save_task_result(task_data: dict) -> None:
    """Save task result to database."""
    db = get_db()
    try:
        task_result = TaskResult(**task_data)
        db.add(task_result)
        db.commit()
        logger.debug(f"Saved task result: {task_data['task_id']}")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save task result: {e}")
        raise
    finally:
        db.close()


def get_latest_benchmark_runs(limit: int = 10) -> list:
    """Get latest benchmark runs."""
    db = get_db()
    try:
        runs = db.query(BenchmarkRun).order_by(BenchmarkRun.start_time.desc()).limit(limit).all()
        return [
            {
                "run_id": run.run_id,
                "run_name": run.run_name,
                "status": run.status,
                "start_time": run.start_time,
                "end_time": run.end_time,
                "duration_minutes": run.total_duration_minutes,
            }
            for run in runs
        ]
    finally:
        db.close()


def get_model_leaderboard(run_id: str = None) -> list:
    """Get model leaderboard data."""
    db = get_db()
    try:
        query = db.query(ModelResult)
        if run_id:
            query = query.filter(ModelResult.run_id == run_id)
        
        results = query.order_by(ModelResult.final_score.desc()).all()
        
        return [
            {
                "model_name": result.model_name,
                "final_score": result.final_score,
                "quality_score": result.quality_score,
                "efficiency_score": result.efficiency_score,
                "low_score": result.low_complexity_score,
                "medium_score": result.medium_complexity_score,
                "high_score": result.high_complexity_score,
                "avg_latency_ms": result.avg_latency_ms,
                "avg_cost_per_task": result.avg_cost_per_task,
                "completion_rate": result.completion_rate,
            }
            for result in results
        ]
    finally:
        db.close()


def get_task_performance(run_id: str = None, model_name: str = None) -> list:
    """Get task performance data."""
    db = get_db()
    try:
        query = db.query(TaskResult)
        if run_id:
            query = query.filter(TaskResult.run_id == run_id)
        if model_name:
            query = query.filter(TaskResult.model_name == model_name)
        
        results = query.all()
        
        return [
            {
                "task_id": result.task_id,
                "model_name": result.model_name,
                "complexity": result.task_complexity,
                "f1_score": result.f1_score,
                "rouge_1_score": result.rouge_1_score,
                "human_rating": result.human_rating,
                "latency_ms": result.latency_ms,
                "cost_usd": result.cost_usd,
                "completed": result.task_completed,
            }
            for result in results
        ]
    finally:
        db.close()