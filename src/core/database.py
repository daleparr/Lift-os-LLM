"""
Database configuration and utilities for Lift-os-LLM microservice.

Handles SQLAlchemy setup, connection management, and database operations.
"""

import asyncio
from typing import AsyncGenerator, Optional
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import redis.asyncio as redis
from contextlib import asynccontextmanager

from .config import settings
from .logging import logger


# Database metadata and base
metadata = MetaData()
Base = declarative_base(metadata=metadata)

# Database engines
async_engine = None
sync_engine = None
async_session_factory = None
sync_session_factory = None

# Redis connection
redis_client = None


async def init_db():
    """Initialize database connections and create tables."""
    global async_engine, sync_engine, async_session_factory, sync_session_factory, redis_client
    
    try:
        # Create async engine
        async_engine = create_async_engine(
            settings.database_url_async,
            echo=settings.DEBUG,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        
        # Create sync engine
        sync_engine = create_engine(
            settings.database_url_sync,
            echo=settings.DEBUG,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        
        # Create session factories
        async_session_factory = async_sessionmaker(
            async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        sync_session_factory = sessionmaker(
            sync_engine,
            autocommit=False,
            autoflush=False
        )
        
        # Initialize Redis
        redis_client = redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        
        # Test Redis connection
        await redis_client.ping()
        
        # Create tables
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("✅ Database connections initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        raise


async def close_db():
    """Close database connections."""
    global async_engine, sync_engine, redis_client
    
    try:
        if redis_client:
            await redis_client.close()
            logger.info("Redis connection closed")
        
        if async_engine:
            await async_engine.dispose()
            logger.info("Async database engine disposed")
        
        if sync_engine:
            sync_engine.dispose()
            logger.info("Sync database engine disposed")
            
        logger.info("✅ Database connections closed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error closing database connections: {e}")


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session with automatic cleanup."""
    if not async_session_factory:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_sync_session() -> Session:
    """Get sync database session."""
    if not sync_session_factory:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    
    return sync_session_factory()


async def get_redis() -> redis.Redis:
    """Get Redis client."""
    if not redis_client:
        raise RuntimeError("Redis not initialized. Call init_db() first.")
    
    return redis_client


class CacheManager:
    """Redis cache manager for application data."""
    
    def __init__(self):
        self.default_ttl = settings.CACHE_TTL_SECONDS
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        try:
            redis_conn = await get_redis()
            value = await redis_conn.get(key)
            if value:
                logger.debug(f"Cache hit for key: {key}")
            return value
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set value in cache with TTL."""
        try:
            redis_conn = await get_redis()
            ttl = ttl or self.default_ttl
            result = await redis_conn.setex(key, ttl, value)
            logger.debug(f"Cache set for key: {key}, TTL: {ttl}")
            return result
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            redis_conn = await get_redis()
            result = await redis_conn.delete(key)
            logger.debug(f"Cache delete for key: {key}")
            return bool(result)
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            redis_conn = await get_redis()
            result = await redis_conn.exists(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment counter in cache."""
        try:
            redis_conn = await get_redis()
            result = await redis_conn.incrby(key, amount)
            return result
        except Exception as e:
            logger.error(f"Cache increment error for key {key}: {e}")
            return None
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for existing key."""
        try:
            redis_conn = await get_redis()
            result = await redis_conn.expire(key, ttl)
            return bool(result)
        except Exception as e:
            logger.error(f"Cache expire error for key {key}: {e}")
            return False
    
    async def get_keys(self, pattern: str) -> list:
        """Get keys matching pattern."""
        try:
            redis_conn = await get_redis()
            keys = await redis_conn.keys(pattern)
            return keys
        except Exception as e:
            logger.error(f"Cache get_keys error for pattern {pattern}: {e}")
            return []
    
    async def flush_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        try:
            keys = await self.get_keys(pattern)
            if keys:
                redis_conn = await get_redis()
                result = await redis_conn.delete(*keys)
                logger.info(f"Flushed {result} keys matching pattern: {pattern}")
                return result
            return 0
        except Exception as e:
            logger.error(f"Cache flush_pattern error for pattern {pattern}: {e}")
            return 0


# Global cache manager instance
cache_manager = CacheManager()


class DatabaseHealthCheck:
    """Database health check utilities."""
    
    @staticmethod
    async def check_async_db() -> bool:
        """Check async database connectivity."""
        try:
            async with get_async_session() as session:
                await session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Async database health check failed: {e}")
            return False
    
    @staticmethod
    def check_sync_db() -> bool:
        """Check sync database connectivity."""
        try:
            with get_sync_session() as session:
                session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Sync database health check failed: {e}")
            return False
    
    @staticmethod
    async def check_redis() -> bool:
        """Check Redis connectivity."""
        try:
            redis_conn = await get_redis()
            await redis_conn.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    @staticmethod
    async def get_health_status() -> dict:
        """Get comprehensive database health status."""
        return {
            "async_database": await DatabaseHealthCheck.check_async_db(),
            "sync_database": DatabaseHealthCheck.check_sync_db(),
            "redis": await DatabaseHealthCheck.check_redis(),
        }


# Database dependency for FastAPI
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database session."""
    async with get_async_session() as session:
        yield session


# Cache dependency for FastAPI
async def get_cache() -> CacheManager:
    """FastAPI dependency for cache manager."""
    return cache_manager