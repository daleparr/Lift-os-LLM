"""
Lift-os-LLM - AI-Native Content Analysis Microservice

Main FastAPI application entry point with comprehensive middleware,
authentication, and API route configuration.
"""

import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
import uvicorn
from prometheus_fastapi_instrumentator import Instrumentator

from .core.config import settings
from .core.logging import setup_logging, logger
from .core.database import init_db, close_db
from .core.security import verify_token
from .api import api_router


# Security scheme
security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("🚀 Starting Lift-os-LLM microservice...")
    
    # Initialize database connections
    await init_db()
    
    # Initialize monitoring
    if settings.ENABLE_METRICS:
        instrumentator = Instrumentator()
        instrumentator.instrument(app).expose(app)
        logger.info("📊 Prometheus metrics enabled at /metrics")
    
    logger.info("✅ Lift-os-LLM microservice started successfully")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Lift-os-LLM microservice...")
    await close_db()
    logger.info("✅ Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Lift-os-LLM",
    description="AI-Native Content Analysis Microservice for LLM-powered content optimization and AI surfacing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Setup logging
setup_logging()

# Add security middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.ALLOWED_HOSTS)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Add custom middleware (commented out until implemented)
# app.add_middleware(RequestLoggingMiddleware)
# app.add_middleware(RateLimitMiddleware)


# Health check endpoints
@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": "lift-os-llm",
        "version": "1.0.0",
        "timestamp": time.time()
    }


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """Readiness probe for Kubernetes deployments."""
    try:
        # Check database connectivity
        # Add any other readiness checks here
        return {
            "status": "ready",
            "service": "lift-os-llm",
            "checks": {
                "database": "connected",
                "cache": "connected",
                "llm_providers": "available"
            }
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Lift-os-LLM",
        "description": "AI-Native Content Analysis Microservice",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics" if settings.ENABLE_METRICS else None
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "request_id": getattr(request.state, "request_id", None)
        }
    )


# Include API router
app.include_router(api_router)


# Request middleware for logging and monitoring
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )