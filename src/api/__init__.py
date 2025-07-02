"""
API package initialization for Lift-os-LLM microservice.

Combines all API route modules and provides the main API router.
"""

from fastapi import APIRouter
from .routes import analysis, models, batch, training

# Create main API router
api_router = APIRouter(prefix="/api/v1")

# Include all route modules
api_router.include_router(analysis.router)
api_router.include_router(models.router)
api_router.include_router(batch.router)
api_router.include_router(training.router)

__all__ = ["api_router"]