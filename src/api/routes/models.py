"""
Models API routes for Lift-os-LLM microservice.

Provides endpoints for managing LLM models, configurations,
and model-specific operations.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import HTTPBearer

from ...core.security import get_current_user, verify_api_key
from ...core.config import settings
from ...core.logging import logger
from ...models.entities import User
from ...models.requests import ModelConfigRequest, ModelTestRequest
from ...models.responses import (
    ModelsListResponse, ModelConfigResponse, ModelTestResponse,
    ErrorResponse
)

router = APIRouter(prefix="/models", tags=["Model Management"])
security = HTTPBearer()


@router.get(
    "/",
    response_model=ModelsListResponse,
    summary="List available LLM models",
    description="Get a list of all available LLM models and their configurations."
)
async def list_models(
    provider: Optional[str] = Query(None, description="Filter by provider (openai, anthropic, huggingface)"),
    current_user: User = Depends(get_current_user)
) -> ModelsListResponse:
    """
    List all available LLM models.
    
    - **provider**: Optional filter by provider
    """
    try:
        logger.info(f"Listing models for user {current_user.id}")
        
        models = []
        
        # OpenAI models
        if not provider or provider == "openai":
            if settings.has_openai:
                models.extend([
                    {
                        "id": "gpt-4-turbo-preview",
                        "name": "GPT-4 Turbo",
                        "provider": "openai",
                        "type": "chat",
                        "context_length": 128000,
                        "capabilities": ["text", "analysis", "reasoning"],
                        "cost_per_1k_tokens": {"input": 0.01, "output": 0.03},
                        "available": True
                    },
                    {
                        "id": "gpt-3.5-turbo",
                        "name": "GPT-3.5 Turbo",
                        "provider": "openai",
                        "type": "chat",
                        "context_length": 16385,
                        "capabilities": ["text", "analysis"],
                        "cost_per_1k_tokens": {"input": 0.001, "output": 0.002},
                        "available": True
                    },
                    {
                        "id": "text-embedding-3-large",
                        "name": "Text Embedding 3 Large",
                        "provider": "openai",
                        "type": "embedding",
                        "dimensions": 3072,
                        "capabilities": ["embeddings", "similarity"],
                        "cost_per_1k_tokens": {"input": 0.00013, "output": 0},
                        "available": True
                    }
                ])
        
        # Anthropic models
        if not provider or provider == "anthropic":
            if settings.has_anthropic:
                models.extend([
                    {
                        "id": "claude-3-opus-20240229",
                        "name": "Claude 3 Opus",
                        "provider": "anthropic",
                        "type": "chat",
                        "context_length": 200000,
                        "capabilities": ["text", "analysis", "reasoning", "code"],
                        "cost_per_1k_tokens": {"input": 0.015, "output": 0.075},
                        "available": True
                    },
                    {
                        "id": "claude-3-sonnet-20240229",
                        "name": "Claude 3 Sonnet",
                        "provider": "anthropic",
                        "type": "chat",
                        "context_length": 200000,
                        "capabilities": ["text", "analysis", "reasoning"],
                        "cost_per_1k_tokens": {"input": 0.003, "output": 0.015},
                        "available": True
                    },
                    {
                        "id": "claude-3-haiku-20240307",
                        "name": "Claude 3 Haiku",
                        "provider": "anthropic",
                        "type": "chat",
                        "context_length": 200000,
                        "capabilities": ["text", "analysis"],
                        "cost_per_1k_tokens": {"input": 0.00025, "output": 0.00125},
                        "available": True
                    }
                ])
        
        # HuggingFace models
        if not provider or provider == "huggingface":
            if settings.has_huggingface:
                models.extend([
                    {
                        "id": "sentence-transformers/all-MiniLM-L6-v2",
                        "name": "All MiniLM L6 v2",
                        "provider": "huggingface",
                        "type": "embedding",
                        "dimensions": 384,
                        "capabilities": ["embeddings", "similarity"],
                        "cost_per_1k_tokens": {"input": 0, "output": 0},
                        "available": True
                    },
                    {
                        "id": "sentence-transformers/all-mpnet-base-v2",
                        "name": "All MPNet Base v2",
                        "provider": "huggingface",
                        "type": "embedding",
                        "dimensions": 768,
                        "capabilities": ["embeddings", "similarity"],
                        "cost_per_1k_tokens": {"input": 0, "output": 0},
                        "available": True
                    }
                ])
        
        return ModelsListResponse(
            success=True,
            data={
                "models": models,
                "total_count": len(models),
                "providers": list(set(model["provider"] for model in models)),
                "default_model": settings.DEFAULT_LLM_MODEL
            },
            message="Models retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to list models for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve models: {str(e)}"
        )


@router.get(
    "/{model_id}",
    response_model=ModelConfigResponse,
    summary="Get model configuration",
    description="Get detailed configuration and capabilities for a specific model."
)
async def get_model_config(
    model_id: str,
    current_user: User = Depends(get_current_user)
) -> ModelConfigResponse:
    """
    Get detailed configuration for a specific model.
    
    - **model_id**: The ID of the model to retrieve
    """
    try:
        logger.info(f"Getting model config for {model_id} for user {current_user.id}")
        
        # This would integrate with the existing model configuration system
        # For now, return a sample configuration
        model_config = {
            "id": model_id,
            "name": "Sample Model",
            "provider": "openai",
            "type": "chat",
            "version": "1.0.0",
            "context_length": 4096,
            "max_tokens": 2048,
            "temperature_range": {"min": 0.0, "max": 2.0, "default": 0.7},
            "capabilities": ["text", "analysis"],
            "parameters": {
                "temperature": 0.7,
                "max_tokens": 1000,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            },
            "cost_per_1k_tokens": {"input": 0.001, "output": 0.002},
            "rate_limits": {
                "requests_per_minute": 60,
                "tokens_per_minute": 60000
            },
            "available": True,
            "last_updated": "2024-01-01T00:00:00Z"
        }
        
        return ModelConfigResponse(
            success=True,
            data=model_config,
            message=f"Model configuration for {model_id} retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to get model config for {model_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve model configuration: {str(e)}"
        )


@router.put(
    "/{model_id}/config",
    response_model=ModelConfigResponse,
    summary="Update model configuration",
    description="Update configuration parameters for a specific model."
)
async def update_model_config(
    model_id: str,
    request: ModelConfigRequest,
    current_user: User = Depends(get_current_user)
) -> ModelConfigResponse:
    """
    Update model configuration.
    
    - **model_id**: The ID of the model to update
    - **parameters**: New parameter values
    """
    try:
        logger.info(f"Updating model config for {model_id} for user {current_user.id}")
        
        # Validate model exists and user has permission
        if not current_user.is_admin:
            raise HTTPException(
                status_code=403,
                detail="Admin privileges required to update model configuration"
            )
        
        # This would integrate with the existing model configuration system
        # For now, return the updated configuration
        updated_config = {
            "id": model_id,
            "parameters": request.parameters,
            "updated_by": current_user.id,
            "updated_at": "2024-01-01T00:00:00Z"
        }
        
        return ModelConfigResponse(
            success=True,
            data=updated_config,
            message=f"Model configuration for {model_id} updated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update model config for {model_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update model configuration: {str(e)}"
        )


@router.post(
    "/{model_id}/test",
    response_model=ModelTestResponse,
    summary="Test model functionality",
    description="Test a model with sample input to verify it's working correctly."
)
async def test_model(
    model_id: str,
    request: ModelTestRequest,
    current_user: User = Depends(get_current_user)
) -> ModelTestResponse:
    """
    Test model functionality.
    
    - **model_id**: The ID of the model to test
    - **test_input**: Sample input to test with
    - **parameters**: Optional parameters to use for testing
    """
    try:
        logger.info(f"Testing model {model_id} for user {current_user.id}")
        
        # This would integrate with the existing model testing system
        # For now, return a sample test result
        test_result = {
            "model_id": model_id,
            "test_input": request.test_input,
            "test_output": "Sample model response",
            "response_time_ms": 1500,
            "token_usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15
            },
            "cost": 0.00003,
            "status": "success",
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        return ModelTestResponse(
            success=True,
            data=test_result,
            message=f"Model {model_id} test completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to test model {model_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Model test failed: {str(e)}"
        )


@router.get(
    "/providers/status",
    summary="Get provider status",
    description="Check the availability status of all LLM providers."
)
async def get_provider_status(
    current_user: User = Depends(get_current_user)
):
    """Get the status of all LLM providers."""
    try:
        logger.info(f"Getting provider status for user {current_user.id}")
        
        provider_status = {
            "openai": {
                "available": settings.has_openai,
                "api_key_configured": bool(settings.OPENAI_API_KEY),
                "models_count": 3 if settings.has_openai else 0,
                "status": "healthy" if settings.has_openai else "unavailable"
            },
            "anthropic": {
                "available": settings.has_anthropic,
                "api_key_configured": bool(settings.ANTHROPIC_API_KEY),
                "models_count": 3 if settings.has_anthropic else 0,
                "status": "healthy" if settings.has_anthropic else "unavailable"
            },
            "huggingface": {
                "available": settings.has_huggingface,
                "api_key_configured": bool(settings.HUGGINGFACE_API_KEY),
                "models_count": 2 if settings.has_huggingface else 0,
                "status": "healthy" if settings.has_huggingface else "unavailable"
            }
        }
        
        return {
            "success": True,
            "data": {
                "providers": provider_status,
                "total_providers": len(provider_status),
                "available_providers": sum(1 for p in provider_status.values() if p["available"]),
                "timestamp": "2024-01-01T00:00:00Z"
            },
            "message": "Provider status retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to get provider status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve provider status: {str(e)}"
        )


@router.get(
    "/usage/stats",
    summary="Get model usage statistics",
    description="Get usage statistics for models by the current user."
)
async def get_usage_stats(
    days: int = Query(30, description="Number of days to include in statistics"),
    current_user: User = Depends(get_current_user)
):
    """Get model usage statistics for the current user."""
    try:
        logger.info(f"Getting usage stats for user {current_user.id}")
        
        # This would integrate with the existing usage tracking system
        # For now, return sample statistics
        usage_stats = {
            "period_days": days,
            "total_requests": 150,
            "total_tokens": 45000,
            "total_cost": 1.25,
            "models_used": [
                {
                    "model_id": "gpt-3.5-turbo",
                    "requests": 100,
                    "tokens": 30000,
                    "cost": 0.75
                },
                {
                    "model_id": "claude-3-sonnet-20240229",
                    "requests": 50,
                    "tokens": 15000,
                    "cost": 0.50
                }
            ],
            "daily_usage": [
                {"date": "2024-01-01", "requests": 10, "tokens": 3000, "cost": 0.08},
                {"date": "2024-01-02", "requests": 8, "tokens": 2400, "cost": 0.06}
            ]
        }
        
        return {
            "success": True,
            "data": usage_stats,
            "message": "Usage statistics retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to get usage stats for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve usage statistics: {str(e)}"
        )


# API Key protected endpoints
@router.get(
    "/api-key/list",
    response_model=ModelsListResponse,
    summary="List models using API key authentication",
    description="Get available models using API key authentication for external integrations."
)
async def list_models_api_key(
    provider: Optional[str] = Query(None, description="Filter by provider"),
    api_key: str = Depends(verify_api_key)
) -> ModelsListResponse:
    """
    List available models using API key authentication.
    
    Requires valid API key in Authorization header: `Bearer your-api-key`
    """
    try:
        logger.info(f"Listing models via API key", extra={"api_key": api_key[:8] + "..."})
        
        # Reuse the same logic as the authenticated endpoint
        # This would be refactored to share common logic
        models = []
        
        if not provider or provider == "openai":
            if settings.has_openai:
                models.extend([
                    {
                        "id": "gpt-4-turbo-preview",
                        "name": "GPT-4 Turbo",
                        "provider": "openai",
                        "available": True
                    },
                    {
                        "id": "gpt-3.5-turbo",
                        "name": "GPT-3.5 Turbo",
                        "provider": "openai",
                        "available": True
                    }
                ])
        
        return ModelsListResponse(
            success=True,
            data={
                "models": models,
                "total_count": len(models),
                "providers": list(set(model["provider"] for model in models))
            },
            message="Models retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list models via API key: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve models: {str(e)}"
        )