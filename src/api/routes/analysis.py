"""
Analysis API routes for Lift-os-LLM microservice.

Provides endpoints for content analysis, AI surfacing scores,
and optimization recommendations.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.security import HTTPBearer
import uuid

from ...core.security import get_current_user, verify_api_key
from ...core.logging import logger
from ...services.content_analysis import ContentAnalysisService
from ...models.entities import User, AnalysisType
from ...models.requests import (
    ContentAnalysisRequest, BatchAnalysisRequest, OptimizationRequest
)
from ...models.responses import (
    ContentAnalysisResponse, BatchAnalysisResponse, OptimizationResponse,
    ErrorResponse
)

router = APIRouter(prefix="/analysis", tags=["Content Analysis"])
security = HTTPBearer()

# Initialize content analysis service
content_service = ContentAnalysisService()


@router.post(
    "/analyze",
    response_model=ContentAnalysisResponse,
    summary="Analyze content for AI surfacing optimization",
    description="Perform comprehensive content analysis including AI surfacing scores, semantic analysis, and optimization recommendations."
)
async def analyze_content(
    request: ContentAnalysisRequest,
    current_user: User = Depends(get_current_user)
) -> ContentAnalysisResponse:
    """
    Analyze content and generate AI surfacing scores.
    
    - **url**: URL to analyze (optional if html provided)
    - **html**: HTML content to analyze (optional if url provided)
    - **title**: Content title (optional, will be extracted if not provided)
    - **description**: Content description (optional, will be extracted if not provided)
    - **analysis_type**: Type of analysis to perform
    - **include_embeddings**: Whether to include vector embeddings
    - **include_knowledge_graph**: Whether to include knowledge graph analysis
    """
    try:
        request_id = str(uuid.uuid4())
        logger.info(f"Starting content analysis for user {current_user.id}", extra={"request_id": request_id})
        
        # Validate input
        if not request.content.url and not request.content.html:
            raise HTTPException(
                status_code=400,
                detail="Either URL or HTML content must be provided"
            )
        
        # Perform analysis
        result = await content_service.analyze_content(
            content=request.content,
            analysis_type=request.analysis_type,
            include_embeddings=request.include_embeddings,
            include_knowledge_graph=request.include_knowledge_graph,
            model_override=request.model_override,
            request_id=request_id
        )
        
        logger.info(
            f"Content analysis completed for user {current_user.id}",
            extra={
                "request_id": request_id,
                "processing_time_ms": result.processing_time_ms,
                "ai_surfacing_score": result.ai_surfacing_score.overall
            }
        )
        
        return ContentAnalysisResponse(
            success=True,
            data=result,
            message="Content analysis completed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Content analysis failed for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Content analysis failed: {str(e)}"
        )


@router.post(
    "/batch",
    response_model=BatchAnalysisResponse,
    summary="Analyze multiple content items in batch",
    description="Submit multiple content items for batch analysis. Returns a job ID for tracking progress."
)
async def batch_analyze(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
) -> BatchAnalysisResponse:
    """
    Submit batch analysis job for multiple content items.
    
    - **content_items**: List of content items to analyze
    - **analysis_type**: Type of analysis to perform for all items
    - **include_embeddings**: Whether to include vector embeddings
    - **include_knowledge_graph**: Whether to include knowledge graph analysis
    - **callback_url**: Optional webhook URL for job completion notification
    """
    try:
        job_id = str(uuid.uuid4())
        logger.info(f"Starting batch analysis for user {current_user.id}", extra={"job_id": job_id})
        
        # Validate input
        if not request.content_items:
            raise HTTPException(
                status_code=400,
                detail="At least one content item must be provided"
            )
        
        if len(request.content_items) > 100:  # Limit batch size
            raise HTTPException(
                status_code=400,
                detail="Batch size cannot exceed 100 items"
            )
        
        # Add batch processing to background tasks
        background_tasks.add_task(
            process_batch_analysis,
            job_id=job_id,
            user_id=current_user.id,
            request=request
        )
        
        return BatchAnalysisResponse(
            success=True,
            data={
                "job_id": job_id,
                "status": "queued",
                "total_items": len(request.content_items),
                "estimated_completion_time": len(request.content_items) * 30  # 30 seconds per item estimate
            },
            message=f"Batch analysis job {job_id} queued successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch analysis submission failed for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Batch analysis submission failed: {str(e)}"
        )


@router.get(
    "/batch/{job_id}",
    response_model=BatchAnalysisResponse,
    summary="Get batch analysis job status",
    description="Retrieve the status and results of a batch analysis job."
)
async def get_batch_status(
    job_id: str,
    current_user: User = Depends(get_current_user)
) -> BatchAnalysisResponse:
    """
    Get the status and results of a batch analysis job.
    
    - **job_id**: The ID of the batch analysis job
    """
    try:
        # This would integrate with the existing job queue system
        # For now, return a placeholder response
        return BatchAnalysisResponse(
            success=True,
            data={
                "job_id": job_id,
                "status": "completed",
                "total_items": 0,
                "completed_items": 0,
                "failed_items": 0,
                "results": []
            },
            message="Batch analysis status retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to get batch status for job {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve batch status: {str(e)}"
        )


@router.post(
    "/optimize",
    response_model=OptimizationResponse,
    summary="Get content optimization recommendations",
    description="Analyze content and provide specific optimization recommendations for AI search engines."
)
async def optimize_content(
    request: OptimizationRequest,
    current_user: User = Depends(get_current_user)
) -> OptimizationResponse:
    """
    Get optimization recommendations for content.
    
    - **url**: URL to analyze and optimize
    - **target_score**: Target AI surfacing score to achieve
    - **focus_areas**: Specific areas to focus optimization on
    - **competitor_urls**: Optional competitor URLs for comparison
    """
    try:
        request_id = str(uuid.uuid4())
        logger.info(f"Starting content optimization for user {current_user.id}", extra={"request_id": request_id})
        
        # Validate input
        if not request.url:
            raise HTTPException(
                status_code=400,
                detail="URL must be provided for optimization"
            )
        
        # Perform content analysis first
        from ...models.entities import ContentInput
        content_input = ContentInput(url=request.url)
        
        analysis_result = await content_service.analyze_content(
            content=content_input,
            analysis_type=AnalysisType.COMPREHENSIVE,
            include_embeddings=True,
            include_knowledge_graph=True,
            request_id=request_id
        )
        
        # Generate optimization plan
        optimization_plan = await generate_optimization_plan(
            analysis_result=analysis_result,
            target_score=request.target_score,
            focus_areas=request.focus_areas,
            competitor_urls=request.competitor_urls
        )
        
        logger.info(
            f"Content optimization completed for user {current_user.id}",
            extra={
                "request_id": request_id,
                "current_score": analysis_result.ai_surfacing_score.overall,
                "target_score": request.target_score
            }
        )
        
        return OptimizationResponse(
            success=True,
            data={
                "current_analysis": analysis_result,
                "optimization_plan": optimization_plan,
                "estimated_improvement": request.target_score - analysis_result.ai_surfacing_score.overall
            },
            message="Content optimization plan generated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Content optimization failed for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Content optimization failed: {str(e)}"
        )


@router.get(
    "/health",
    summary="Analysis service health check",
    description="Check the health status of the content analysis service."
)
async def analysis_health():
    """Check analysis service health."""
    try:
        # Test service components
        service_status = {
            "content_analysis": "healthy",
            "llm_providers": {
                "openai": "available" if content_service.openai_client else "unavailable",
                "anthropic": "available" if content_service.anthropic_client else "unavailable"
            },
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        return {
            "status": "healthy",
            "services": service_status
        }
        
    except Exception as e:
        logger.error(f"Analysis health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Analysis service unhealthy"
        )


# API Key protected endpoints
@router.post(
    "/api-key/analyze",
    response_model=ContentAnalysisResponse,
    summary="Analyze content using API key authentication",
    description="Perform content analysis using API key authentication for external integrations."
)
async def analyze_content_api_key(
    request: ContentAnalysisRequest,
    api_key: str = Depends(verify_api_key)
) -> ContentAnalysisResponse:
    """
    Analyze content using API key authentication.
    
    Requires valid API key in Authorization header: `Bearer your-api-key`
    """
    try:
        request_id = str(uuid.uuid4())
        logger.info(f"Starting API key content analysis", extra={"request_id": request_id, "api_key": api_key[:8] + "..."})
        
        # Validate input
        if not request.content.url and not request.content.html:
            raise HTTPException(
                status_code=400,
                detail="Either URL or HTML content must be provided"
            )
        
        # Perform analysis
        result = await content_service.analyze_content(
            content=request.content,
            analysis_type=request.analysis_type,
            include_embeddings=request.include_embeddings,
            include_knowledge_graph=request.include_knowledge_graph,
            model_override=request.model_override,
            request_id=request_id
        )
        
        logger.info(
            f"API key content analysis completed",
            extra={
                "request_id": request_id,
                "processing_time_ms": result.processing_time_ms,
                "ai_surfacing_score": result.ai_surfacing_score.overall
            }
        )
        
        return ContentAnalysisResponse(
            success=True,
            data=result,
            message="Content analysis completed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API key content analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Content analysis failed: {str(e)}"
        )


# Helper functions
async def process_batch_analysis(job_id: str, user_id: int, request: BatchAnalysisRequest):
    """Process batch analysis in background."""
    try:
        logger.info(f"Processing batch analysis job {job_id} for user {user_id}")
        
        # This would integrate with the existing job queue system
        # Process each content item and store results
        
        # For now, just log the completion
        logger.info(f"Batch analysis job {job_id} completed for user {user_id}")
        
    except Exception as e:
        logger.error(f"Batch analysis job {job_id} failed: {e}", exc_info=True)


async def generate_optimization_plan(
    analysis_result,
    target_score: float,
    focus_areas: Optional[List[str]] = None,
    competitor_urls: Optional[List[str]] = None
):
    """Generate optimization plan based on analysis results."""
    try:
        # This would implement the optimization planning logic
        # For now, return the existing recommendations
        return {
            "recommendations": analysis_result.recommendations,
            "priority_actions": [rec for rec in analysis_result.recommendations if rec.priority == "critical"],
            "estimated_timeline": "2-4 weeks",
            "success_probability": 0.85
        }
        
    except Exception as e:
        logger.error(f"Optimization plan generation failed: {e}")
        return {
            "recommendations": [],
            "priority_actions": [],
            "estimated_timeline": "unknown",
            "success_probability": 0.0
        }