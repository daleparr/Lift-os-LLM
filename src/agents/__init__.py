"""
LangGraph-based agent pipeline for LLM Finance Leaderboard.

This module implements a multi-agent workflow using LangGraph for financial analysis tasks.
The pipeline consists of specialized agents that work together to process financial documents
and generate comprehensive analysis.
"""

from .base_agent import (
    BaseAgent,
    AgentState,
    FinancialAgentPipeline,
    create_agent_pipeline
)

__all__ = [
    "BaseAgent",
    "AgentState", 
    "FinancialAgentPipeline",
    "create_agent_pipeline",
]