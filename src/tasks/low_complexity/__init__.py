"""
Low complexity tasks for LLM Finance Leaderboard.
"""

from .eps_extraction import EPSExtractionTask, create_eps_extraction_task

__all__ = [
    "EPSExtractionTask",
    "create_eps_extraction_task"
]