"""
LLM Finance Leaderboard - Core Package

A reproducible benchmark harness for evaluating Large Language Models 
on G-SIB financial analysis tasks.
"""

__version__ = "0.1.0"
__author__ = "Your Organization"
__email__ = "contact@yourorg.com"

from .config.settings import Settings

# Initialize global settings
settings = Settings()

__all__ = ["settings", "__version__"]