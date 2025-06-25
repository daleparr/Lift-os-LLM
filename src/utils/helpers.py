"""
Helper utilities for LLM Finance Leaderboard.

Common utility functions used throughout the application.
"""

import re
import statistics
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import numpy as np


def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency amount for display."""
    if currency == "USD":
        if amount >= 1_000_000:
            return f"${amount/1_000_000:.1f}M"
        elif amount >= 1_000:
            return f"${amount/1_000:.1f}K"
        else:
            return f"${amount:.2f}"
    else:
        return f"{amount:.2f} {currency}"


def format_percentage(value: float, decimal_places: int = 1) -> str:
    """Format percentage for display."""
    return f"{value * 100:.{decimal_places}f}%"


def calculate_percentile(values: List[float], percentile: float) -> float:
    """Calculate percentile of a list of values."""
    if not values:
        return 0.0
    return np.percentile(values, percentile)


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a list of values."""
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "count": 0
        }
    
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "count": len(values)
    }


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    
    return text


def extract_financial_numbers(text: str) -> List[Dict[str, Any]]:
    """Extract financial numbers from text."""
    patterns = [
        # Currency amounts
        (r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(million|billion|M|B)?', 'currency'),
        # Percentages
        (r'(\d+(?:\.\d+)?)\s*%', 'percentage'),
        # Ratios
        (r'(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)', 'ratio'),
        # Basic numbers with units
        (r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(shares|basis\s+points|bps)', 'metric'),
    ]
    
    extracted = []
    
    for pattern, number_type in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            extracted.append({
                'type': number_type,
                'value': match.group(0),
                'position': match.span(),
                'groups': match.groups()
            })
    
    return extracted


def normalize_ticker(ticker: str) -> str:
    """Normalize stock ticker symbol."""
    if not ticker:
        return ""
    
    return ticker.upper().strip()


def parse_fiscal_period(period_str: str) -> Dict[str, Optional[int]]:
    """Parse fiscal period string into year and quarter."""
    if not period_str:
        return {"year": None, "quarter": None}
    
    # Common patterns for fiscal periods
    patterns = [
        r'Q(\d)\s+(\d{4})',  # Q1 2024
        r'(\d{4})\s+Q(\d)',  # 2024 Q1
        r'FY\s*(\d{4})',     # FY 2024
        r'(\d{4})',          # 2024
    ]
    
    for pattern in patterns:
        match = re.search(pattern, period_str, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                # Quarter and year pattern
                if groups[0].isdigit() and len(groups[0]) == 1:
                    return {"year": int(groups[1]), "quarter": int(groups[0])}
                else:
                    return {"year": int(groups[0]), "quarter": int(groups[1])}
            else:
                # Year only
                return {"year": int(groups[0]), "quarter": None}
    
    return {"year": None, "quarter": None}


def calculate_time_difference(start_time: datetime, end_time: datetime) -> Dict[str, float]:
    """Calculate time difference in various units."""
    if not start_time or not end_time:
        return {"seconds": 0.0, "minutes": 0.0, "hours": 0.0}
    
    diff = end_time - start_time
    total_seconds = diff.total_seconds()
    
    return {
        "seconds": total_seconds,
        "minutes": total_seconds / 60,
        "hours": total_seconds / 3600,
        "days": total_seconds / 86400
    }


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks."""
    if not text or chunk_size <= 0:
        return []
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 100 characters
            sentence_end = text.rfind('.', end - 100, end)
            if sentence_end > start:
                end = sentence_end + 1
        
        chunks.append(text[start:end].strip())
        
        if end >= len(text):
            break
        
        start = end - overlap
    
    return chunks


def validate_model_name(model_name: str) -> bool:
    """Validate model name format."""
    if not model_name:
        return False
    
    # Basic validation - should contain alphanumeric characters and common separators
    pattern = r'^[a-zA-Z0-9\-_/\.]+$'
    return bool(re.match(pattern, model_name))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length with suffix."""
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def get_gsib_bank_info(ticker: str) -> Dict[str, str]:
    """Get G-SIB bank information by ticker."""
    gsib_banks = {
        "JPM": {"name": "JPMorgan Chase & Co.", "country": "US"},
        "BAC": {"name": "Bank of America Corporation", "country": "US"},
        "WFC": {"name": "Wells Fargo & Company", "country": "US"},
        "C": {"name": "Citigroup Inc.", "country": "US"},
        "GS": {"name": "The Goldman Sachs Group, Inc.", "country": "US"},
        "MS": {"name": "Morgan Stanley", "country": "US"},
        "USB": {"name": "U.S. Bancorp", "country": "US"},
        "PNC": {"name": "The PNC Financial Services Group, Inc.", "country": "US"},
        "TD": {"name": "The Toronto-Dominion Bank", "country": "CA"},
        "RY": {"name": "Royal Bank of Canada", "country": "CA"},
        "HSBC": {"name": "HSBC Holdings plc", "country": "UK"},
        "BNP": {"name": "BNP Paribas", "country": "FR"},
        "ICBC": {"name": "Industrial and Commercial Bank of China", "country": "CN"},
    }
    
    return gsib_banks.get(ticker.upper(), {"name": "Unknown", "country": "Unknown"})


def format_model_size(param_count: str) -> str:
    """Format model parameter count for display."""
    if not param_count:
        return "Unknown"
    
    # Handle common formats
    param_count = param_count.upper().strip()
    
    if param_count.endswith('B'):
        return f"{param_count[:-1]}B parameters"
    elif param_count.endswith('M'):
        return f"{param_count[:-1]}M parameters"
    else:
        return f"{param_count} parameters"


def calculate_cost_efficiency(quality_score: float, cost_per_task: float) -> float:
    """Calculate cost efficiency metric."""
    if cost_per_task <= 0:
        return 0.0
    
    return quality_score / cost_per_task


def is_business_day(date: datetime) -> bool:
    """Check if a date is a business day (Monday-Friday)."""
    return date.weekday() < 5


def get_next_business_day(date: datetime) -> datetime:
    """Get the next business day after the given date."""
    next_day = date + timedelta(days=1)
    while not is_business_day(next_day):
        next_day += timedelta(days=1)
    return next_day


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file system usage."""
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Limit length
    if len(filename) > 255:
        filename = filename[:255]
    
    return filename