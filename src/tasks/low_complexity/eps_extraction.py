"""
EPS Extraction Task for LLM Finance Leaderboard.

Low complexity task that extracts earnings per share from financial documents.
"""

import re
from typing import Dict, List, Any, Optional
from loguru import logger

from ..base_task import FinancialExtractionTask
from ...data.schemas.data_models import TaskComplexity


class EPSExtractionTask(FinancialExtractionTask):
    """Task for extracting earnings per share from financial documents."""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="EPS Extraction",
            description="Extract earnings per share (EPS) from quarterly financial reports",
            complexity=TaskComplexity.LOW,
            timeout_minutes=5,
            **kwargs
        )
        
        # EPS extraction patterns
        self.eps_patterns = [
            r"(?i)(?:basic\s+)?(?:earnings?\s+per\s+share|eps).*?\$?\s*(\d+\.?\d*)",
            r"(?i)diluted\s+(?:earnings?\s+per\s+share|eps).*?\$?\s*(\d+\.?\d*)",
            r"(?i)\$(\d+\.?\d*)\s+per\s+(?:diluted\s+)?share",
            r"(?i)(?:earnings?\s+per\s+share|eps).*?of\s+\$?\s*(\d+\.?\d*)",
            r"(?i)(?:basic|diluted)\s+eps.*?\$?\s*(\d+\.?\d*)",
            r"(?i)per\s+share\s+earnings?.*?\$?\s*(\d+\.?\d*)"
        ]
    
    def generate_prompt(self, context: Dict[str, Any]) -> str:
        """Generate prompt for EPS extraction."""
        documents = context.get("documents", [])
        
        # Combine relevant document content
        document_content = ""
        for doc in documents[:3]:  # Limit to top 3 documents
            content = doc.get("content", "")
            title = doc.get("title", "")
            document_content += f"\n--- {title} ---\n{content[:2000]}\n"
        
        prompt = f"""You are a financial analyst tasked with extracting earnings per share (EPS) information from financial documents.

TASK: Extract the earnings per share (EPS) value from the provided financial documents.

DOCUMENTS:
{document_content}

INSTRUCTIONS:
1. Look for earnings per share (EPS) values in the documents
2. Focus on the most recent quarterly or annual EPS figure
3. Prefer diluted EPS over basic EPS if both are available
4. Extract the numerical value only (e.g., "3.12" not "$3.12 per share")
5. If multiple EPS values are found, choose the most recent or most prominently featured one

RESPONSE FORMAT:
Provide only the numerical EPS value as a decimal number (e.g., 3.12, 0.85, -1.23).
If no EPS value can be found, respond with "N/A".

EPS VALUE:"""
        
        return prompt
    
    def get_expected_answer(self, context: Dict[str, Any]) -> str:
        """Get expected EPS answer from context or documents."""
        # In a real implementation, this would come from ground truth data
        # For now, we'll try to extract from the documents as a fallback
        
        expected_eps = context.get("expected_eps")
        if expected_eps is not None:
            return str(expected_eps)
        
        # Try to extract from documents
        documents = context.get("documents", [])
        for doc in documents:
            content = doc.get("content", "")
            extracted_eps = self.extract_financial_value(content, self.eps_patterns)
            if extracted_eps:
                # Clean and normalize
                normalized = self.normalize_eps_value(extracted_eps)
                if normalized is not None:
                    return str(normalized)
        
        return "N/A"
    
    def validate_response(self, response: str, expected_answer: str) -> Dict[str, float]:
        """Validate EPS extraction response."""
        scores = {
            "exact_match": 0.0,
            "f1_score": 0.0,
            "numerical_accuracy": 0.0,
            "format_correctness": 0.0
        }
        
        try:
            # Clean response
            response_clean = response.strip()
            expected_clean = expected_answer.strip()
            
            # Check format correctness
            if self._is_valid_eps_format(response_clean):
                scores["format_correctness"] = 1.0
            
            # Handle N/A cases
            if response_clean.upper() == "N/A" and expected_clean.upper() == "N/A":
                scores["exact_match"] = 1.0
                scores["f1_score"] = 1.0
                scores["numerical_accuracy"] = 1.0
                return scores
            elif response_clean.upper() == "N/A" or expected_clean.upper() == "N/A":
                return scores
            
            # Parse numerical values
            response_value = self.parse_eps_value(response_clean)
            expected_value = self.parse_eps_value(expected_clean)
            
            if response_value is not None and expected_value is not None:
                # Exact match (with small tolerance for floating point)
                if abs(response_value - expected_value) < 0.01:
                    scores["exact_match"] = 1.0
                    scores["f1_score"] = 1.0
                    scores["numerical_accuracy"] = 1.0
                else:
                    # Partial credit based on numerical closeness
                    diff = abs(response_value - expected_value)
                    max_expected = max(abs(expected_value), 1.0)
                    
                    # Accuracy score based on relative error
                    relative_error = diff / max_expected
                    scores["numerical_accuracy"] = max(0.0, 1.0 - relative_error)
                    
                    # F1 score for partial matches
                    if relative_error < 0.1:  # Within 10%
                        scores["f1_score"] = 0.8
                    elif relative_error < 0.2:  # Within 20%
                        scores["f1_score"] = 0.6
                    elif relative_error < 0.5:  # Within 50%
                        scores["f1_score"] = 0.3
            
            logger.debug(f"EPS validation - Response: {response_clean}, Expected: {expected_clean}, Scores: {scores}")
            
        except Exception as e:
            logger.error(f"Error validating EPS response: {e}")
        
        return scores
    
    def normalize_eps_value(self, value: str) -> Optional[float]:
        """Normalize EPS value to standard format."""
        if not value:
            return None
        
        try:
            # Remove common formatting
            cleaned = re.sub(r'[$,\s]', '', value)
            
            # Handle negative values
            is_negative = cleaned.startswith('-') or cleaned.startswith('(')
            cleaned = re.sub(r'[()-]', '', cleaned)
            
            # Convert to float
            result = float(cleaned)
            
            return -result if is_negative else result
            
        except (ValueError, TypeError):
            return None
    
    def parse_eps_value(self, value: str) -> Optional[float]:
        """Parse EPS value from string."""
        if not value or value.upper() == "N/A":
            return None
        
        # Try direct float conversion first
        try:
            return float(value)
        except ValueError:
            pass
        
        # Try extracting number from text
        number_match = re.search(r'-?\d+\.?\d*', value)
        if number_match:
            try:
                return float(number_match.group(0))
            except ValueError:
                pass
        
        return None
    
    def _is_valid_eps_format(self, value: str) -> bool:
        """Check if the response is in valid EPS format."""
        if not value:
            return False
        
        value_upper = value.upper().strip()
        
        # Valid formats: number, N/A, or simple text with number
        if value_upper == "N/A":
            return True
        
        # Check if it contains a valid number
        if re.search(r'-?\d+\.?\d*', value):
            return True
        
        return False
    
    def extract_eps_from_documents(self, documents: List[Dict[str, Any]]) -> Optional[str]:
        """Extract EPS value from documents using patterns."""
        for doc in documents:
            content = doc.get("content", "")
            
            # Try each pattern
            for pattern in self.eps_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    eps_value = match.group(1) if match.groups() else match.group(0)
                    normalized = self.normalize_eps_value(eps_value)
                    if normalized is not None:
                        return str(normalized)
        
        return None
    
    def prepare_context(self, documents: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Prepare context for EPS extraction task."""
        context = super().prepare_context(documents, **kwargs)
        
        # Add EPS-specific context
        context.update({
            "extraction_patterns": self.eps_patterns,
            "target_metric": "earnings_per_share",
            "expected_format": "numerical_value"
        })
        
        return context


def create_eps_extraction_task(**kwargs) -> EPSExtractionTask:
    """Factory function to create EPS extraction task."""
    return EPSExtractionTask(**kwargs)


# Sample test cases for EPS extraction
SAMPLE_EPS_TEST_CASES = [
    {
        "document_content": "For the quarter ended March 31, 2024, diluted earnings per share was $3.12, compared to $2.85 in the prior year quarter.",
        "expected_eps": "3.12",
        "description": "Standard quarterly EPS reporting"
    },
    {
        "document_content": "Basic EPS of $1.45 and diluted EPS of $1.42 for the fiscal year.",
        "expected_eps": "1.42",
        "description": "Both basic and diluted EPS (should prefer diluted)"
    },
    {
        "document_content": "The company reported a loss of $0.23 per share for the quarter.",
        "expected_eps": "-0.23",
        "description": "Negative EPS (loss)"
    },
    {
        "document_content": "Revenue increased 15% year-over-year to $2.1 billion. Operating margin improved to 18.5%.",
        "expected_eps": "N/A",
        "description": "No EPS information available"
    }
]