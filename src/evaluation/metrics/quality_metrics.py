"""
Quality metrics for LLM Finance Leaderboard evaluation.

Implements various quality scoring methods including F1, ROUGE, BLEU, and custom metrics.
"""

import re
import string
from typing import Dict, List, Any, Optional, Set
from collections import Counter
import statistics
from loguru import logger

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logger.warning("rouge-score not available. Install with: pip install rouge-score")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available. Install with: pip install nltk")


class QualityMetrics:
    """Quality metrics calculator for model responses."""
    
    def __init__(self):
        """Initialize quality metrics calculator."""
        self.rouge_scorer = None
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True
            )
        
        # Download NLTK data if needed
        if NLTK_AVAILABLE:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                logger.info("Downloading NLTK punkt tokenizer...")
                nltk.download('punkt', quiet=True)
        
        logger.info("Initialized QualityMetrics")
    
    def calculate_all_metrics(
        self,
        response: str,
        reference: str,
        task_type: str = "general"
    ) -> Dict[str, float]:
        """
        Calculate all available quality metrics.
        
        Args:
            response: Model response
            reference: Reference/expected answer
            task_type: Type of task for specialized metrics
            
        Returns:
            Dictionary of metric scores
        """
        metrics = {}
        
        try:
            # Basic metrics
            metrics["exact_match"] = self.exact_match_score(response, reference)
            metrics["f1_score"] = self.f1_score(response, reference)
            metrics["token_overlap"] = self.token_overlap_score(response, reference)
            
            # ROUGE metrics
            if ROUGE_AVAILABLE:
                rouge_scores = self.rouge_scores(response, reference)
                metrics.update(rouge_scores)
            
            # BLEU score
            if NLTK_AVAILABLE:
                metrics["bleu_score"] = self.bleu_score(response, reference)
            
            # Task-specific metrics
            if task_type == "numerical":
                metrics["numerical_accuracy"] = self.numerical_accuracy(response, reference)
            elif task_type == "financial":
                metrics["financial_accuracy"] = self.financial_accuracy(response, reference)
            
            # Semantic similarity (basic implementation)
            metrics["semantic_similarity"] = self.semantic_similarity(response, reference)
            
            # Quality indicators
            metrics["response_length_ratio"] = self.response_length_ratio(response, reference)
            metrics["hallucination_score"] = self.detect_hallucination(response, reference)
            
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
            # Return default scores on error
            metrics = {
                "exact_match": 0.0,
                "f1_score": 0.0,
                "token_overlap": 0.0,
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
                "bleu_score": 0.0
            }
        
        return metrics
    
    def exact_match_score(self, response: str, reference: str) -> float:
        """Calculate exact match score."""
        if not response or not reference:
            return 0.0
        
        # Normalize text
        response_norm = self._normalize_text(response)
        reference_norm = self._normalize_text(reference)
        
        return 1.0 if response_norm == reference_norm else 0.0
    
    def f1_score(self, response: str, reference: str) -> float:
        """Calculate F1 score based on token overlap."""
        if not response or not reference:
            return 0.0
        
        response_tokens = set(self._tokenize(response))
        reference_tokens = set(self._tokenize(reference))
        
        if not response_tokens and not reference_tokens:
            return 1.0
        if not response_tokens or not reference_tokens:
            return 0.0
        
        # Calculate precision and recall
        common_tokens = response_tokens.intersection(reference_tokens)
        
        precision = len(common_tokens) / len(response_tokens)
        recall = len(common_tokens) / len(reference_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def token_overlap_score(self, response: str, reference: str) -> float:
        """Calculate token overlap score."""
        if not response or not reference:
            return 0.0
        
        response_tokens = Counter(self._tokenize(response))
        reference_tokens = Counter(self._tokenize(reference))
        
        # Calculate overlap
        overlap = sum((response_tokens & reference_tokens).values())
        total = sum(reference_tokens.values())
        
        return overlap / total if total > 0 else 0.0
    
    def rouge_scores(self, response: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        if not ROUGE_AVAILABLE or not response or not reference:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        try:
            scores = self.rouge_scorer.score(reference, response)
            return {
                "rouge1": scores['rouge1'].fmeasure,
                "rouge2": scores['rouge2'].fmeasure,
                "rougeL": scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.error(f"Error calculating ROUGE scores: {e}")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    def bleu_score(self, response: str, reference: str) -> float:
        """Calculate BLEU score."""
        if not NLTK_AVAILABLE or not response or not reference:
            return 0.0
        
        try:
            response_tokens = self._tokenize(response)
            reference_tokens = [self._tokenize(reference)]  # BLEU expects list of references
            
            if not response_tokens or not reference_tokens[0]:
                return 0.0
            
            # Use smoothing to handle short sentences
            smoothing = SmoothingFunction().method1
            score = sentence_bleu(reference_tokens, response_tokens, smoothing_function=smoothing)
            return score
            
        except Exception as e:
            logger.error(f"Error calculating BLEU score: {e}")
            return 0.0
    
    def numerical_accuracy(self, response: str, reference: str) -> float:
        """Calculate accuracy for numerical responses."""
        try:
            response_nums = self._extract_numbers(response)
            reference_nums = self._extract_numbers(reference)
            
            if not response_nums and not reference_nums:
                return 1.0
            if not response_nums or not reference_nums:
                return 0.0
            
            # Compare primary numbers (first extracted)
            response_val = response_nums[0]
            reference_val = reference_nums[0]
            
            # Exact match
            if abs(response_val - reference_val) < 1e-6:
                return 1.0
            
            # Relative error
            if reference_val != 0:
                relative_error = abs(response_val - reference_val) / abs(reference_val)
                return max(0.0, 1.0 - relative_error)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating numerical accuracy: {e}")
            return 0.0
    
    def financial_accuracy(self, response: str, reference: str) -> float:
        """Calculate accuracy for financial data responses."""
        try:
            # Extract financial values
            response_values = self._extract_financial_values(response)
            reference_values = self._extract_financial_values(reference)
            
            if not response_values and not reference_values:
                return 1.0
            if not response_values or not reference_values:
                return 0.0
            
            # Compare financial values with appropriate tolerance
            scores = []
            for resp_val, ref_val in zip(response_values, reference_values):
                if abs(resp_val - ref_val) < 0.01:  # Penny tolerance
                    scores.append(1.0)
                elif ref_val != 0:
                    relative_error = abs(resp_val - ref_val) / abs(ref_val)
                    if relative_error < 0.05:  # 5% tolerance
                        scores.append(0.8)
                    elif relative_error < 0.1:  # 10% tolerance
                        scores.append(0.6)
                    else:
                        scores.append(0.0)
                else:
                    scores.append(0.0)
            
            return statistics.mean(scores) if scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating financial accuracy: {e}")
            return 0.0
    
    def semantic_similarity(self, response: str, reference: str) -> float:
        """Calculate basic semantic similarity."""
        if not response or not reference:
            return 0.0
        
        # Simple implementation based on common words and phrases
        response_words = set(self._tokenize(response.lower()))
        reference_words = set(self._tokenize(reference.lower()))
        
        # Jaccard similarity
        intersection = len(response_words.intersection(reference_words))
        union = len(response_words.union(reference_words))
        
        return intersection / union if union > 0 else 0.0
    
    def response_length_ratio(self, response: str, reference: str) -> float:
        """Calculate response length ratio (1.0 = ideal length)."""
        if not response or not reference:
            return 0.0
        
        response_len = len(self._tokenize(response))
        reference_len = len(self._tokenize(reference))
        
        if reference_len == 0:
            return 1.0 if response_len == 0 else 0.0
        
        ratio = response_len / reference_len
        
        # Penalize responses that are too short or too long
        if 0.5 <= ratio <= 2.0:
            return 1.0 - abs(1.0 - ratio) * 0.5
        else:
            return max(0.0, 1.0 - abs(1.0 - ratio))
    
    def detect_hallucination(self, response: str, reference: str) -> float:
        """Detect potential hallucination (0.0 = no hallucination, 1.0 = likely hallucination)."""
        if not response or not reference:
            return 1.0
        
        # Simple heuristics for hallucination detection
        hallucination_score = 0.0
        
        # Check for completely unrelated content
        response_words = set(self._tokenize(response.lower()))
        reference_words = set(self._tokenize(reference.lower()))
        
        overlap_ratio = len(response_words.intersection(reference_words)) / len(response_words) if response_words else 0
        
        if overlap_ratio < 0.1:  # Very low overlap
            hallucination_score += 0.5
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r"(?i)i don't know",
            r"(?i)i cannot",
            r"(?i)as an ai",
            r"(?i)i'm not sure",
            r"(?i)according to my knowledge"
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, response):
                hallucination_score += 0.3
                break
        
        return min(1.0, hallucination_score)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if not text:
            return []
        
        # Simple tokenization
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        
        # Remove empty tokens
        return [token for token in tokens if token.strip()]
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numerical values from text."""
        if not text:
            return []
        
        # Pattern for numbers (including decimals and negatives)
        pattern = r'-?\d+\.?\d*'
        matches = re.findall(pattern, text)
        
        numbers = []
        for match in matches:
            try:
                numbers.append(float(match))
            except ValueError:
                continue
        
        return numbers
    
    def _extract_financial_values(self, text: str) -> List[float]:
        """Extract financial values from text."""
        if not text:
            return []
        
        # Patterns for financial values
        patterns = [
            r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',  # Dollar amounts
            r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*%',   # Percentages
            r'(\d+\.?\d*)\s*(?:million|billion|M|B)',  # Millions/billions
        ]
        
        values = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # Clean and convert
                    cleaned = re.sub(r'[,$]', '', match)
                    values.append(float(cleaned))
                except ValueError:
                    continue
        
        return values


def calculate_quality_metrics(
    response: str,
    reference: str,
    task_type: str = "general"
) -> Dict[str, float]:
    """
    Calculate quality metrics for a response.
    
    Args:
        response: Model response
        reference: Reference answer
        task_type: Type of task for specialized metrics
        
    Returns:
        Dictionary of quality scores
    """
    metrics_calculator = QualityMetrics()
    return metrics_calculator.calculate_all_metrics(response, reference, task_type)


def aggregate_quality_scores(
    task_results: List[Dict[str, float]],
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Aggregate quality scores across multiple tasks.
    
    Args:
        task_results: List of task result dictionaries
        weights: Optional weights for different metrics
        
    Returns:
        Aggregated quality scores
    """
    if not task_results:
        return {}
    
    # Default weights
    default_weights = {
        "exact_match": 0.2,
        "f1_score": 0.3,
        "rouge1": 0.2,
        "rouge2": 0.1,
        "rougeL": 0.1,
        "bleu_score": 0.1
    }
    
    weights = weights or default_weights
    
    # Aggregate scores
    aggregated = {}
    
    # Get all available metrics
    all_metrics = set()
    for result in task_results:
        all_metrics.update(result.keys())
    
    # Calculate mean for each metric
    for metric in all_metrics:
        values = [result.get(metric, 0.0) for result in task_results]
        aggregated[metric] = statistics.mean(values)
    
    # Calculate weighted composite score
    composite_score = 0.0
    total_weight = 0.0
    
    for metric, weight in weights.items():
        if metric in aggregated:
            composite_score += aggregated[metric] * weight
            total_weight += weight
    
    if total_weight > 0:
        aggregated["composite_quality_score"] = composite_score / total_weight
    
    return aggregated