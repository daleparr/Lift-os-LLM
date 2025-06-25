"""
Embedding generation and processing for LLM Finance Leaderboard.

Handles text embedding generation using various models.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ...config.settings import settings


class EmbeddingGenerator:
    """Generate embeddings for text using various models."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        model_type: str = "sentence_transformer",
        batch_size: int = 32
    ):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Name of the embedding model
            model_type: Type of model ('sentence_transformer', 'openai')
            batch_size: Batch size for processing
        """
        self.model_name = model_name or settings.default_embedding_model
        self.model_type = model_type
        self.batch_size = batch_size
        self.model = None
        
        self._initialize_model()
        
        logger.info(f"Initialized EmbeddingGenerator with {self.model_type}: {self.model_name}")
    
    def _initialize_model(self):
        """Initialize the embedding model."""
        try:
            if self.model_type == "sentence_transformer":
                if not SENTENCE_TRANSFORMERS_AVAILABLE:
                    raise ImportError("sentence-transformers not available")
                self.model = SentenceTransformer(self.model_name)
                self.dimension = self.model.get_sentence_embedding_dimension()
                
            elif self.model_type == "openai":
                if not OPENAI_AVAILABLE:
                    raise ImportError("openai not available")
                if not settings.openai_api_key:
                    raise ValueError("OpenAI API key not configured")
                openai.api_key = settings.openai_api_key
                self.dimension = 1536  # Default for text-embedding-ada-002
                
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    def generate_embeddings(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            normalize: Whether to normalize embeddings
            show_progress: Whether to show progress
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            if self.model_type == "sentence_transformer":
                return self._generate_sentence_transformer_embeddings(
                    texts, normalize, show_progress
                )
            elif self.model_type == "openai":
                return self._generate_openai_embeddings(texts, show_progress)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def _generate_sentence_transformer_embeddings(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = False
    ) -> List[List[float]]:
        """Generate embeddings using sentence-transformers."""
        start_time = time.time()
        
        # Process in batches
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            if show_progress:
                logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")
            
            batch_embeddings = self.model.encode(
                batch,
                convert_to_tensor=False,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )
            
            all_embeddings.extend(batch_embeddings.tolist())
        
        duration = time.time() - start_time
        logger.debug(f"Generated {len(all_embeddings)} embeddings in {duration:.2f}s")
        
        return all_embeddings
    
    def _generate_openai_embeddings(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        start_time = time.time()
        all_embeddings = []
        
        # OpenAI has rate limits, so process in smaller batches
        openai_batch_size = min(self.batch_size, 20)
        
        for i in range(0, len(texts), openai_batch_size):
            batch = texts[i:i + openai_batch_size]
            
            if show_progress:
                logger.info(f"Processing OpenAI batch {i//openai_batch_size + 1}/{(len(texts)-1)//openai_batch_size + 1}")
            
            try:
                response = openai.Embedding.create(
                    input=batch,
                    model=self.model_name
                )
                
                batch_embeddings = [item['embedding'] for item in response['data']]
                all_embeddings.extend(batch_embeddings)
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"OpenAI embedding error for batch {i}: {e}")
                # Add zero embeddings for failed batch
                all_embeddings.extend([[0.0] * self.dimension] * len(batch))
        
        duration = time.time() - start_time
        logger.debug(f"Generated {len(all_embeddings)} OpenAI embeddings in {duration:.2f}s")
        
        return all_embeddings
    
    def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embeddings = self.generate_embeddings([text])
        return embeddings[0] if embeddings else [0.0] * self.dimension
    
    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
        metric: str = "cosine"
    ) -> float:
        """
        Compute similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: Similarity metric ('cosine', 'dot', 'euclidean')
            
        Returns:
            Similarity score
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            if metric == "cosine":
                # Cosine similarity
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                return float(dot_product / (norm1 * norm2))
                
            elif metric == "dot":
                # Dot product
                return float(np.dot(vec1, vec2))
                
            elif metric == "euclidean":
                # Euclidean distance (converted to similarity)
                distance = np.linalg.norm(vec1 - vec2)
                return float(1.0 / (1.0 + distance))
                
            else:
                raise ValueError(f"Unsupported similarity metric: {metric}")
                
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0
    
    def find_most_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5,
        metric: str = "cosine"
    ) -> List[Tuple[int, float]]:
        """
        Find most similar embeddings to a query.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
            metric: Similarity metric
            
        Returns:
            List of (index, similarity_score) tuples
        """
        try:
            similarities = []
            
            for i, candidate in enumerate(candidate_embeddings):
                similarity = self.compute_similarity(query_embedding, candidate, metric)
                similarities.append((i, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to find most similar embeddings: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "dimension": self.dimension,
            "batch_size": self.batch_size
        }


class MockEmbeddingGenerator:
    """Mock embedding generator for testing."""
    
    def __init__(self, dimension: int = 384, *args, **kwargs):
        self.dimension = dimension
        self.model_name = "mock-embedding-model"
        self.model_type = "mock"
        logger.warning("Using MockEmbeddingGenerator")
    
    def generate_embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate mock embeddings."""
        embeddings = []
        for text in texts:
            # Generate deterministic embeddings based on text hash
            hash_val = hash(text) % 1000000
            embedding = [float((hash_val + i) % 1000) / 1000.0 for i in range(self.dimension)]
            embeddings.append(embedding)
        
        logger.debug(f"Generated {len(embeddings)} mock embeddings")
        return embeddings
    
    def generate_single_embedding(self, text: str) -> List[float]:
        return self.generate_embeddings([text])[0]
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float], metric: str = "cosine") -> float:
        # Simple mock similarity
        return 0.8 if embedding1[:5] == embedding2[:5] else 0.3
    
    def find_most_similar(self, query_embedding: List[float], candidate_embeddings: List[List[float]], top_k: int = 5, metric: str = "cosine") -> List[Tuple[int, float]]:
        # Return mock results
        results = [(i, 0.8 - i * 0.1) for i in range(min(top_k, len(candidate_embeddings)))]
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "dimension": self.dimension,
            "batch_size": 32
        }


def create_embedding_generator(
    model_name: Optional[str] = None,
    model_type: str = "sentence_transformer",
    batch_size: int = 32,
    use_mock: bool = False
) -> EmbeddingGenerator:
    """
    Factory function to create an embedding generator.
    
    Args:
        model_name: Name of the embedding model
        model_type: Type of model ('sentence_transformer', 'openai')
        batch_size: Batch size for processing
        use_mock: Whether to use mock generator
        
    Returns:
        EmbeddingGenerator instance
    """
    if use_mock:
        return MockEmbeddingGenerator()
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.warning("sentence-transformers not available, falling back to mock")
        return MockEmbeddingGenerator()
    
    return EmbeddingGenerator(
        model_name=model_name,
        model_type=model_type,
        batch_size=batch_size
    )


def precompute_document_embeddings(
    documents: List[str],
    model_name: Optional[str] = None,
    batch_size: int = 32,
    save_path: Optional[str] = None
) -> List[List[float]]:
    """
    Precompute embeddings for a list of documents.
    
    Args:
        documents: List of document texts
        model_name: Embedding model name
        batch_size: Processing batch size
        save_path: Optional path to save embeddings
        
    Returns:
        List of embedding vectors
    """
    generator = create_embedding_generator(
        model_name=model_name,
        batch_size=batch_size
    )
    
    logger.info(f"Precomputing embeddings for {len(documents)} documents")
    embeddings = generator.generate_embeddings(documents, show_progress=True)
    
    if save_path:
        try:
            import pickle
            with open(save_path, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.info(f"Saved embeddings to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
    
    return embeddings