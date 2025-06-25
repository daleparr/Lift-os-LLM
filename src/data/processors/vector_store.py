"""
Vector store integration for LLM Finance Leaderboard.

Handles document embedding, storage, and retrieval using Pinecone.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import numpy as np

try:
    import pinecone
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logger.warning("Pinecone not available. Install with: pip install pinecone-client")

from sentence_transformers import SentenceTransformer
from ..schemas.data_models import Document
from ...config.settings import settings


class PineconeVectorStore:
    """Vector store implementation using Pinecone."""
    
    def __init__(
        self,
        index_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        dimension: int = 384
    ):
        """Initialize Pinecone vector store."""
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone is required. Install with: pip install pinecone-client")
        
        self.index_name = index_name or settings.vector_index_name
        self.embedding_model_name = embedding_model or settings.default_embedding_model
        self.dimension = dimension
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Initialize or connect to index
        self.index = self._get_or_create_index()
        
        logger.info(f"Initialized PineconeVectorStore with index: {self.index_name}")
    
    def _get_or_create_index(self):
        """Get existing index or create new one."""
        try:
            # Check if index exists
            if self.index_name in self.pc.list_indexes().names():
                logger.info(f"Using existing index: {self.index_name}")
                return self.pc.Index(self.index_name)
            
            # Create new index
            logger.info(f"Creating new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            
            # Wait for index to be ready
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
            
            return self.pc.Index(self.index_name)
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone index: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def upsert_documents(self, documents: List[Document], chunk_size: int = 1000) -> bool:
        """
        Upsert documents to the vector store.
        
        Args:
            documents: List of Document objects to upsert
            chunk_size: Size of text chunks for embedding
            
        Returns:
            True if successful, False otherwise
        """
        try:
            vectors_to_upsert = []
            
            for doc in documents:
                # Chunk the document content
                chunks = self._chunk_text(doc.content, chunk_size)
                
                # Generate embeddings for chunks
                embeddings = self.generate_embeddings(chunks)
                
                # Create vectors for each chunk
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    vector_id = f"{doc.id}_chunk_{i}"
                    metadata = {
                        "document_id": doc.id,
                        "document_type": doc.document_type,
                        "title": doc.title,
                        "chunk_index": i,
                        "chunk_text": chunk[:1000],  # Limit metadata size
                        "source_url": doc.source_url or "",
                        "created_at": doc.created_at.isoformat(),
                    }
                    
                    # Add document-specific metadata
                    if hasattr(doc, 'ticker'):
                        metadata["ticker"] = doc.ticker
                    if hasattr(doc, 'company_name'):
                        metadata["company_name"] = doc.company_name
                    if hasattr(doc, 'filing_type'):
                        metadata["filing_type"] = doc.filing_type
                    
                    vectors_to_upsert.append({
                        "id": vector_id,
                        "values": embedding,
                        "metadata": metadata
                    })
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch)
                logger.debug(f"Upserted batch {i//batch_size + 1}/{(len(vectors_to_upsert)-1)//batch_size + 1}")
            
            logger.info(f"Successfully upserted {len(vectors_to_upsert)} vectors from {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert documents: {e}")
            return False
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search for a query.
        
        Args:
            query: Search query text
            k: Number of results to return
            filter_dict: Optional metadata filters
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of search results with scores and metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.generate_embeddings([query])[0]
            
            # Perform search
            search_results = self.index.query(
                vector=query_embedding,
                top_k=k,
                filter=filter_dict,
                include_metadata=include_metadata
            )
            
            # Format results
            results = []
            for match in search_results.matches:
                result = {
                    "id": match.id,
                    "score": float(match.score),
                }
                
                if include_metadata and match.metadata:
                    result["metadata"] = match.metadata
                    result["content"] = match.metadata.get("chunk_text", "")
                    result["document_id"] = match.metadata.get("document_id", "")
                    result["document_type"] = match.metadata.get("document_type", "")
                    result["title"] = match.metadata.get("title", "")
                
                results.append(result)
            
            logger.debug(f"Found {len(results)} results for query: {query[:100]}...")
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            return []
    
    def search_by_document_type(
        self,
        query: str,
        document_type: str,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for documents of a specific type."""
        filter_dict = {"document_type": document_type}
        return self.similarity_search(query, k, filter_dict)
    
    def search_by_ticker(
        self,
        query: str,
        ticker: str,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for documents related to a specific ticker."""
        filter_dict = {"ticker": ticker}
        return self.similarity_search(query, k, filter_dict)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": dict(stats.namespaces) if stats.namespaces else {}
            }
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents by their IDs."""
        try:
            # Find all vector IDs for these documents
            vector_ids = []
            for doc_id in document_ids:
                # Query to find all chunks for this document
                results = self.index.query(
                    vector=[0.0] * self.dimension,  # Dummy vector
                    filter={"document_id": doc_id},
                    top_k=10000,  # Large number to get all chunks
                    include_metadata=False
                )
                vector_ids.extend([match.id for match in results.matches])
            
            if vector_ids:
                self.index.delete(ids=vector_ids)
                logger.info(f"Deleted {len(vector_ids)} vectors for {len(document_ids)} documents")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
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
                sentence_end = text.rfind('.', end - 100, end)
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunks.append(text[start:end].strip())
            
            if end >= len(text):
                break
            
            start = end - overlap
        
        return chunks


class MockVectorStore:
    """Mock vector store for testing when Pinecone is not available."""
    
    def __init__(self, *args, **kwargs):
        self.documents = {}
        logger.warning("Using MockVectorStore - Pinecone not available")
    
    def upsert_documents(self, documents: List[Document], **kwargs) -> bool:
        for doc in documents:
            self.documents[doc.id] = doc
        logger.info(f"Mock: Stored {len(documents)} documents")
        return True
    
    def similarity_search(self, query: str, k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        # Simple keyword matching for mock
        results = []
        query_lower = query.lower()
        
        for doc_id, doc in list(self.documents.items())[:k]:
            if query_lower in doc.content.lower() or query_lower in doc.title.lower():
                results.append({
                    "id": f"{doc_id}_chunk_0",
                    "score": 0.8,
                    "content": doc.content[:500],
                    "document_id": doc_id,
                    "document_type": doc.document_type,
                    "title": doc.title,
                    "metadata": {"document_id": doc_id}
                })
        
        logger.debug(f"Mock: Found {len(results)} results for query")
        return results
    
    def search_by_document_type(self, query: str, document_type: str, k: int = 5) -> List[Dict[str, Any]]:
        return self.similarity_search(query, k)
    
    def search_by_ticker(self, query: str, ticker: str, k: int = 5) -> List[Dict[str, Any]]:
        return self.similarity_search(query, k)
    
    def get_index_stats(self) -> Dict[str, Any]:
        return {"total_vector_count": len(self.documents), "dimension": 384}
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        for doc_id in document_ids:
            self.documents.pop(doc_id, None)
        return True


def create_vector_store(
    index_name: Optional[str] = None,
    embedding_model: Optional[str] = None,
    use_mock: bool = False
) -> PineconeVectorStore:
    """Factory function to create a vector store."""
    if use_mock:
        return MockVectorStore()
    
    if not PINECONE_AVAILABLE:
        logger.warning("Pinecone not available, falling back to mock vector store")
        return MockVectorStore()
    
    return PineconeVectorStore(
        index_name=index_name,
        embedding_model=embedding_model
    )