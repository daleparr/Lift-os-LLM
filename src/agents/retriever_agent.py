"""
Retriever Agent for LLM Finance Leaderboard.

Handles document retrieval from vector store based on queries.
"""

from typing import Dict, List, Any, Optional
from loguru import logger

from .base_agent import BaseAgent, AgentState
from ..data.processors.vector_store import create_vector_store
from ..data.schemas.data_models import ModelProvider
from ..config.settings import settings


class RetrieverAgent(BaseAgent):
    """Agent responsible for retrieving relevant documents from vector store."""
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        model_provider: ModelProvider = ModelProvider.OPENAI,
        vector_store=None,
        max_documents: int = 5,
        similarity_threshold: float = 0.7,
        **kwargs
    ):
        """Initialize retriever agent."""
        super().__init__(model_name, model_provider, **kwargs)
        
        self.max_documents = max_documents
        self.similarity_threshold = similarity_threshold
        
        # Initialize vector store
        self.vector_store = vector_store or create_vector_store()
        
        logger.info(f"Initialized RetrieverAgent with max_documents={max_documents}")
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for retriever agent."""
        return """You are a financial document retrieval specialist. Your role is to:

1. Analyze the user's query to understand what financial information they need
2. Identify the most relevant document types and sources
3. Generate effective search queries for the vector database
4. Filter and rank retrieved documents by relevance

Focus on:
- Understanding financial terminology and concepts
- Identifying specific metrics, ratios, or data points needed
- Considering temporal aspects (quarters, years, reporting periods)
- Prioritizing authoritative sources (SEC filings, earnings transcripts)

Always provide clear reasoning for your retrieval strategy."""
    
    def process(self, state: AgentState) -> AgentState:
        """Process retrieval for the given state."""
        try:
            query = state["original_query"]
            task_complexity = state["task_complexity"]
            
            logger.info(f"Retrieving documents for query: {query[:100]}...")
            
            # Generate search strategy
            search_strategy = self._generate_search_strategy(query, task_complexity)
            
            # Perform retrieval
            retrieved_docs = self._retrieve_documents(search_strategy)
            
            # Filter and rank documents
            filtered_docs = self._filter_and_rank_documents(retrieved_docs, query)
            
            # Update state
            state["retrieved_documents"] = filtered_docs
            state["sources_used"] = [doc.get("title", doc.get("id", "Unknown")) for doc in filtered_docs]
            
            # Add retrieval metrics to execution metrics
            state["execution_metrics"]["retrieval_count"] = len(filtered_docs)
            state["execution_metrics"]["retrieval_avg_score"] = (
                sum(doc.get("score", 0) for doc in filtered_docs) / len(filtered_docs)
                if filtered_docs else 0
            )
            
            logger.info(f"Retrieved {len(filtered_docs)} relevant documents")
            return state
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            state["error_message"] = f"Retrieval error: {str(e)}"
            state["retrieved_documents"] = []
            return state
    
    def _generate_search_strategy(self, query: str, task_complexity: str) -> Dict[str, Any]:
        """Generate search strategy based on query and complexity."""
        strategy = {
            "primary_query": query,
            "search_queries": [query],
            "document_types": [],
            "filters": {},
            "boost_factors": {}
        }
        
        query_lower = query.lower()
        
        # Identify document types based on query content
        if any(term in query_lower for term in ["eps", "earnings per share", "quarterly", "10-q"]):
            strategy["document_types"].extend(["sec_10q", "earnings_transcript"])
            strategy["search_queries"].append("earnings per share quarterly results")
        
        if any(term in query_lower for term in ["annual", "yearly", "10-k", "business overview"]):
            strategy["document_types"].append("sec_10k")
            strategy["search_queries"].append("annual report business overview")
        
        if any(term in query_lower for term in ["revenue", "sales", "income"]):
            strategy["search_queries"].extend([
                "revenue growth sales performance",
                "net income financial results"
            ])
        
        if any(term in query_lower for term in ["ratio", "capital", "tier 1", "basel"]):
            strategy["document_types"].append("sec_10q")
            strategy["search_queries"].extend([
                "capital ratios regulatory requirements",
                "tier 1 capital basel III"
            ])
        
        if any(term in query_lower for term in ["sentiment", "outlook", "guidance"]):
            strategy["document_types"].append("earnings_transcript")
            strategy["search_queries"].extend([
                "management outlook guidance",
                "forward looking statements"
            ])
        
        # Extract ticker if present
        import re
        ticker_match = re.search(r'\b([A-Z]{2,5})\b', query)
        if ticker_match:
            ticker = ticker_match.group(1)
            strategy["filters"]["ticker"] = ticker
            strategy["search_queries"].append(f"{ticker} financial performance")
        
        # Adjust strategy based on task complexity
        if task_complexity == "high":
            strategy["search_queries"].extend([
                "analysis forecast prediction",
                "risk factors challenges opportunities"
            ])
            strategy["boost_factors"]["earnings_transcript"] = 1.2
        elif task_complexity == "medium":
            strategy["search_queries"].extend([
                "trends comparison analysis",
                "performance metrics indicators"
            ])
        else:  # low complexity
            strategy["search_queries"].extend([
                "financial data metrics numbers",
                "specific figures amounts"
            ])
            strategy["boost_factors"]["sec_10q"] = 1.1
        
        return strategy
    
    def _retrieve_documents(self, strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve documents using the search strategy."""
        all_results = []
        
        try:
            # Perform searches for each query
            for search_query in strategy["search_queries"]:
                # General search
                results = self.vector_store.similarity_search(
                    query=search_query,
                    k=self.max_documents * 2,  # Get more to filter later
                    filter_dict=strategy.get("filters"),
                    include_metadata=True
                )
                
                # Add query context to results
                for result in results:
                    result["search_query"] = search_query
                    result["strategy_match"] = True
                
                all_results.extend(results)
                
                # Document type specific searches
                for doc_type in strategy.get("document_types", []):
                    type_results = self.vector_store.search_by_document_type(
                        query=search_query,
                        document_type=doc_type,
                        k=self.max_documents
                    )
                    
                    # Apply boost factors
                    boost = strategy.get("boost_factors", {}).get(doc_type, 1.0)
                    for result in type_results:
                        result["score"] = result.get("score", 0) * boost
                        result["search_query"] = search_query
                        result["document_type_match"] = doc_type
                    
                    all_results.extend(type_results)
            
            # Remove duplicates based on document ID
            seen_ids = set()
            unique_results = []
            for result in all_results:
                doc_id = result.get("document_id", result.get("id"))
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_results.append(result)
            
            return unique_results
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []
    
    def _filter_and_rank_documents(
        self,
        documents: List[Dict[str, Any]],
        original_query: str
    ) -> List[Dict[str, Any]]:
        """Filter and rank documents by relevance."""
        if not documents:
            return []
        
        # Filter by similarity threshold
        filtered_docs = [
            doc for doc in documents
            if doc.get("score", 0) >= self.similarity_threshold
        ]
        
        if not filtered_docs:
            # If no documents meet threshold, take top scoring ones
            filtered_docs = sorted(documents, key=lambda x: x.get("score", 0), reverse=True)[:self.max_documents]
        
        # Calculate additional relevance scores
        for doc in filtered_docs:
            doc["relevance_score"] = self._calculate_relevance_score(doc, original_query)
        
        # Sort by combined score
        filtered_docs.sort(
            key=lambda x: (x.get("relevance_score", 0) + x.get("score", 0)) / 2,
            reverse=True
        )
        
        # Limit to max documents
        return filtered_docs[:self.max_documents]
    
    def _calculate_relevance_score(self, document: Dict[str, Any], query: str) -> float:
        """Calculate additional relevance score for a document."""
        score = 0.0
        
        try:
            content = document.get("content", "").lower()
            title = document.get("title", "").lower()
            query_lower = query.lower()
            
            # Keyword matching in content
            query_words = query_lower.split()
            content_words = content.split()
            
            if content_words:
                keyword_matches = sum(1 for word in query_words if word in content_words)
                score += (keyword_matches / len(query_words)) * 0.4
            
            # Title relevance
            if title:
                title_matches = sum(1 for word in query_words if word in title)
                score += (title_matches / len(query_words)) * 0.3
            
            # Document type relevance
            doc_type = document.get("document_type", "")
            if "10-q" in query_lower and "10q" in doc_type.lower():
                score += 0.2
            elif "10-k" in query_lower and "10k" in doc_type.lower():
                score += 0.2
            elif "earnings" in query_lower and "transcript" in doc_type.lower():
                score += 0.2
            
            # Recency bonus (if metadata available)
            metadata = document.get("metadata", {})
            if "created_at" in metadata:
                # Simple recency bonus - more recent documents get slight boost
                score += 0.1
            
        except Exception as e:
            logger.warning(f"Error calculating relevance score: {e}")
        
        return min(score, 1.0)  # Cap at 1.0
    
    def search_specific_documents(
        self,
        query: str,
        document_type: Optional[str] = None,
        ticker: Optional[str] = None,
        max_results: int = None
    ) -> List[Dict[str, Any]]:
        """Search for specific documents with filters."""
        max_results = max_results or self.max_documents
        
        try:
            if document_type:
                results = self.vector_store.search_by_document_type(
                    query=query,
                    document_type=document_type,
                    k=max_results
                )
            elif ticker:
                results = self.vector_store.search_by_ticker(
                    query=query,
                    ticker=ticker,
                    k=max_results
                )
            else:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=max_results,
                    include_metadata=True
                )
            
            return self._filter_and_rank_documents(results, query)
            
        except Exception as e:
            logger.error(f"Specific document search failed: {e}")
            return []


def create_retriever_agent(
    model_name: str = "gpt-3.5-turbo",
    model_provider: ModelProvider = ModelProvider.OPENAI,
    vector_store=None,
    **kwargs
) -> RetrieverAgent:
    """Factory function to create a retriever agent."""
    return RetrieverAgent(
        model_name=model_name,
        model_provider=model_provider,
        vector_store=vector_store,
        **kwargs
    )