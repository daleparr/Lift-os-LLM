"""
Base agent implementation using LangGraph for LLM Finance Leaderboard.

This module provides the foundation for the multi-agent pipeline using LangGraph's
state management and workflow orchestration capabilities.
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime
import time
from abc import ABC, abstractmethod

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseLLM
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langgraph.checkpoint.sqlite import SqliteSaver
from loguru import logger

from ..config.settings import settings
from ..data.schemas.data_models import AgentResponse, ModelProvider


class AgentState(TypedDict):
    """State shared across all agents in the pipeline."""
    messages: Annotated[List[BaseMessage], "The conversation messages"]
    task_id: str
    task_complexity: str
    original_query: str
    retrieved_documents: List[Dict[str, Any]]
    parsed_information: Dict[str, Any]
    analysis_results: Dict[str, Any]
    draft_response: str
    final_response: str
    confidence_score: float
    sources_used: List[str]
    tool_calls: List[Dict[str, Any]]
    execution_metrics: Dict[str, float]
    error_message: Optional[str]


class BaseAgent(ABC):
    """Base class for all agents in the pipeline."""
    
    def __init__(
        self,
        model_name: str,
        model_provider: ModelProvider = ModelProvider.OPENAI,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs
    ):
        """Initialize the base agent."""
        self.model_name = model_name
        self.model_provider = model_provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize the language model
        self.llm = self._create_llm()
        
        # Agent-specific configuration
        self.agent_name = self.__class__.__name__
        self.system_prompt = self._get_system_prompt()
        
        logger.info(f"Initialized {self.agent_name} with model {model_name}")
    
    def _create_llm(self) -> BaseLLM:
        """Create the language model instance."""
        if self.model_provider == ModelProvider.OPENAI:
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=settings.openai_api_key
            )
        elif self.model_provider == ModelProvider.ANTHROPIC:
            return ChatAnthropic(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=settings.anthropic_api_key
            )
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        pass
    
    @abstractmethod
    def process(self, state: AgentState) -> AgentState:
        """Process the current state and return updated state."""
        pass
    
    def _invoke_llm(
        self,
        messages: List[BaseMessage],
        **kwargs
    ) -> AIMessage:
        """Invoke the language model with error handling and metrics."""
        start_time = time.time()
        
        try:
            # Add system message if not present
            if not any(isinstance(msg, SystemMessage) for msg in messages):
                messages = [SystemMessage(content=self.system_prompt)] + messages
            
            response = self.llm.invoke(messages, **kwargs)
            
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            
            logger.debug(f"{self.agent_name} execution time: {execution_time:.2f}ms")
            
            return response
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"{self.agent_name} failed after {execution_time:.2f}ms: {e}")
            raise
    
    def _create_agent_response(
        self,
        response_text: str,
        confidence_score: float = 0.8,
        reasoning: str = "",
        sources_used: List[str] = None,
        tool_calls: List[Dict[str, Any]] = None,
        execution_time_ms: float = 0.0,
        token_usage: Dict[str, int] = None
    ) -> AgentResponse:
        """Create a standardized agent response."""
        return AgentResponse(
            agent_name=self.agent_name,
            response_text=response_text,
            confidence_score=confidence_score,
            reasoning=reasoning,
            sources_used=sources_used or [],
            tool_calls=tool_calls or [],
            execution_time_ms=execution_time_ms,
            token_usage=token_usage or {},
            created_at=datetime.utcnow()
        )


class FinancialAgentPipeline:
    """LangGraph-based pipeline for financial analysis agents."""
    
    def __init__(
        self,
        model_name: str,
        model_provider: ModelProvider = ModelProvider.OPENAI,
        checkpoint_path: Optional[str] = None
    ):
        """Initialize the agent pipeline."""
        self.model_name = model_name
        self.model_provider = model_provider
        
        # Initialize agents
        self.retriever_agent = None  # Will be implemented in retriever_agent.py
        self.parser_agent = None     # Will be implemented in parser_agent.py
        self.analysis_agent = None   # Will be implemented in analysis_agent.py
        self.draft_agent = None      # Will be implemented in draft_agent.py
        self.critic_agent = None     # Will be implemented in critic_agent.py
        
        # Initialize checkpointer for state persistence
        self.checkpointer = None
        if checkpoint_path:
            self.checkpointer = SqliteSaver.from_conn_string(checkpoint_path)
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
        logger.info(f"Initialized FinancialAgentPipeline with {model_name}")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each agent
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("parse", self._parse_node)
        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("draft", self._draft_node)
        workflow.add_node("critique", self._critique_node)
        
        # Define the workflow edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "parse")
        workflow.add_edge("parse", "analyze")
        workflow.add_edge("analyze", "draft")
        workflow.add_edge("draft", "critique")
        workflow.add_edge("critique", END)
        
        return workflow
    
    def _retrieve_node(self, state: AgentState) -> AgentState:
        """Retrieval node - retrieve relevant documents for the query."""
        logger.info("Executing retrieval step")
        
        try:
            query = state.get("query", "")
            if not query:
                logger.warning("No query provided for retrieval")
                state["retrieved_documents"] = []
                state["sources_used"] = []
                return state
            
            # Use vector store for retrieval if available
            if hasattr(self, 'vector_store') and self.vector_store:
                results = self.vector_store.similarity_search(query, k=5)
                state["retrieved_documents"] = results
                state["sources_used"] = [doc.get("source", "Unknown") for doc in results]
            else:
                # Fallback to context documents if provided
                context_docs = state.get("context_documents", [])
                if context_docs:
                    state["retrieved_documents"] = context_docs[:3]  # Take first 3
                    state["sources_used"] = [doc.get("source", "Context Document") for doc in context_docs]
                else:
                    logger.warning("No vector store or context documents available")
                    state["retrieved_documents"] = []
                    state["sources_used"] = []
            
            logger.info(f"Retrieved {len(state['retrieved_documents'])} documents")
            return state
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            state["retrieved_documents"] = []
            state["sources_used"] = []
            return state
    
    def _parse_node(self, state: AgentState) -> AgentState:
        """Parsing node - extract structured data from retrieved documents."""
        logger.info("Executing parsing step")
        
        try:
            retrieved_docs = state.get("retrieved_documents", [])
            task_type = state.get("task_type", "")
            
            parsed_information = {
                "financial_metrics": {},
                "key_insights": []
            }
            
            if not retrieved_docs:
                logger.warning("No documents to parse")
                state["parsed_information"] = parsed_information
                return state
            
            # Parse documents for financial information
            for doc in retrieved_docs:
                content = doc.get("content", "")
                
                # Extract financial metrics using regex
                import re
                
                # EPS extraction
                eps_patterns = [
                    r'earnings per share.*?\$?(\d+\.?\d*)',
                    r'EPS.*?\$?(\d+\.?\d*)',
                    r'\$(\d+\.?\d*)\s*per share'
                ]
                
                for pattern in eps_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        parsed_information["financial_metrics"]["EPS"] = float(matches[0])
                        break
                
                # Revenue extraction
                revenue_match = re.search(r'revenue.*?\$?(\d+\.?\d*)\s*([BMK]?)', content, re.IGNORECASE)
                if revenue_match:
                    value = float(revenue_match.group(1))
                    unit = revenue_match.group(2).upper()
                    if unit == 'B':
                        value *= 1e9
                    elif unit == 'M':
                        value *= 1e6
                    elif unit == 'K':
                        value *= 1e3
                    parsed_information["financial_metrics"]["Revenue"] = value
                
                # Extract key insights from positive/negative language
                positive_indicators = ["strong", "robust", "exceeded", "growth", "improved"]
                negative_indicators = ["weak", "declined", "disappointed", "challenging", "decreased"]
                
                content_lower = content.lower()
                insights = []
                
                for indicator in positive_indicators:
                    if indicator in content_lower:
                        insights.append(f"Positive indicator: {indicator}")
                
                for indicator in negative_indicators:
                    if indicator in content_lower:
                        insights.append(f"Risk factor: {indicator}")
                
                parsed_information["key_insights"].extend(insights)
            
            # Remove duplicates from insights
            parsed_information["key_insights"] = list(set(parsed_information["key_insights"]))
            
            state["parsed_information"] = parsed_information
            logger.info(f"Parsed {len(parsed_information['financial_metrics'])} metrics and {len(parsed_information['key_insights'])} insights")
            return state
            
        except Exception as e:
            logger.error(f"Parsing failed: {e}")
            state["parsed_information"] = {"financial_metrics": {}, "key_insights": []}
            return state
    
    def _analyze_node(self, state: AgentState) -> AgentState:
        """Analysis node - analyze parsed financial data."""
        logger.info("Executing analysis step")
        
        try:
            parsed_info = state.get("parsed_information", {})
            financial_metrics = parsed_info.get("financial_metrics", {})
            key_insights = parsed_info.get("key_insights", [])
            
            analysis_results = {
                "trend_analysis": "",
                "risk_factors": [],
                "opportunities": []
            }
            
            # Analyze financial metrics
            if "EPS" in financial_metrics:
                eps = financial_metrics["EPS"]
                if eps > 3.0:
                    analysis_results["trend_analysis"] = "Strong earnings performance with high EPS"
                    analysis_results["opportunities"].append("Strong profitability indicates growth potential")
                elif eps > 1.0:
                    analysis_results["trend_analysis"] = "Moderate earnings performance"
                else:
                    analysis_results["trend_analysis"] = "Weak earnings performance"
                    analysis_results["risk_factors"].append("Low EPS indicates profitability concerns")
            
            if "Revenue" in financial_metrics:
                revenue = financial_metrics["Revenue"]
                if revenue > 20e9:  # > $20B
                    analysis_results["opportunities"].append("Large revenue base provides market stability")
                elif revenue < 1e9:  # < $1B
                    analysis_results["risk_factors"].append("Small revenue base may indicate limited market presence")
            
            # Analyze insights for sentiment
            positive_count = sum(1 for insight in key_insights if "positive" in insight.lower())
            risk_count = sum(1 for insight in key_insights if "risk" in insight.lower())
            
            if positive_count > risk_count:
                if not analysis_results["trend_analysis"]:
                    analysis_results["trend_analysis"] = "Overall positive market sentiment"
                analysis_results["opportunities"].append("Positive market indicators suggest growth potential")
            elif risk_count > positive_count:
                analysis_results["risk_factors"].append("Market sentiment indicates potential challenges")
            
            # Add default analysis if none found
            if not analysis_results["trend_analysis"]:
                analysis_results["trend_analysis"] = "Insufficient data for comprehensive trend analysis"
            
            if not analysis_results["risk_factors"]:
                analysis_results["risk_factors"] = ["Standard market risks apply"]
            
            if not analysis_results["opportunities"]:
                analysis_results["opportunities"] = ["Further analysis needed to identify opportunities"]
            
            state["analysis_results"] = analysis_results
            logger.info("Analysis completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            state["analysis_results"] = {
                "trend_analysis": "Analysis failed due to data processing error",
                "risk_factors": ["Data quality issues"],
                "opportunities": ["Improve data collection and processing"]
            }
            return state
    
    def _draft_node(self, state: AgentState) -> AgentState:
        """Draft generation node - create response based on analysis."""
        logger.info("Executing draft step")
        
        try:
            analysis_results = state.get("analysis_results", {})
            parsed_info = state.get("parsed_information", {})
            financial_metrics = parsed_info.get("financial_metrics", {})
            
            # Build response based on available data
            response_parts = []
            
            # Add financial metrics summary
            if financial_metrics:
                metrics_summary = "Key financial metrics: "
                metric_strings = []
                
                if "EPS" in financial_metrics:
                    metric_strings.append(f"EPS of ${financial_metrics['EPS']:.2f}")
                
                if "Revenue" in financial_metrics:
                    revenue = financial_metrics["Revenue"]
                    if revenue >= 1e9:
                        metric_strings.append(f"Revenue of ${revenue/1e9:.1f}B")
                    elif revenue >= 1e6:
                        metric_strings.append(f"Revenue of ${revenue/1e6:.1f}M")
                    else:
                        metric_strings.append(f"Revenue of ${revenue:,.0f}")
                
                if metric_strings:
                    metrics_summary += ", ".join(metric_strings) + "."
                    response_parts.append(metrics_summary)
            
            # Add trend analysis
            trend_analysis = analysis_results.get("trend_analysis", "")
            if trend_analysis:
                response_parts.append(f"Analysis indicates: {trend_analysis}")
            
            # Add opportunities
            opportunities = analysis_results.get("opportunities", [])
            if opportunities:
                response_parts.append(f"Key opportunities: {'; '.join(opportunities[:2])}")
            
            # Add risk factors
            risk_factors = analysis_results.get("risk_factors", [])
            if risk_factors:
                response_parts.append(f"Risk considerations: {'; '.join(risk_factors[:2])}")
            
            # Combine response
            if response_parts:
                draft_response = " ".join(response_parts)
            else:
                draft_response = "Based on the available financial information, further analysis is needed to provide comprehensive insights."
            
            state["draft_response"] = draft_response
            logger.info("Draft response generated successfully")
            return state
            
        except Exception as e:
            logger.error(f"Draft generation failed: {e}")
            state["draft_response"] = "Unable to generate response due to processing error."
            return state
    
    def _critique_node(self, state: AgentState) -> AgentState:
        """Critique and refinement node - review and improve draft response."""
        logger.info("Executing critique step")
        
        try:
            draft_response = state.get("draft_response", "")
            parsed_info = state.get("parsed_information", {})
            analysis_results = state.get("analysis_results", {})
            
            # Calculate confidence score based on data quality
            confidence_factors = []
            
            # Check data availability
            financial_metrics = parsed_info.get("financial_metrics", {})
            if financial_metrics:
                confidence_factors.append(0.3)  # Base confidence for having metrics
                
                # Additional confidence for specific metrics
                if "EPS" in financial_metrics:
                    confidence_factors.append(0.2)
                if "Revenue" in financial_metrics:
                    confidence_factors.append(0.2)
            
            # Check analysis quality
            if analysis_results.get("trend_analysis"):
                confidence_factors.append(0.2)
            
            # Check response completeness
            if len(draft_response) > 100:
                confidence_factors.append(0.1)
            
            confidence_score = min(0.95, sum(confidence_factors))
            
            # Improve response based on critique
            final_response = draft_response
            
            # Add confidence qualifier if low confidence
            if confidence_score < 0.5:
                final_response = f"Based on limited available data: {final_response}"
            elif confidence_score < 0.7:
                final_response = f"Preliminary analysis suggests: {final_response}"
            
            # Ensure response has proper conclusion
            if not final_response.endswith('.'):
                final_response += "."
            
            # Add data source acknowledgment
            sources_used = state.get("sources_used", [])
            if sources_used:
                unique_sources = list(set(sources_used))
                if len(unique_sources) == 1:
                    final_response += f" Analysis based on {unique_sources[0]}."
                else:
                    final_response += f" Analysis based on {len(unique_sources)} sources."
            
            state["final_response"] = final_response
            state["confidence_score"] = confidence_score
            
            logger.info(f"Critique completed with confidence score: {confidence_score:.2f}")
            return state
            
        except Exception as e:
            logger.error(f"Critique failed: {e}")
            state["final_response"] = state.get("draft_response", "Analysis could not be completed.")
            state["confidence_score"] = 0.1
            return state
    
    def run(
        self,
        query: str,
        task_id: str,
        task_complexity: str,
        config: Optional[Dict[str, Any]] = None
    ) -> AgentState:
        """Run the complete agent pipeline."""
        logger.info(f"Starting pipeline for task {task_id}")
        
        # Initialize state
        initial_state = AgentState(
            messages=[HumanMessage(content=query)],
            task_id=task_id,
            task_complexity=task_complexity,
            original_query=query,
            retrieved_documents=[],
            parsed_information={},
            analysis_results={},
            draft_response="",
            final_response="",
            confidence_score=0.0,
            sources_used=[],
            tool_calls=[],
            execution_metrics={},
            error_message=None
        )
        
        try:
            # Compile and run the workflow
            app = self.workflow.compile(checkpointer=self.checkpointer)
            
            # Execute the pipeline
            final_state = app.invoke(
                initial_state,
                config=config or {"configurable": {"thread_id": task_id}}
            )
            
            logger.info(f"Pipeline completed for task {task_id}")
            return final_state
            
        except Exception as e:
            logger.error(f"Pipeline failed for task {task_id}: {e}")
            initial_state["error_message"] = str(e)
            return initial_state
    
    async def arun(
        self,
        query: str,
        task_id: str,
        task_complexity: str,
        config: Optional[Dict[str, Any]] = None
    ) -> AgentState:
        """Async version of the pipeline run."""
        logger.info(f"Starting async pipeline for task {task_id}")
        
        # Initialize state (same as sync version)
        initial_state = AgentState(
            messages=[HumanMessage(content=query)],
            task_id=task_id,
            task_complexity=task_complexity,
            original_query=query,
            retrieved_documents=[],
            parsed_information={},
            analysis_results={},
            draft_response="",
            final_response="",
            confidence_score=0.0,
            sources_used=[],
            tool_calls=[],
            execution_metrics={},
            error_message=None
        )
        
        try:
            # Compile and run the workflow asynchronously
            app = self.workflow.compile(checkpointer=self.checkpointer)
            
            # Execute the pipeline
            final_state = await app.ainvoke(
                initial_state,
                config=config or {"configurable": {"thread_id": task_id}}
            )
            
            logger.info(f"Async pipeline completed for task {task_id}")
            return final_state
            
        except Exception as e:
            logger.error(f"Async pipeline failed for task {task_id}: {e}")
            initial_state["error_message"] = str(e)
            return initial_state


def create_agent_pipeline(
    model_name: str,
    model_provider: ModelProvider = ModelProvider.OPENAI,
    checkpoint_path: Optional[str] = None
) -> FinancialAgentPipeline:
    """Factory function to create an agent pipeline."""
    return FinancialAgentPipeline(
        model_name=model_name,
        model_provider=model_provider,
        checkpoint_path=checkpoint_path
    )