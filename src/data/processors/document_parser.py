"""
Document processing and parsing for LLM Finance Leaderboard.

Handles document chunking, text extraction, and preprocessing.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from loguru import logger

from ..schemas.data_models import Document, DocumentType
from ...utils.helpers import chunk_text, clean_text, extract_financial_numbers


class DocumentParser:
    """Document parser for financial documents."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        """Initialize document parser."""
        self.chunk_size = chunk_size
        self.overlap = overlap
        logger.info(f"Initialized DocumentParser with chunk_size={chunk_size}, overlap={overlap}")
    
    def parse_document(self, document: Document) -> Dict[str, Any]:
        """
        Parse a document and extract structured information.
        
        Args:
            document: Document to parse
            
        Returns:
            Dictionary with parsed information
        """
        try:
            parsed_info = {
                "document_id": document.id,
                "document_type": document.document_type,
                "title": document.title,
                "chunks": [],
                "financial_numbers": [],
                "key_sections": {},
                "metadata": document.metadata,
                "processing_timestamp": datetime.utcnow().isoformat()
            }
            
            # Clean the content
            cleaned_content = clean_text(document.content)
            
            # Extract financial numbers
            financial_numbers = extract_financial_numbers(cleaned_content)
            parsed_info["financial_numbers"] = financial_numbers
            
            # Parse based on document type
            if document.document_type in [DocumentType.SEC_10Q, DocumentType.SEC_10K]:
                parsed_info["key_sections"] = self._parse_sec_filing(cleaned_content, document.document_type)
            elif document.document_type == DocumentType.EARNINGS_TRANSCRIPT:
                parsed_info["key_sections"] = self._parse_earnings_transcript(cleaned_content)
            elif document.document_type == DocumentType.NEWS_ARTICLE:
                parsed_info["key_sections"] = self._parse_news_article(cleaned_content)
            
            # Create chunks
            chunks = self.chunk_document(cleaned_content)
            parsed_info["chunks"] = [
                {
                    "index": i,
                    "content": chunk,
                    "word_count": len(chunk.split()),
                    "char_count": len(chunk)
                }
                for i, chunk in enumerate(chunks)
            ]
            
            logger.debug(f"Parsed document {document.id}: {len(chunks)} chunks, {len(financial_numbers)} financial numbers")
            return parsed_info
            
        except Exception as e:
            logger.error(f"Failed to parse document {document.id}: {e}")
            return {
                "document_id": document.id,
                "error": str(e),
                "processing_timestamp": datetime.utcnow().isoformat()
            }
    
    def chunk_document(self, content: str) -> List[str]:
        """Split document content into chunks."""
        return chunk_text(content, self.chunk_size, self.overlap)
    
    def _parse_sec_filing(self, content: str, filing_type: DocumentType) -> Dict[str, str]:
        """Parse SEC filing content to extract key sections."""
        sections = {}
        
        try:
            if filing_type == DocumentType.SEC_10Q:
                patterns = {
                    "financial_statements": r"(?i)(?:condensed\s+)?consolidated\s+(?:balance\s+sheets?|statements?\s+of\s+(?:income|operations|earnings)).*?(?=item\s+\d|$)",
                    "management_discussion": r"(?i)management'?s\s+discussion\s+and\s+analysis.*?(?=item\s+\d|$)",
                    "controls_procedures": r"(?i)(?:disclosure\s+)?controls\s+and\s+procedures.*?(?=item\s+\d|$)",
                    "legal_proceedings": r"(?i)legal\s+proceedings.*?(?=item\s+\d|$)",
                    "risk_factors": r"(?i)risk\s+factors.*?(?=item\s+\d|$)"
                }
            else:  # 10-K
                patterns = {
                    "business": r"(?i)item\s+1\.?\s*business.*?(?=item\s+\d|$)",
                    "risk_factors": r"(?i)item\s+1a\.?\s*risk\s+factors.*?(?=item\s+\d|$)",
                    "properties": r"(?i)item\s+2\.?\s*properties.*?(?=item\s+\d|$)",
                    "legal_proceedings": r"(?i)item\s+3\.?\s*legal\s+proceedings.*?(?=item\s+\d|$)",
                    "selected_financial_data": r"(?i)item\s+6\.?\s*selected\s+financial\s+data.*?(?=item\s+\d|$)",
                    "management_discussion": r"(?i)item\s+7\.?\s*management'?s\s+discussion.*?(?=item\s+\d|$)",
                    "financial_statements": r"(?i)item\s+8\.?\s*financial\s+statements.*?(?=item\s+\d|$)",
                    "controls_procedures": r"(?i)item\s+9a\.?\s*controls\s+and\s+procedures.*?(?=item\s+\d|$)"
                }
            
            # Extract sections using regex
            for section_name, pattern in patterns.items():
                match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
                if match:
                    section_content = match.group(0).strip()
                    # Limit section size
                    if len(section_content) > 10000:
                        section_content = section_content[:10000] + "... [truncated]"
                    sections[section_name] = section_content
                    logger.debug(f"Extracted {section_name} section ({len(section_content)} chars)")
            
            # Extract financial metrics
            sections["financial_metrics"] = self._extract_financial_metrics(content)
            
        except Exception as e:
            logger.warning(f"Error parsing SEC filing sections: {e}")
            sections["parsing_error"] = str(e)
        
        return sections
    
    def _parse_earnings_transcript(self, content: str) -> Dict[str, str]:
        """Parse earnings call transcript content."""
        sections = {}
        
        try:
            # Common patterns in earnings transcripts
            patterns = {
                "operator_introduction": r"(?i)operator.*?(?=\n.*?(?:chief|ceo|president))",
                "management_remarks": r"(?i)(?:chief|ceo|president|cfo).*?(?=operator|question|q&a)",
                "qa_session": r"(?i)(?:question|q&a|questions?).*?(?=operator|thank\s+you|end\s+of)",
                "forward_looking": r"(?i)forward[- ]looking\s+statements?.*?(?=\n\n|\.|$)",
                "financial_highlights": r"(?i)(?:financial\s+highlights?|key\s+metrics|results?).*?(?=\n\n|operator)"
            }
            
            for section_name, pattern in patterns.items():
                match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
                if match:
                    section_content = match.group(0).strip()
                    if len(section_content) > 5000:
                        section_content = section_content[:5000] + "... [truncated]"
                    sections[section_name] = section_content
            
            # Extract participants
            participants = re.findall(r"(?i)(?:^|\n)([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s*[-–]\s*(?:Chief|CEO|CFO|President))", content)
            sections["participants"] = list(set(participants))
            
            # Extract key financial mentions
            sections["financial_metrics"] = self._extract_financial_metrics(content)
            
        except Exception as e:
            logger.warning(f"Error parsing earnings transcript: {e}")
            sections["parsing_error"] = str(e)
        
        return sections
    
    def _parse_news_article(self, content: str) -> Dict[str, str]:
        """Parse news article content."""
        sections = {}
        
        try:
            # Split into paragraphs
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            if paragraphs:
                sections["headline"] = paragraphs[0] if len(paragraphs[0]) < 200 else paragraphs[0][:200]
                sections["lead_paragraph"] = paragraphs[1] if len(paragraphs) > 1 else ""
                sections["body"] = '\n\n'.join(paragraphs[2:]) if len(paragraphs) > 2 else ""
            
            # Extract quotes
            quotes = re.findall(r'"([^"]{20,200})"', content)
            sections["quotes"] = quotes[:5]  # Limit to 5 quotes
            
            # Extract financial metrics
            sections["financial_metrics"] = self._extract_financial_metrics(content)
            
        except Exception as e:
            logger.warning(f"Error parsing news article: {e}")
            sections["parsing_error"] = str(e)
        
        return sections
    
    def _extract_financial_metrics(self, content: str) -> Dict[str, List[str]]:
        """Extract financial metrics from content."""
        metrics = {
            "earnings_per_share": [],
            "revenue": [],
            "net_income": [],
            "ratios": [],
            "percentages": [],
            "currency_amounts": []
        }
        
        try:
            # EPS patterns
            eps_patterns = [
                r"(?i)(?:earnings?\s+per\s+share|eps).*?\$?\s*(\d+\.?\d*)",
                r"(?i)\$(\d+\.?\d*)\s+(?:per\s+share|eps)",
                r"(?i)diluted\s+eps.*?\$?\s*(\d+\.?\d*)"
            ]
            
            for pattern in eps_patterns:
                matches = re.findall(pattern, content)
                metrics["earnings_per_share"].extend(matches)
            
            # Revenue patterns
            revenue_patterns = [
                r"(?i)(?:revenue|sales|income).*?\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|M|B)?",
                r"(?i)\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|M|B)?\s+(?:in\s+)?(?:revenue|sales)"
            ]
            
            for pattern in revenue_patterns:
                matches = re.findall(pattern, content)
                metrics["revenue"].extend(matches)
            
            # Net income patterns
            income_patterns = [
                r"(?i)net\s+income.*?\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|M|B)?",
                r"(?i)\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|M|B)?\s+(?:in\s+)?net\s+income"
            ]
            
            for pattern in income_patterns:
                matches = re.findall(pattern, content)
                metrics["net_income"].extend(matches)
            
            # Ratio patterns
            ratio_patterns = [
                r"(?i)(?:tier\s+1\s+capital\s+ratio|capital\s+ratio).*?(\d+\.?\d*%?)",
                r"(?i)(?:return\s+on\s+equity|roe).*?(\d+\.?\d*%?)",
                r"(?i)(?:return\s+on\s+assets|roa).*?(\d+\.?\d*%?)"
            ]
            
            for pattern in ratio_patterns:
                matches = re.findall(pattern, content)
                metrics["ratios"].extend(matches)
            
            # General percentage patterns
            percentage_matches = re.findall(r"(\d+\.?\d*%)", content)
            metrics["percentages"] = percentage_matches[:10]  # Limit to 10
            
            # Currency amounts
            currency_matches = re.findall(r"\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|M|B)?", content)
            metrics["currency_amounts"] = currency_matches[:10]  # Limit to 10
            
        except Exception as e:
            logger.warning(f"Error extracting financial metrics: {e}")
            metrics["extraction_error"] = str(e)
        
        return metrics
    
    def extract_key_information(self, document: Document, query_context: str = "") -> Dict[str, Any]:
        """
        Extract key information relevant to a specific query context.
        
        Args:
            document: Document to extract from
            query_context: Context or query to focus extraction
            
        Returns:
            Dictionary with extracted key information
        """
        try:
            parsed_doc = self.parse_document(document)
            
            # Focus on relevant sections based on query context
            relevant_info = {
                "document_id": document.id,
                "document_type": document.document_type,
                "title": document.title,
                "relevant_chunks": [],
                "key_metrics": {},
                "context_relevance_score": 0.0
            }
            
            if query_context:
                query_lower = query_context.lower()
                
                # Score chunks by relevance to query
                for chunk in parsed_doc.get("chunks", []):
                    chunk_lower = chunk["content"].lower()
                    relevance_score = 0.0
                    
                    # Simple keyword matching
                    query_words = query_lower.split()
                    for word in query_words:
                        if word in chunk_lower:
                            relevance_score += 1.0 / len(query_words)
                    
                    if relevance_score > 0.1:  # Threshold for relevance
                        relevant_info["relevant_chunks"].append({
                            **chunk,
                            "relevance_score": relevance_score
                        })
                
                # Sort by relevance
                relevant_info["relevant_chunks"].sort(
                    key=lambda x: x["relevance_score"], 
                    reverse=True
                )
                
                # Limit to top 5 most relevant chunks
                relevant_info["relevant_chunks"] = relevant_info["relevant_chunks"][:5]
                
                # Calculate overall relevance
                if relevant_info["relevant_chunks"]:
                    relevant_info["context_relevance_score"] = sum(
                        chunk["relevance_score"] for chunk in relevant_info["relevant_chunks"]
                    ) / len(relevant_info["relevant_chunks"])
            
            # Extract key metrics
            financial_metrics = parsed_doc.get("financial_metrics", {})
            relevant_info["key_metrics"] = financial_metrics
            
            return relevant_info
            
        except Exception as e:
            logger.error(f"Failed to extract key information from document {document.id}: {e}")
            return {
                "document_id": document.id,
                "error": str(e),
                "context_relevance_score": 0.0
            }


def create_document_parser(chunk_size: int = 1000, overlap: int = 100) -> DocumentParser:
    """Factory function to create a document parser."""
    return DocumentParser(chunk_size=chunk_size, overlap=overlap)