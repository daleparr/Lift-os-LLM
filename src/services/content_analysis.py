"""
Content Analysis Service for Lift-os-LLM microservice.

Provides comprehensive content analysis including AI surfacing scores,
semantic analysis, and structured data evaluation.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import re
from urllib.parse import urlparse
import aiohttp
from bs4 import BeautifulSoup
import openai
import anthropic

from ..core.config import settings
from ..core.logging import logger, log_analysis_request
from ..core.database import cache_manager
from ..models.entities import (
    ContentInput, AISurfacingScore, Recommendation, StructuredDataAnalysis,
    SemanticAnalysis, VectorEmbeddings, KnowledgeGraph, ContentAnalysisResult,
    AnalysisType
)


class ContentAnalysisService:
    """Service for analyzing content and generating AI surfacing scores."""
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize LLM clients based on available API keys."""
        if settings.has_openai:
            openai.api_key = settings.OPENAI_API_KEY
            self.openai_client = openai
            logger.info("OpenAI client initialized")
        
        if settings.has_anthropic:
            self.anthropic_client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            logger.info("Anthropic client initialized")
    
    async def analyze_content(
        self,
        content: ContentInput,
        analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE,
        include_embeddings: bool = True,
        include_knowledge_graph: bool = True,
        **options
    ) -> ContentAnalysisResult:
        """
        Perform comprehensive content analysis.
        
        Args:
            content: Input content to analyze
            analysis_type: Type of analysis to perform
            include_embeddings: Whether to include vector embeddings
            include_knowledge_graph: Whether to include knowledge graph analysis
            **options: Additional analysis options
        
        Returns:
            ContentAnalysisResult with complete analysis
        """
        start_time = time.time()
        
        try:
            # Extract content from URL if provided
            if content.url and not content.html:
                content.html = await self._fetch_content_from_url(content.url)
            
            # Parse and extract content
            extracted_content = await self._extract_content(content)
            
            # Perform parallel analysis tasks
            tasks = []
            
            # Core AI surfacing score
            tasks.append(self._calculate_ai_surfacing_score(extracted_content))
            
            # Structured data analysis
            tasks.append(self._analyze_structured_data(content.html or ""))
            
            # Semantic analysis
            if analysis_type in [AnalysisType.COMPREHENSIVE, AnalysisType.SEMANTIC]:
                tasks.append(self._perform_semantic_analysis(extracted_content))
            
            # Vector embeddings
            if include_embeddings:
                tasks.append(self._generate_embeddings(extracted_content))
            
            # Knowledge graph
            if include_knowledge_graph:
                tasks.append(self._extract_knowledge_graph(extracted_content))
            
            # Execute all tasks in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            ai_surfacing_score = results[0] if not isinstance(results[0], Exception) else self._default_ai_score()
            structured_data = results[1] if not isinstance(results[1], Exception) else None
            semantic_analysis = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else None
            vector_embeddings = results[3] if include_embeddings and len(results) > 3 and not isinstance(results[3], Exception) else None
            knowledge_graph = results[4] if include_knowledge_graph and len(results) > 4 and not isinstance(results[4], Exception) else None
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                ai_surfacing_score, structured_data, semantic_analysis
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Create analysis result
            result = ContentAnalysisResult(
                url=content.url,
                ai_surfacing_score=ai_surfacing_score,
                recommendations=recommendations,
                content_extraction=extracted_content,
                structured_data=structured_data,
                semantic_analysis=semantic_analysis,
                vector_embeddings=vector_embeddings,
                knowledge_graph=knowledge_graph,
                processing_time_ms=processing_time,
                model_used=options.get('model_override', settings.DEFAULT_LLM_MODEL),
                analysis_type=analysis_type
            )
            
            # Log analysis request
            log_analysis_request(
                analysis_type=analysis_type.value,
                content_length=len(str(extracted_content)),
                processing_time_ms=processing_time,
                ai_surfacing_score=ai_surfacing_score.overall,
                request_id=options.get('request_id')
            )
            
            # Cache result if URL provided
            if content.url:
                cache_key = f"analysis:{hash(content.url)}"
                await cache_manager.set(cache_key, result.json(), ttl=3600)
            
            return result
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}", exc_info=True)
            raise
    
    async def _fetch_content_from_url(self, url: str) -> str:
        """Fetch HTML content from URL."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        content = await response.text()
                        logger.debug(f"Fetched content from URL: {url}")
                        return content
                    else:
                        logger.warning(f"Failed to fetch URL {url}: HTTP {response.status}")
                        return ""
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return ""
    
    async def _extract_content(self, content: ContentInput) -> Dict[str, Any]:
        """Extract and structure content from input."""
        extracted = {
            "title": content.title or "",
            "description": content.description or "",
            "url": content.url,
            "metadata": content.metadata
        }
        
        if content.html:
            soup = BeautifulSoup(content.html, 'html.parser')
            
            # Extract title if not provided
            if not extracted["title"]:
                title_tag = soup.find('title')
                if title_tag:
                    extracted["title"] = title_tag.get_text().strip()
            
            # Extract meta description if not provided
            if not extracted["description"]:
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if meta_desc:
                    extracted["description"] = meta_desc.get('content', '').strip()
            
            # Extract main content
            extracted["content"] = self._extract_main_content(soup)
            
            # Extract structured data
            extracted["structured_data"] = self._extract_structured_data(soup)
            
            # Extract images
            extracted["images"] = [img.get('src') for img in soup.find_all('img') if img.get('src')]
            
            # Extract links
            extracted["links"] = [a.get('href') for a in soup.find_all('a') if a.get('href')]
        
        return extracted
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from HTML."""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Try to find main content areas
        main_selectors = [
            'main', 'article', '.content', '#content', '.main', '#main',
            '.post-content', '.entry-content', '.article-content'
        ]
        
        for selector in main_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                return main_content.get_text(strip=True)
        
        # Fallback to body content
        body = soup.find('body')
        if body:
            return body.get_text(strip=True)
        
        return soup.get_text(strip=True)
    
    def _extract_structured_data(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract structured data from HTML."""
        structured_data = []
        
        # JSON-LD
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                structured_data.append({"type": "json-ld", "data": data})
            except json.JSONDecodeError:
                continue
        
        # Microdata
        microdata_items = soup.find_all(attrs={"itemscope": True})
        for item in microdata_items:
            item_type = item.get('itemtype', '')
            properties = {}
            for prop in item.find_all(attrs={"itemprop": True}):
                prop_name = prop.get('itemprop')
                prop_value = prop.get('content') or prop.get_text(strip=True)
                properties[prop_name] = prop_value
            
            if properties:
                structured_data.append({
                    "type": "microdata",
                    "itemtype": item_type,
                    "properties": properties
                })
        
        return structured_data
    
    async def _calculate_ai_surfacing_score(self, content: Dict[str, Any]) -> AISurfacingScore:
        """Calculate AI surfacing score using LLM analysis."""
        try:
            # Prepare content for analysis
            analysis_content = f"""
            Title: {content.get('title', '')}
            Description: {content.get('description', '')}
            Content: {content.get('content', '')[:2000]}  # Limit content length
            Structured Data: {json.dumps(content.get('structured_data', []))}
            """
            
            # Use available LLM to analyze content
            if self.openai_client:
                score = await self._analyze_with_openai(analysis_content)
            elif self.anthropic_client:
                score = await self._analyze_with_anthropic(analysis_content)
            else:
                # Fallback to rule-based scoring
                score = self._calculate_rule_based_score(content)
            
            return score
            
        except Exception as e:
            logger.error(f"AI surfacing score calculation failed: {e}")
            return self._default_ai_score()
    
    async def _analyze_with_openai(self, content: str) -> AISurfacingScore:
        """Analyze content using OpenAI."""
        try:
            prompt = f"""
            Analyze the following content for AI search engine optimization and provide scores (0-100) for each dimension:

            {content}

            Provide scores for:
            1. Vector Compatibility (how well content works with vector search)
            2. Structured Data Quality (schema.org, JSON-LD quality)
            3. Semantic Clarity (how clear and understandable the content is)
            4. AI SERP Readiness (optimization for AI search engines like Perplexity)
            5. Knowledge Graph Density (entity relationships and connections)
            6. Intent Alignment (how well content matches user intent)

            Respond with JSON format:
            {{
                "vector_compatibility": score,
                "structured_data_quality": score,
                "semantic_clarity": score,
                "ai_serp_readiness": score,
                "knowledge_graph_density": score,
                "intent_alignment": score
            }}
            """
            
            response = await self.openai_client.ChatCompletion.acreate(
                model=settings.DEFAULT_LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Calculate overall score
            scores = list(result.values())
            overall = sum(scores) / len(scores)
            
            return AISurfacingScore(
                overall=overall,
                vector_compatibility=result["vector_compatibility"],
                structured_data_quality=result["structured_data_quality"],
                semantic_clarity=result["semantic_clarity"],
                ai_serp_readiness=result["ai_serp_readiness"],
                knowledge_graph_density=result["knowledge_graph_density"],
                intent_alignment=result["intent_alignment"]
            )
            
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            return self._default_ai_score()
    
    async def _analyze_with_anthropic(self, content: str) -> AISurfacingScore:
        """Analyze content using Anthropic Claude."""
        try:
            prompt = f"""
            Analyze the following content for AI search engine optimization and provide scores (0-100) for each dimension:

            {content}

            Provide scores for:
            1. Vector Compatibility (how well content works with vector search)
            2. Structured Data Quality (schema.org, JSON-LD quality)
            3. Semantic Clarity (how clear and understandable the content is)
            4. AI SERP Readiness (optimization for AI search engines like Perplexity)
            5. Knowledge Graph Density (entity relationships and connections)
            6. Intent Alignment (how well content matches user intent)

            Respond with JSON format only.
            """
            
            message = await self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=500,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = json.loads(message.content[0].text)
            
            # Calculate overall score
            scores = list(result.values())
            overall = sum(scores) / len(scores)
            
            return AISurfacingScore(
                overall=overall,
                vector_compatibility=result["vector_compatibility"],
                structured_data_quality=result["structured_data_quality"],
                semantic_clarity=result["semantic_clarity"],
                ai_serp_readiness=result["ai_serp_readiness"],
                knowledge_graph_density=result["knowledge_graph_density"],
                intent_alignment=result["intent_alignment"]
            )
            
        except Exception as e:
            logger.error(f"Anthropic analysis failed: {e}")
            return self._default_ai_score()
    
    def _calculate_rule_based_score(self, content: Dict[str, Any]) -> AISurfacingScore:
        """Calculate score using rule-based approach as fallback."""
        # Vector compatibility
        vector_score = 60  # Base score
        if content.get('title') and len(content['title']) > 10:
            vector_score += 10
        if content.get('description') and len(content['description']) > 50:
            vector_score += 10
        if content.get('content') and len(content['content']) > 200:
            vector_score += 20
        
        # Structured data quality
        structured_score = 30  # Base score
        if content.get('structured_data'):
            structured_score += 40
            if len(content['structured_data']) > 1:
                structured_score += 20
            if any('Product' in str(item) for item in content['structured_data']):
                structured_score += 10
        
        # Semantic clarity
        semantic_score = 50  # Base score
        if content.get('title'):
            semantic_score += 15
        if content.get('description'):
            semantic_score += 15
        if content.get('content') and len(content['content'].split()) > 100:
            semantic_score += 20
        
        # AI SERP readiness
        serp_score = 40  # Base score
        if content.get('structured_data'):
            serp_score += 20
        if content.get('images'):
            serp_score += 10
        if content.get('title') and any(word in content['title'].lower() for word in ['best', 'top', 'review', 'guide']):
            serp_score += 20
        if content.get('description') and len(content['description']) > 120:
            serp_score += 10
        
        # Knowledge graph density
        kg_score = 35  # Base score
        if content.get('structured_data'):
            kg_score += 25
        if content.get('links') and len(content['links']) > 5:
            kg_score += 20
        if content.get('images') and len(content['images']) > 2:
            kg_score += 20
        
        # Intent alignment
        intent_score = 55  # Base score
        if content.get('title') and content.get('description'):
            intent_score += 20
        if content.get('content') and 'how to' in content['content'].lower():
            intent_score += 15
        if content.get('structured_data') and any('FAQ' in str(item) for item in content['structured_data']):
            intent_score += 10
        
        # Ensure scores are within bounds
        scores = [vector_score, structured_score, semantic_score, serp_score, kg_score, intent_score]
        scores = [min(100, max(0, score)) for score in scores]
        overall = sum(scores) / len(scores)
        
        return AISurfacingScore(
            overall=overall,
            vector_compatibility=scores[0],
            structured_data_quality=scores[1],
            semantic_clarity=scores[2],
            ai_serp_readiness=scores[3],
            knowledge_graph_density=scores[4],
            intent_alignment=scores[5]
        )
    
    def _default_ai_score(self) -> AISurfacingScore:
        """Return default AI surfacing score."""
        return AISurfacingScore(
            overall=50.0,
            vector_compatibility=50.0,
            structured_data_quality=50.0,
            semantic_clarity=50.0,
            ai_serp_readiness=50.0,
            knowledge_graph_density=50.0,
            intent_alignment=50.0
        )
    
    async def _analyze_structured_data(self, html: str) -> Optional[StructuredDataAnalysis]:
        """Analyze structured data in HTML."""
        if not html:
            return None
        
        soup = BeautifulSoup(html, 'html.parser')
        structured_data = self._extract_structured_data(soup)
        
        present = len(structured_data) > 0
        completeness = 0
        validation_results = {}
        missing_properties = []
        recommendations = []
        
        if present:
            # Calculate completeness based on common schema.org properties
            total_properties = 0
            found_properties = 0
            
            for item in structured_data:
                if item.get('type') == 'json-ld':
                    data = item.get('data', {})
                    if isinstance(data, dict):
                        total_properties += 10  # Expected properties
                        found_properties += len(data.keys())
                elif item.get('type') == 'microdata':
                    properties = item.get('properties', {})
                    total_properties += 8  # Expected properties
                    found_properties += len(properties)
            
            if total_properties > 0:
                completeness = min(100, (found_properties / total_properties) * 100)
            
            # Generate recommendations
            if completeness < 70:
                recommendations.append("Add more structured data properties")
            if not any('Product' in str(item) for item in structured_data):
                recommendations.append("Consider adding Product schema markup")
                missing_properties.append("Product schema")
        else:
            recommendations.append("Add structured data markup (JSON-LD or microdata)")
            missing_properties.append("All structured data")
        
        return StructuredDataAnalysis(
            present=present,
            type="json-ld" if any(item.get('type') == 'json-ld' for item in structured_data) else "microdata" if structured_data else None,
            completeness=completeness,
            validation_results=validation_results,
            missing_properties=missing_properties,
            recommendations=recommendations
        )
    
    async def _perform_semantic_analysis(self, content: Dict[str, Any]) -> Optional[SemanticAnalysis]:
        """Perform semantic analysis of content."""
        try:
            text_content = f"{content.get('title', '')} {content.get('description', '')} {content.get('content', '')}"
            
            # Basic semantic metrics
            clarity = self._calculate_clarity_score(text_content)
            intent_signals = self._extract_intent_signals(text_content)
            target_audience = self._identify_target_audience(text_content)
            key_terms = self._extract_key_terms(text_content)
            readability = self._calculate_readability_score(text_content)
            sentiment = self._calculate_sentiment_score(text_content)
            
            return SemanticAnalysis(
                clarity=clarity,
                intent_signals=intent_signals,
                target_audience=target_audience,
                key_terms=key_terms,
                readability_score=readability,
                sentiment_score=sentiment
            )
            
        except Exception as e:
            logger.error(f"Semantic analysis failed: {e}")
            return None
    
    def _calculate_clarity_score(self, text: str) -> float:
        """Calculate content clarity score."""
        if not text:
            return 0.0
        
        # Basic clarity metrics
        words = text.split()
        sentences = text.split('.')
        
        if not words or not sentences:
            return 0.0
        
        avg_words_per_sentence = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Score based on readability factors
        clarity = 100
        
        # Penalize very long sentences
        if avg_words_per_sentence > 25:
            clarity -= 20
        elif avg_words_per_sentence > 20:
            clarity -= 10
        
        # Penalize very long words
        if avg_word_length > 7:
            clarity -= 15
        elif avg_word_length > 6:
            clarity -= 10
        
        # Bonus for good structure
        if 10 <= avg_words_per_sentence <= 20:
            clarity += 10
        
        return max(0, min(100, clarity))
    
    def _extract_intent_signals(self, text: str) -> List[str]:
        """Extract intent signals from text."""
        intent_patterns = {
            'purchase': ['buy', 'purchase', 'order', 'shop', 'price', 'cost', 'sale'],
            'comparison': ['vs', 'versus', 'compare', 'best', 'top', 'review'],
            'information': ['how to', 'what is', 'guide', 'tutorial', 'learn'],
            'research': ['analysis', 'study', 'research', 'data', 'statistics']
        }
        
        text_lower = text.lower()
        signals = []
        
        for intent, keywords in intent_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                signals.append(intent)
        
        return signals
    
    def _identify_target_audience(self, text: str) -> Optional[str]:
        """Identify target audience from content."""
        audience_patterns = {
            'professionals': ['professional', 'business', 'enterprise', 'corporate'],
            'consumers': ['consumer', 'customer', 'user', 'people'],
            'technical': ['technical', 'developer', 'engineer', 'advanced'],
            'general': ['everyone', 'anyone', 'general', 'public']
        }
        
        text_lower = text.lower()
        
        for audience, keywords in audience_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                return audience
        
        return "general"
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text."""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Filter common words
        stop_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'more', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were'}
        
        filtered_words = [word for word in words if word not in stop_words]
        
        # Count frequency and return top terms
        from collections import Counter
        word_counts = Counter(filtered_words)
        
        return [word for word, count in word_counts.most_common(10)]
    
    def _calculate_readability_score(self, text: str) -> float:
        """Calculate readability score."""
        if not text:
            return 0.0
        
        words = text.split()
        sentences = text.split('.')
        
        if not words or not sentences:
            return 0.0
        
        avg_words_per_sentence = len(words) / len(sentences)
        avg_syllables_per_word = sum(self._count_syllables(word) for word in words) / len(words)
        
        # Simplified Flesch Reading Ease formula
        score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
        
        return max(0, min(100, score))
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word."""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Handle silent e
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _calculate_sentiment_score(self, text: str) -> Optional[float]:
        """Calculate sentiment score."""
        # Simple sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'love', 'perfect', 'awesome']
        negative_words = ['bad', 'terrible', 'awful', 'worst', 'hate', 'horrible', 'poor', 'disappointing']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return 0.0
        
        return (positive_count - negative_count) / total_sentiment_words
    
    async def _generate_embeddings(self, content: Dict[str, Any]) -> Optional[VectorEmbeddings]:
        """Generate vector embeddings for content."""
        # This would integrate with the existing vector embeddings service
        # For now, return a placeholder
        return None
    
    async def _extract_knowledge_graph(self, content: Dict[str, Any]) -> Optional[KnowledgeGraph]:
        """Extract knowledge graph from content."""
        # This would integrate with the existing knowledge graph service
        # For now, return a placeholder
        return None
    
    async def _generate_recommendations(
        self,
        ai_score: AISurfacingScore,
        structured_data: Optional[StructuredDataAnalysis],
        semantic_analysis: Optional[SemanticAnalysis]
    ) -> List[Recommendation]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # AI surfacing score recommendations
        if ai_score.structured_data_quality < 70:
            recommendations.append(Recommendation(
                category="structured_data",
                priority="high",
                description="Add or improve structured data markup (schema.org)",
                impact="15-25 point score improvement",
                implementation={"effort": "medium", "time_estimate": "2-4 hours"}
            ))
        
        if ai_score.semantic_clarity < 60:
            recommendations.append(Recommendation(
                category="content",
                priority="medium",
                description="Improve content clarity and readability",
                impact="10-15 point score improvement",
                implementation={"effort": "low", "time_estimate": "1-2 hours"}
            ))
        
        if ai_score.vector_compatibility < 70:
            recommendations.append(Recommendation(
                category="seo",
                priority="medium",
                description="Optimize content for vector search compatibility",
                impact="8-12 point score improvement",
                implementation={"effort": "medium", "time_estimate": "2-3 hours"}
            ))
        
        # Structured data recommendations
        if structured_data and not structured_data.present:
            recommendations.append(Recommendation(
                category="structured_data",
                priority="critical",
                description="Add structured data markup to improve AI search visibility",
                impact="20-30 point score improvement",
                implementation={"effort": "high", "time_estimate": "4-6 hours"}
            ))
        
        # Semantic analysis recommendations
        if semantic_analysis and semantic_analysis.clarity < 50:
            recommendations.append(Recommendation(
                category="content",
                priority="high",
                description="Simplify language and improve content structure",
                impact="12-18 point score improvement",
                implementation={"effort": "medium", "time_estimate": "3-4 hours"}
            ))
        
        return recommendations