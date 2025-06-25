#!/usr/bin/env python3
"""
Test script to verify LLM Finance Leaderboard system functionality.

This script tests all the critical components we've implemented.
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.settings import settings
from src.utils.database import init_database
from src.data.processors.vector_store import create_vector_store
from src.data.processors.document_parser import create_document_parser
from src.data.processors.embeddings import create_embedding_generator
from src.agents.retriever_agent import create_retriever_agent
from src.tasks.low_complexity.eps_extraction import create_eps_extraction_task
from src.evaluation.metrics.quality_metrics import calculate_quality_metrics
from src.evaluation.runners.benchmark_runner import run_quick_benchmark
from src.data.schemas.data_models import Document, DocumentType
from loguru import logger


async def test_system_components():
    """Test all system components."""
    logger.info("🚀 Starting LLM Finance Leaderboard System Test")
    
    # Test 1: Database initialization
    logger.info("📊 Testing database initialization...")
    try:
        init_database()
        logger.success("✅ Database initialization successful")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False
    
    # Test 2: Vector store (using mock)
    logger.info("🔍 Testing vector store...")
    try:
        vector_store = create_vector_store(use_mock=True)
        
        # Create sample document
        sample_doc = Document(
            title="Sample 10-Q Filing",
            content="JPMorgan Chase reported diluted earnings per share of $4.44 for Q1 2024.",
            document_type=DocumentType.SEC_10Q
        )
        
        # Test upsert
        success = vector_store.upsert_documents([sample_doc])
        if success:
            logger.success("✅ Vector store upsert successful")
        else:
            logger.error("❌ Vector store upsert failed")
            return False
        
        # Test search
        results = vector_store.similarity_search("earnings per share", k=3)
        if results:
            logger.success(f"✅ Vector store search successful - found {len(results)} results")
        else:
            logger.warning("⚠️ Vector store search returned no results")
        
    except Exception as e:
        logger.error(f"❌ Vector store test failed: {e}")
        return False
    
    # Test 3: Document parser
    logger.info("📄 Testing document parser...")
    try:
        parser = create_document_parser()
        
        sample_doc = Document(
            title="Test Document",
            content="The company reported earnings per share of $3.12 for the quarter.",
            document_type=DocumentType.SEC_10Q
        )
        
        parsed_info = parser.parse_document(sample_doc)
        if parsed_info and "chunks" in parsed_info:
            logger.success(f"✅ Document parser successful - created {len(parsed_info['chunks'])} chunks")
        else:
            logger.error("❌ Document parser failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Document parser test failed: {e}")
        return False
    
    # Test 4: Embedding generator (using mock)
    logger.info("🧠 Testing embedding generator...")
    try:
        embedder = create_embedding_generator(use_mock=True)
        
        test_texts = ["earnings per share", "revenue growth", "financial performance"]
        embeddings = embedder.generate_embeddings(test_texts)
        
        if embeddings and len(embeddings) == len(test_texts):
            logger.success(f"✅ Embedding generator successful - created {len(embeddings)} embeddings")
        else:
            logger.error("❌ Embedding generator failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Embedding generator test failed: {e}")
        return False
    
    # Test 5: EPS extraction task
    logger.info("📈 Testing EPS extraction task...")
    try:
        task = create_eps_extraction_task()
        
        # Test prompt generation
        context = {
            "documents": [{
                "content": "Diluted earnings per share was $4.44 for the quarter.",
                "title": "Q1 Results"
            }],
            "expected_eps": "4.44"
        }
        
        prompt = task.generate_prompt(context)
        if prompt and "earnings per share" in prompt.lower():
            logger.success("✅ EPS task prompt generation successful")
        else:
            logger.error("❌ EPS task prompt generation failed")
            return False
        
        # Test validation
        scores = task.validate_response("4.44", "4.44")
        if scores.get("exact_match", 0) == 1.0:
            logger.success("✅ EPS task validation successful")
        else:
            logger.error("❌ EPS task validation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ EPS extraction task test failed: {e}")
        return False
    
    # Test 6: Quality metrics
    logger.info("📊 Testing quality metrics...")
    try:
        metrics = calculate_quality_metrics(
            response="The earnings per share was $4.44",
            reference="Earnings per share: $4.44",
            task_type="numerical"
        )
        
        if metrics and "f1_score" in metrics:
            logger.success(f"✅ Quality metrics successful - F1: {metrics['f1_score']:.3f}")
        else:
            logger.error("❌ Quality metrics failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Quality metrics test failed: {e}")
        return False
    
    # Test 7: Quick benchmark (mock)
    logger.info("🏃 Testing quick benchmark...")
    try:
        # This will use mock implementations
        result = await run_quick_benchmark(
            models=["mock-model-1"],
            tasks=["eps_extraction"]
        )
        
        if result and result.status == "completed":
            logger.success(f"✅ Quick benchmark successful - Duration: {result.total_duration_minutes:.2f}min")
        else:
            logger.error("❌ Quick benchmark failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Quick benchmark test failed: {e}")
        return False
    
    logger.success("🎉 All system tests passed!")
    return True


def test_configuration():
    """Test configuration loading."""
    logger.info("⚙️ Testing configuration...")
    
    try:
        # Test settings loading
        logger.info(f"Database URL: {settings.database_url}")
        logger.info(f"Vector index: {settings.vector_index_name}")
        logger.info(f"Embedding model: {settings.default_embedding_model}")
        logger.info(f"Benchmark seeds: {settings.benchmark_seeds}")
        
        # Test directory creation
        settings.create_directories()
        logger.success("✅ Configuration test successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False


async def main():
    """Main test function."""
    logger.info("🧪 LLM Finance Leaderboard - System Test Suite")
    logger.info("=" * 60)
    
    # Test configuration first
    if not test_configuration():
        logger.error("❌ Configuration tests failed")
        return 1
    
    # Test system components
    if not await test_system_components():
        logger.error("❌ System component tests failed")
        return 1
    
    logger.success("🎉 All tests passed! System is functional.")
    logger.info("=" * 60)
    logger.info("📋 Next steps:")
    logger.info("1. Configure API keys in .env file")
    logger.info("2. Run: streamlit run streamlit_app/main.py")
    logger.info("3. Run: python scripts/run_benchmark.py --verbose")
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))