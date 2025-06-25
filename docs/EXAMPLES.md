# LLM Finance Leaderboard - Code Examples

## 🚀 **Ready-to-Use Scripts**

This document contains complete, copy-paste ready code examples for common tasks with the LLM Finance Leaderboard system.

## 📊 **1. Model Evaluation Examples**

### Quick Evaluation Script
```python
#!/usr/bin/env python3
"""
Quick start script for model evaluation.
Save as: quick_evaluate.py
Usage: python quick_evaluate.py
"""

import asyncio
import sys
import json
from datetime import datetime
from src.evaluation.runners.benchmark_runner import BenchmarkRunner

async def main():
    """Run quick model evaluation."""
    
    print("🚀 Starting LLM Finance Leaderboard Evaluation")
    
    # Initialize runner
    runner = BenchmarkRunner()
    
    # Quick evaluation setup
    models = ["gpt-3.5-turbo"]
    tasks = ["eps_extraction", "sentiment_analysis"]
    seeds = [42]
    
    try:
        # Run evaluation
        print("Running benchmark...")
        results = await runner.run_benchmark(
            models=models,
            tasks=tasks,
            evaluation_seeds=seeds,
            max_concurrent=1
        )
        
        # Display results
        print("\n📊 Evaluation Results:")
        print("=" * 50)
        
        for result in results:
            print(f"Model: {result.model_name}")
            print(f"Task: {result.task_name}")
            print(f"Score: {result.score:.3f}")
            print(f"Latency: {result.latency_ms}ms")
            print("-" * 40)
        
        # Save results to file
        results_data = [
            {
                "model": r.model_name,
                "task": r.task_name,
                "score": r.score,
                "latency_ms": r.latency_ms,
                "timestamp": datetime.now().isoformat()
            }
            for r in results
        ]
        
        output_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"✅ Evaluation completed! Results saved to {output_file}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
```

### Batch Model Comparison
```python
#!/usr/bin/env python3
"""
Compare multiple models across financial tasks.
Save as: compare_models.py
Usage: python compare_models.py
"""

import asyncio
import pandas as pd
from src.evaluation.runners.benchmark_runner import BenchmarkRunner

async def compare_models():
    """Compare multiple models on financial tasks."""
    
    # Define evaluation matrix
    models = [
        "gpt-3.5-turbo",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "meta-llama/Llama-2-7b-chat-hf"
    ]
    
    tasks = [
        "eps_extraction",
        "sentiment_analysis",
        "ratio_calculation"
    ]
    
    runner = BenchmarkRunner()
    all_results = []
    
    print("🔄 Running model comparison...")
    
    for model in models:
        print(f"\nEvaluating {model}...")
        
        try:
            results = await runner.run_benchmark(
                models=[model],
                tasks=tasks,
                evaluation_seeds=[42, 123, 456],
                max_concurrent=1
            )
            
            for result in results:
                all_results.append({
                    "Model": result.model_name,
                    "Task": result.task_name,
                    "Score": result.score,
                    "Latency (ms)": result.latency_ms
                })
                
        except Exception as e:
            print(f"❌ Error evaluating {model}: {e}")
    
    # Create comparison table
    df = pd.DataFrame(all_results)
    
    if not df.empty:
        # Pivot table for better visualization
        pivot_scores = df.pivot(index='Model', columns='Task', values='Score')
        pivot_latency = df.pivot(index='Model', columns='Task', values='Latency (ms)')
        
        print("\n📊 Model Comparison - Scores:")
        print("=" * 60)
        print(pivot_scores.round(3))
        
        print("\n⏱️ Model Comparison - Latency (ms):")
        print("=" * 60)
        print(pivot_latency.round(0))
        
        # Save to CSV
        df.to_csv("model_comparison.csv", index=False)
        pivot_scores.to_csv("model_scores_matrix.csv")
        
        print("\n✅ Comparison completed! Results saved to CSV files.")
    else:
        print("❌ No results to display")

if __name__ == "__main__":
    asyncio.run(compare_models())
```

## 🎯 **2. Fine-tuning Examples**

### Basic Training Script
```python
#!/usr/bin/env python3
"""
Train a financial model using LoRA fine-tuning.
Save as: train_model.py
Usage: python train_model.py
"""

import time
import json
from src.training.local_orchestrator import LocalTrainingOrchestrator

def train_financial_model():
    """Train a model on financial data."""
    
    print("🚀 Starting LLM Finance Model Training")
    
    # Initialize orchestrator
    orchestrator = LocalTrainingOrchestrator()
    
    # Training configuration
    config = {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.1",
        "dataset_path": "data/training/synthetic_finance_gsib_v3.jsonl",
        "output_dir": "models/mistral_finance_specialist",
        
        # LoRA parameters
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        
        # Training parameters
        "learning_rate": 2e-5,
        "num_epochs": 3,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 50,
        
        # Hardware optimization
        "use_4bit_quantization": True,
        "use_gradient_checkpointing": True
    }
    
    print("Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    try:
        # Submit training job
        job_id = orchestrator.submit_training_job(**config)
        print(f"\n✅ Training job submitted: {job_id}")
        
        # Monitor progress
        print("\n🔄 Monitoring training progress...")
        start_time = time.time()
        
        while True:
            status = orchestrator.get_job_status(job_id)
            elapsed = time.time() - start_time
            
            print(f"Status: {status['status']} (Elapsed: {elapsed/60:.1f}min)")
            
            if 'progress' in status:
                print(f"Progress: {status['progress']}")
            
            if status['status'] in ['completed', 'failed']:
                break
            
            time.sleep(30)  # Check every 30 seconds
        
        # Final results
        if status['status'] == 'completed':
            print(f"\n🎉 Training completed successfully!")
            print(f"Model saved to: {status['output_path']}")
            
            # Save training summary
            summary = {
                "job_id": job_id,
                "config": config,
                "status": status,
                "training_time_minutes": elapsed / 60
            }
            
            with open(f"training_summary_{job_id}.json", "w") as f:
                json.dump(summary, f, indent=2)
            
            print(f"Training summary saved to training_summary_{job_id}.json")
            
        else:
            print(f"\n❌ Training failed: {status.get('error', 'Unknown error')}")
            return False
    
    except Exception as e:
        print(f"❌ Training setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = train_financial_model()
    exit(0 if success else 1)
```

### Model Comparison After Training
```python
#!/usr/bin/env python3
"""
Compare base model vs fine-tuned model performance.
Save as: compare_trained_model.py
Usage: python compare_trained_model.py <path_to_trained_model>
"""

import sys
import asyncio
from src.training.comparison_engine import ComparisonEngine
from src.evaluation.runners.benchmark_runner import BenchmarkRunner

async def compare_trained_model(trained_model_path):
    """Compare base vs trained model."""
    
    base_model = "mistralai/Mistral-7B-Instruct-v0.1"
    test_dataset = "data/training/synthetic_finance_gsib_v3.jsonl"
    
    print(f"🔄 Comparing models:")
    print(f"  Base: {base_model}")
    print(f"  Trained: {trained_model_path}")
    
    # Initialize comparison engine
    comparison_engine = ComparisonEngine()
    
    try:
        # Run comparison
        result = comparison_engine.compare_models(
            base_model=base_model,
            finetuned_model=trained_model_path,
            test_dataset=test_dataset,
            metrics=["accuracy", "f1_score", "regulatory_compliance"]
        )
        
        # Display results
        print("\n📊 Comparison Results:")
        print("=" * 50)
        print(f"Base Model Score: {result['base_score']:.3f}")
        print(f"Fine-tuned Score: {result['finetuned_score']:.3f}")
        print(f"Improvement: {result['improvement']:.3f}")
        print(f"Relative Improvement: {result['improvement']/result['base_score']*100:.1f}%")
        
        if 'p_value' in result:
            significance = "significant" if result['p_value'] < 0.05 else "not significant"
            print(f"Statistical Significance: {significance} (p={result['p_value']:.4f})")
        
        # Generate detailed report
        report = comparison_engine.generate_comparison_report(result)
        
        report_file = f"model_comparison_report.md"
        with open(report_file, "w") as f:
            f.write(report)
        
        print(f"\n✅ Detailed report saved to {report_file}")
        
        return result
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compare_trained_model.py <path_to_trained_model>")
        sys.exit(1)
    
    trained_model_path = sys.argv[1]
    result = asyncio.run(compare_trained_model(trained_model_path))
    
    if result is None:
        sys.exit(1)
```

## 🤖 **3. Financial Agent Examples**

### Basic Financial Analysis
```python
#!/usr/bin/env python3
"""
Run financial analysis using AI agent.
Save as: analyze_financial_data.py
Usage: python analyze_financial_data.py
"""

from src.agents.base_agent import FinancialAgent
from src.data.processors.vector_store import create_vector_store

def analyze_financial_documents():
    """Analyze financial documents using AI agent."""
    
    print("🤖 Starting Financial Analysis")
    
    # Initialize components
    vector_store = create_vector_store(use_mock=False)
    agent = FinancialAgent(
        model_name="gpt-3.5-turbo",
        vector_store=vector_store,
        temperature=0.1  # Low temperature for factual analysis
    )
    
    # Sample financial documents
    documents = [
        {
            "content": """
            JPMorgan Chase Q3 2024 Results:
            Net income: $13.4 billion
            Earnings per share: $4.44
            Return on equity: 17%
            Net interest income: $22.9 billion
            Credit loss provision: $3.1 billion
            """,
            "source": "JPM 10-Q Q3 2024",
            "ticker": "JPM"
        },
        {
            "content": """
            Bank of America Q3 2024 Results:
            Net income: $6.9 billion
            Earnings per share: $0.81
            Return on equity: 11.2%
            Net interest income: $14.4 billion
            Credit loss provision: $1.5 billion
            """,
            "source": "BAC 10-Q Q3 2024",
            "ticker": "BAC"
        }
    ]
    
    # Analysis queries
    queries = [
        "What is the EPS for JPMorgan Chase in Q3 2024?",
        "Compare the ROE between JPMorgan and Bank of America",
        "What are the credit loss provisions for both banks?",
        "Which bank has better profitability metrics?"
    ]
    
    results = []
    
    for i, query in enumerate(queries, 1):
        print(f"\n🔍 Analysis {i}: {query}")
        print("-" * 50)
        
        try:
            # Run analysis
            result = agent.run(
                query=query,
                context_documents=documents,
                task_type="financial_analysis"
            )
            
            print(f"Response: {result['final_response']}")
            print(f"Confidence: {result['confidence_score']:.2f}")
            print(f"Sources: {', '.join(result['sources_used'])}")
            
            results.append({
                "query": query,
                "response": result['final_response'],
                "confidence": result['confidence_score'],
                "sources": result['sources_used']
            })
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            results.append({
                "query": query,
                "error": str(e)
            })
    
    # Save results
    import json
    with open("financial_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Analysis completed! Results saved to financial_analysis_results.json")
    return results

if __name__ == "__main__":
    analyze_financial_documents()
```

### Specialized Banking Analysis
```python
#!/usr/bin/env python3
"""
G-SIB banking analysis using specialized agent.
Save as: gsib_analysis.py
Usage: python gsib_analysis.py
"""

from src.agents.base_agent import FinancialAgent

def gsib_banking_analysis():
    """Perform G-SIB banking regulatory analysis."""
    
    print("🏦 Starting G-SIB Banking Analysis")
    
    # Create specialized G-SIB agent
    agent = FinancialAgent(
        model_name="mistralai/Mistral-7B-Instruct-v0.1",
        temperature=0.05,  # Very low for regulatory precision
        max_tokens=2048,
        system_prompt="""
        You are a specialized financial analyst focusing on G-SIB 
        (Global Systemically Important Banks) regulatory analysis.
        Provide precise calculations and regulatory interpretations
        based on Basel III requirements.
        """
    )
    
    # G-SIB regulatory documents
    gsib_documents = [
        {
            "content": """
            JPMorgan Chase Basel III Metrics Q3 2024:
            CET1 Ratio: 15.0%
            Tier 1 Capital Ratio: 16.2%
            Total Capital Ratio: 18.1%
            Leverage Ratio: 5.1%
            LCR: 125%
            NSFR: 118%
            G-SIB Buffer: 2.5%
            """,
            "source": "JPM Basel III Disclosure",
            "ticker": "JPM"
        },
        {
            "content": """
            Bank of America Basel III Metrics Q3 2024:
            CET1 Ratio: 14.8%
            Tier 1 Capital Ratio: 15.9%
            Total Capital Ratio: 17.8%
            Leverage Ratio: 4.9%
            LCR: 120%
            NSFR: 115%
            G-SIB Buffer: 2.0%
            """,
            "source": "BAC Basel III Disclosure",
            "ticker": "BAC"
        }
    ]
    
    # G-SIB analysis queries
    gsib_queries = [
        "Calculate the excess CET1 capital above regulatory minimums for both banks",
        "Assess liquidity risk based on LCR and NSFR ratios",
        "Compare G-SIB buffer requirements and compliance",
        "Evaluate overall regulatory capital strength",
        "Identify any potential regulatory concerns"
    ]
    
    print("\n📋 G-SIB Regulatory Analysis:")
    print("=" * 60)
    
    analysis_results = []
    
    for i, query in enumerate(gsib_queries, 1):
        print(f"\n{i}. {query}")
        print("-" * 50)
        
        try:
            result = agent.run(
                query=query,
                context_documents=gsib_documents,
                task_type="regulatory_compliance"
            )
            
            print(f"Analysis: {result['final_response']}")
            print(f"Confidence: {result['confidence_score']:.2f}")
            
            analysis_results.append({
                "query": query,
                "analysis": result['final_response'],
                "confidence": result['confidence_score']
            })
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
    
    # Generate regulatory report
    report = generate_gsib_report(analysis_results)
    
    with open("gsib_regulatory_analysis.md", "w") as f:
        f.write(report)
    
    print(f"\n✅ G-SIB analysis completed! Report saved to gsib_regulatory_analysis.md")
    return analysis_results

def generate_gsib_report(results):
    """Generate formatted G-SIB analysis report."""
    
    report = """# G-SIB Regulatory Analysis Report

## Executive Summary

This report provides a comprehensive analysis of G-SIB (Global Systemically Important Banks) 
regulatory compliance based on Basel III requirements.

## Analysis Results

"""
    
    for i, result in enumerate(results, 1):
        report += f"""
### {i}. {result['query']}

**Analysis:** {result['analysis']}

**Confidence Score:** {result['confidence']:.2f}

---
"""
    
    report += """
## Regulatory Framework

This analysis is based on:
- Basel III Capital Requirements
- G-SIB Buffer Requirements
- Liquidity Coverage Ratio (LCR)
- Net Stable Funding Ratio (NSFR)
- Leverage Ratio Requirements

## Disclaimer

This analysis is for informational purposes only and should not be considered as 
regulatory advice. Always consult with qualified regulatory experts for official 
compliance assessments.
"""
    
    return report

if __name__ == "__main__":
    gsib_banking_analysis()
```

## 📊 **4. Data Processing Examples**

### SEC Data Collection
```python
#!/usr/bin/env python3
"""
Collect and process SEC filings.
Save as: collect_sec_data.py
Usage: python collect_sec_data.py
"""

from src.data.collectors.sec_filings import SECFilingsCollector
from src.data.processors.vector_store import create_vector_store

def collect_and_process_sec_data():
    """Collect SEC filings and add to vector store."""
    
    print("📄 Starting SEC Data Collection")
    
    # Initialize components
    collector = SECFilingsCollector()
    vector_store = create_vector_store(use_mock=False)
    
    # Major banks to collect data for
    banks = [
        {"ticker": "JPM", "name": "JPMorgan Chase"},
        {"ticker": "BAC", "name": "Bank of America"},
        {"ticker": "WFC", "name": "Wells Fargo"},
        {"ticker": "C", "name": "Citigroup"},
        {"ticker": "GS", "name": "Goldman Sachs"},
        {"ticker": "MS", "name": "Morgan Stanley"}
    ]
    
    filing_types = ["10-Q", "10-K"]
    collected_count = 0
    
    for bank in banks:
        ticker = bank["ticker"]
        name = bank["name"]
        
        print(f"\n🏦 Collecting data for {name} ({ticker})")
        print("-" * 40)
        
        for filing_type in filing_types:
            try:
                print(f"  Fetching {filing_type} filings...")
                
                filings = collector.get_recent_filings(
                    ticker=ticker,
                    filing_type=filing_type,
                    count=2  # Last 2 filings
                )
                
                for filing in filings:
                    print(f"    Processing: {filing['filing_date']}")
                    
                    # Extract financial data
                    processed_content = collector.extract_financial_data(filing)
                    
                    # Add to vector store
                    doc_id = f"{ticker}_{filing_type}_{filing['filing_date']}"
                    vector_store.add_document(
                        doc_id=doc_id,
                        content=processed_content,
                        metadata={
                            "ticker": ticker,
                            "company_name": name,
                            "filing_type": filing_type,
                            "filing_date": filing['filing_date'],
                            "url": filing.get('url', '')
                        }
                    )
                    
                    collected_count += 1
                    print(f"    ✅ Added to vector store: {doc_id}")
                    
            except Exception as e:
                print(f"    ❌ Error collecting {ticker} {filing_type}: {e}")
    
    print(f"\n🎉 Collection completed!")
    print(f"Total documents collected: {collected_count}")
    
    # Test retrieval
    print("\n🔍 Testing document retrieval...")
    test_queries = [
        "JPMorgan earnings per share",
        "Bank of America net income",
        "Wells Fargo loan loss provisions"
    ]
    
    for query in test_queries:
        results = vector_store.similarity_search(query, k=3)
        print(f"\nQuery: {query}")
        for result in results[:2]:  # Show top 2 results
            print(f"  Found: {result['id']} (score: {result['score']:.3f})")
    
    return collected_count

if __name__ == "__main__":
    collect_and_process_sec_data()
```

### Vector Store Management
```python
#!/usr/bin/env python3
"""
Manage vector store operations.
Save as: manage_vector_store.py
Usage: python manage_vector_store.py
"""

from src.data.processors.vector_store import create_vector_store
from src.data.processors.embeddings import create_embedding_generator
import json

def manage_vector_store():
    """Demonstrate vector store management operations."""
    
    print("🗄️ Vector Store Management Demo")
    
    # Initialize components
    vector_store = create_vector_store(use_mock=False)
    embedder = create_embedding_generator(use_mock=False)
    
    # Sample financial documents
    sample_documents = [
        {
            "id": "jpm_q3_2024_earnings",
            "content": """
            JPMorgan Chase Q3 2024 Earnings Summary:
            - Net income: $13.4 billion
            - Earnings per share: $4.44
            - Return on equity: 17%
            - Net interest income: $22.9 billion
            - Provision for credit losses: $3.1 billion
            - Book value per share: $95.10
            """,
            "metadata": {
                "ticker": "JPM",
                "document_type": "earnings_report",
                "quarter": "Q3",
                "year": 2024,
                "category": "financial_results"
            }
        },
        {
            "id": "bac_q3_2024_earnings",
            "content": """
            Bank of America Q3 2024 Earnings Summary:
            - Net income: $6.9 billion
            - Earnings per share: $0.81
            - Return on equity: 11.2%
            - Net interest income: $14.4 billion
            - Provision for credit losses: $1.5 billion
            - Book value per share: $35.52
            """,
            "metadata": {
                "ticker": "BAC",
                "document_type": "earnings_report",
                "quarter": "Q3",
                "year": 2024,
                "category": "financial_results"
            }
        },
        {
            "id": "market_analysis_q3_2024",
            "content": """
            Q3 2024 Banking Sector Analysis:
            - Overall sector performance remained strong
            - Net interest margins under pressure from rate environment
            - Credit quality metrics stable across major banks
            - Regulatory capital ratios well above minimums
            - Digital transformation investments continue
            """,
            "metadata": {
                "document_type": "market_analysis",
                "quarter": "Q3",
                "year": 2024,
                "category": "sector_analysis"
            }
        }
    ]
    
    print("\n📝 Adding documents to vector store...")
    
    # Add documents
    for doc in sample_documents:
        try:
            vector_store.add_document(
                doc_id=doc["id"],
                content=doc["content"],
                metadata=doc["metadata"]
            )
            print(f"  ✅ Added: {doc['id']}")
        except Exception as e:
            print(f"  ❌ Failed to add {doc['id']}: {e}")
    
    print(f"\n📊 Vector store now contains {len(sample_documents)} documents")
    
    # Demonstrate search capabilities
    print("\n🔍 Testing search capabilities...")
    
    search_queries = [
        {
            "query": "JPMorgan earnings per share Q3 2024",
            "description": "Specific EPS lookup"
        },
        {
            "query": "bank net income comparison",
            "description": "Comparative analysis"
        },
        {
            "query": "credit loss provisions banking sector",
            "description": "Risk metrics search"
        },
        {
            "query": "return on equity performance",
            "description": "Profitability metrics"
        }
    ]
    
    search_results = []
    
    for search in search_queries:
        query = search["query"]
        description = search["description"]
        
        print(f"\n  Query: {query}")
        print(f"  Purpose: {description}")
        
        try:
            results = vector_store.similarity_search(query, k=3)
            
            print("  Results:")
            for i, result in enumerate(results, 1):
                print(f"    {i}. {result['id']} (score: {result['score']:.3f})")
                if 'metadata' in result:
                    ticker = result['metadata'].get('ticker', 'N/A')
                    doc_type = result['metadata'].get('document_type', 'N/A')
                    print(f"       Ticker: {ticker}, Type: {doc_type}")
            
            search_results.append({
                "query": query,
                "description": description,
                "results": results
            })
            
        except Exception as e:
            print(f"    ❌ Search failed: {e}")
    
    # Save search results
    with open("vector_store_search_results.json", "w") as f:
        # Convert results to JSON-serializable format
        json_results = []
        for search in search_results:
            json_search = {
                "query": search["query"],
                "description": search["description"],
                "results": [
                    {
                        "id": r["id"],
                        "score": r["score"],
                        "metadata": r.get("metadata", {})
                    }
                    for r in search["results"]
                ]
            }
            json_results.append(json_search)
        
        json.dump(json_results, f, indent=2)
    
    print(f"\n✅ Vector store demo completed!")
    print(f"Search results saved to vector_store_search_results.json")
    
    return search_results

if __name__ == "__main__":
    manage_vector_store()
```

## 🔧 **5. System Utilities**

### System Health Check
```python
#!/usr/bin/env python3
"""
Comprehensive system health check.
Save as: health_check.py
Usage: python health_check.py
"""

import sys
import json
from datetime import datetime

def comprehensive_health_check():
    """Perform comprehensive system health check."""
    
    print("🏥 LLM Finance Leaderboard - System Health Check")
    print("=" * 60)
    
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "overall_health": True,
        "components": {}
    }
    
    # 1. Environment Variables Check
    print("\n1. 🔧 Environment Variables")
    print("-" * 30)
    
    required_env_vars = [
        "PINECONE_API_KEY",
        "PINECONE_ENVIRONMENT",
        "OPENAI_API_KEY"
    ]
    
    optional_env_vars = [
        "ANTHROPIC_API_KEY",
        "HUGGINGFACE_API_TOKEN",
        "FRED_API_KEY",
        "ALPHA_VANTAGE_API_KEY"
    ]
    
    env_status = {"required": {}, "optional": {}}
    
    import os
    for var in required_env_vars:
        is_set = bool(os.getenv(var))
        env_status["required"][var] = is_set
        status_icon = "✅" if is_set else "❌"
        print(f"  {status_icon} {var}: {'Set' if is_set else 'Missing'}")
        
        if not is_set:
            health_status["overall_health"] = False
    
    for var in optional_env_vars:
        is_set = bool(os.getenv(var))
        env_status["optional"][var] = is_set
        status_icon = "✅" if is_set else "⚠️"
        print(f"  {status_icon} {var}: {'Set' if is_set else 'Not set (optional)'}")
    
    health_status["components"]["environment"] = env_status
    
    # 2. Configuration Check
    print("\
n2. ⚙️ Configuration")
    print("-" * 30)
    
    config_status = {}
    
    try:
        from src.config.settings import settings
        
        # Test configuration loading
        config_tests = [
            ("Database URL", hasattr(settings, 'database_url')),
            ("Vector Index", hasattr(settings, 'vector_index_name')),
            ("Benchmark Seeds", hasattr(settings, 'benchmark_seeds')),
            ("Default Temperature", hasattr(settings, 'default_temperature')),
            ("Max Tokens", hasattr(settings, 'default_max_tokens'))
        ]
        
        for test_name, test_result in config_tests:
            status_icon = "✅" if test_result else "❌"
            print(f"  {status_icon} {test_name}: {'OK' if test_result else 'Missing'}")
            config_status[test_name] = test_result
            
            if not test_result:
                health_status["overall_health"] = False
        
        # Test directory creation
        try:
            settings.create_directories()
            print("  ✅ Directory creation: OK")
            config_status["directory_creation"] = True
        except Exception as e:
            print(f"  ❌ Directory creation: Failed ({e})")
            config_status["directory_creation"] = False
            health_status["overall_health"] = False
            
    except Exception as e:
        print(f"  ❌ Configuration loading failed: {e}")
        config_status["loading_error"] = str(e)
        health_status["overall_health"] = False
    
    health_status["components"]["configuration"] = config_status
    
    # 3. Model Loading Check
    print("\n3. 🤖 Model Loading")
    print("-" * 30)
    
    model_status = {}
    
    try:
        from src.models.model_loader import ModelLoader
        loader = ModelLoader()
        
        # Test with a lightweight model
        test_model = "gpt-3.5-turbo"
        print(f"  Testing model loading: {test_model}")
        
        model = loader.load_model(test_model)
        if model:
            print("  ✅ Model loading: OK")
            model_status["loading"] = True
            
            # Test response generation
            try:
                response = loader.generate_response(
                    model=model,
                    prompt="Test prompt",
                    max_tokens=10
                )
                print("  ✅ Response generation: OK")
                model_status["generation"] = True
            except Exception as e:
                print(f"  ❌ Response generation: Failed ({e})")
                model_status["generation"] = False
                health_status["overall_health"] = False
        else:
            print("  ❌ Model loading: Failed")
            model_status["loading"] = False
            health_status["overall_health"] = False
            
    except Exception as e:
        print(f"  ❌ Model loader initialization failed: {e}")
        model_status["initialization_error"] = str(e)
        health_status["overall_health"] = False
    
    health_status["components"]["model_loading"] = model_status
    
    # 4. Vector Store Check
    print("\n4. 🗄️ Vector Store")
    print("-" * 30)
    
    vector_status = {}
    
    try:
        from src.data.processors.vector_store import create_vector_store
        
        vector_store = create_vector_store(use_mock=False)
        print("  ✅ Vector store creation: OK")
        vector_status["creation"] = True
        
        # Test search functionality
        try:
            results = vector_store.similarity_search("test query", k=1)
            print("  ✅ Search functionality: OK")
            vector_status["search"] = True
        except Exception as e:
            print(f"  ⚠️ Search functionality: Limited ({e})")
            vector_status["search"] = False
            
    except Exception as e:
        print(f"  ❌ Vector store initialization failed: {e}")
        vector_status["initialization_error"] = str(e)
        health_status["overall_health"] = False
    
    health_status["components"]["vector_store"] = vector_status
    
    # 5. Training System Check
    print("\n5. 🎯 Training System")
    print("-" * 30)
    
    training_status = {}
    
    try:
        from src.training.local_orchestrator import LocalTrainingOrchestrator
        
        orchestrator = LocalTrainingOrchestrator()
        print("  ✅ Training orchestrator: OK")
        training_status["orchestrator"] = True
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                print(f"  ✅ GPU availability: {gpu_count} GPU(s) available")
                training_status["gpu_available"] = True
                training_status["gpu_count"] = gpu_count
            else:
                print("  ⚠️ GPU availability: No GPU detected (CPU training only)")
                training_status["gpu_available"] = False
        except ImportError:
            print("  ⚠️ PyTorch not available for GPU check")
            training_status["pytorch_available"] = False
            
    except Exception as e:
        print(f"  ❌ Training system initialization failed: {e}")
        training_status["initialization_error"] = str(e)
        health_status["overall_health"] = False
    
    health_status["components"]["training_system"] = training_status
    
    # 6. Data Files Check
    print("\n6. 📄 Data Files")
    print("-" * 30)
    
    data_status = {}
    
    required_files = [
        "data/training/synthetic_finance_gsib_v3.jsonl",
        "src/config/training_config.yaml",
        "configs/benchmark_config.yaml"
    ]
    
    for file_path in required_files:
        file_exists = os.path.exists(file_path)
        status_icon = "✅" if file_exists else "❌"
        print(f"  {status_icon} {file_path}: {'Exists' if file_exists else 'Missing'}")
        data_status[file_path] = file_exists
        
        if not file_exists:
            health_status["overall_health"] = False
    
    health_status["components"]["data_files"] = data_status
    
    # Overall Health Summary
    print("\n" + "=" * 60)
    overall_icon = "✅" if health_status["overall_health"] else "❌"
    overall_text = "HEALTHY" if health_status["overall_health"] else "ISSUES DETECTED"
    print(f"{overall_icon} Overall System Health: {overall_text}")
    
    if not health_status["overall_health"]:
        print("\n⚠️ Issues found. Please review the failed components above.")
        print("   Refer to the setup documentation for resolution steps.")
    
    # Save health report
    report_file = f"health_check_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump(health_status, f, indent=2)
    
    print(f"\n📄 Detailed health report saved to: {report_file}")
    
    return health_status

if __name__ == "__main__":
    health = comprehensive_health_check()
    
    # Exit with error code if unhealthy
    if not health["overall_health"]:
        sys.exit(1)
    else:
        print("\n🎉 System is ready for use!")
        sys.exit(0)
```

## 📋 **Summary**

This examples document provides ready-to-use scripts for:

### 🚀 **Model Evaluation**
- **Quick Evaluation**: Single model testing with results export
- **Batch Comparison**: Multi-model performance comparison with CSV output

### 🎯 **Fine-tuning**
- **Basic Training**: LoRA fine-tuning with progress monitoring
- **Model Comparison**: Base vs fine-tuned performance analysis

### 🤖 **Financial Agents**
- **Basic Analysis**: Document analysis with AI agents
- **G-SIB Banking**: Specialized regulatory compliance analysis

### 📊 **Data Processing**
- **SEC Collection**: Automated SEC filings collection and processing
- **Vector Store**: Document storage and similarity search

### 🔧 **System Utilities**
- **Health Check**: Comprehensive system diagnostics
- **Configuration**: Environment and setup validation

### 💡 **Usage Tips**

1. **Save scripts** with the suggested filenames for easy reference
2. **Modify configurations** to match your specific requirements
3. **Check logs** in the generated files for detailed debugging
4. **Run health checks** before starting major operations
5. **Use batch processing** for efficient multi-model evaluations

### 🔗 **Related Documentation**
- [Developer API Guide](DEVELOPER_API_GUIDE.md) - Complete API reference
- [Quick Reference](QUICK_REFERENCE.md) - Essential commands
- [Technical Setup](TECHNICAL_SETUP_GUIDE.md) - Installation and configuration

---

**Ready to start?** Copy any script above, save it with the suggested filename, and run it to begin using the LLM Finance Leaderboard system programmatically!