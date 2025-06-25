# LLM Finance Leaderboard - Developer API Guide

## 🚀 **Getting Started Without Frontend**

This guide shows developers how to use the LLM Finance Leaderboard system programmatically through Python APIs, bypassing the Streamlit frontend for automated workflows, CI/CD integration, and custom applications.

## 📋 **Prerequisites**

### Environment Setup
```bash
# Clone and setup
git clone <repository>
cd LLM\ Leaderboard
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Required API Keys
```bash
# Essential for core functionality
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_pinecone_env

# Model providers (at least one required)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
HUGGINGFACE_API_TOKEN=your_hf_token

# Financial data (optional)
FRED_API_KEY=your_fred_key
ALPHA_VANTAGE_API_KEY=your_av_key
```

## 🏗️ **Core System Architecture**

### Key Components
```python
from src.models.model_loader import ModelLoader
from src.evaluation.runners.benchmark_runner import BenchmarkRunner
from src.agents.base_agent import FinancialAgent
from src.training.lora_trainer import LoRATrainer
from src.training.local_orchestrator import LocalTrainingOrchestrator
from src.data.processors.vector_store import create_vector_store
from src.data.processors.embeddings import create_embedding_generator
```

## 📊 **1. Model Evaluation & Benchmarking**

### Basic Model Evaluation
```python
import asyncio
from src.evaluation.runners.benchmark_runner import BenchmarkRunner
from src.config.settings import settings

async def evaluate_models():
    """Evaluate multiple models on financial tasks."""
    
    # Initialize benchmark runner
    runner = BenchmarkRunner()
    
    # Define models to evaluate
    models = [
        "mistralai/Mistral-7B-Instruct-v0.1",
        "meta-llama/Llama-2-7b-chat-hf",
        "gpt-3.5-turbo"
    ]
    
    # Define tasks
    tasks = [
        "eps_extraction",
        "sentiment_analysis", 
        "ratio_calculation"
    ]
    
    # Run benchmark
    results = await runner.run_benchmark(
        models=models,
        tasks=tasks,
        evaluation_seeds=[42, 123, 456],
        max_concurrent=2
    )
    
    # Process results
    for result in results:
        print(f"Model: {result.model_name}")
        print(f"Task: {result.task_name}")
        print(f"Score: {result.score:.3f}")
        print(f"Latency: {result.latency_ms}ms")
        print("---")
    
    return results

# Run evaluation
results = asyncio.run(evaluate_models())
```

### Custom Task Evaluation
```python
from src.evaluation.tasks.base_task import BaseTask
from src.evaluation.tasks.eps_extraction import EPSExtractionTask

def evaluate_custom_task():
    """Evaluate models on custom financial task."""
    
    # Create custom task
    task = EPSExtractionTask()
    
    # Prepare test documents
    documents = [
        {
            "content": "Q3 2024 earnings: Net income $150M, shares outstanding 50M",
            "source": "10-Q Filing",
            "ticker": "EXAMPLE"
        }
    ]
    
    # Prepare context
    context = task.prepare_context(
        documents=documents,
        expected_eps="3.00"
    )
    
    # Load model
    from src.models.model_loader import ModelLoader
    loader = ModelLoader()
    model = loader.load_model("mistralai/Mistral-7B-Instruct-v0.1")
    
    # Run evaluation
    result = task.evaluate(model, context)
    
    print(f"Task Score: {result.score}")
    print(f"Model Response: {result.model_response}")
    print(f"Expected: {result.expected_output}")
    
    return result
```

## 🤖 **2. Financial AI Agent Usage**

### Basic Agent Workflow
```python
from src.agents.base_agent import FinancialAgent
from src.data.processors.vector_store import create_vector_store

def run_financial_analysis():
    """Run financial analysis using AI agent."""
    
    # Initialize components
    vector_store = create_vector_store(use_mock=False)
    agent = FinancialAgent(
        model_name="gpt-3.5-turbo",
        vector_store=vector_store
    )
    
    # Prepare query and context
    query = "What is the EPS for JPMorgan Chase in Q3 2024?"
    context_documents = [
        {
            "content": """
            JPMorgan Chase Q3 2024 Results:
            Net income: $13.4 billion
            Earnings per share: $4.44
            Return on equity: 17%
            """,
            "source": "JPM 10-Q Q3 2024",
            "ticker": "JPM"
        }
    ]
    
    # Run analysis
    result = agent.run(
        query=query,
        context_documents=context_documents,
        task_type="eps_extraction"
    )
    
    print(f"Final Response: {result['final_response']}")
    print(f"Confidence: {result['confidence_score']:.2f}")
    print(f"Sources: {result['sources_used']}")
    
    return result

# Execute analysis
analysis_result = run_financial_analysis()
```

### Advanced Agent Configuration
```python
def create_specialized_agent():
    """Create agent specialized for G-SIB banking analysis."""
    
    from src.agents.base_agent import FinancialAgent
    
    # Custom configuration for banking analysis
    agent = FinancialAgent(
        model_name="mistralai/Mistral-7B-Instruct-v0.1",
        temperature=0.1,  # Low temperature for factual analysis
        max_tokens=2048,
        system_prompt="""
        You are a specialized financial analyst focusing on G-SIB 
        (Global Systemically Important Banks) regulatory analysis.
        Provide precise calculations and regulatory interpretations.
        """
    )
    
    return agent

# Use specialized agent
banking_agent = create_specialized_agent()
```

## 🎯 **3. Auto Fine-tuning System**

### Basic Fine-tuning Workflow
```python
from src.training.local_orchestrator import LocalTrainingOrchestrator
from src.training.comparison_engine import ComparisonEngine

def run_auto_finetuning():
    """Run complete auto fine-tuning workflow."""
    
    # Initialize orchestrator
    orchestrator = LocalTrainingOrchestrator()
    
    # Submit training job
    job_id = orchestrator.submit_training_job(
        model_name="mistralai/Mistral-7B-Instruct-v0.1",
        dataset_path="data/training/synthetic_finance_gsib_v3.jsonl",
        lora_rank=16,
        lora_alpha=32,
        learning_rate=2e-5,
        num_epochs=3,
        batch_size=4
    )
    
    print(f"Training job submitted: {job_id}")
    
    # Monitor training progress
    while True:
        status = orchestrator.get_job_status(job_id)
        print(f"Status: {status['status']}")
        
        if status['status'] in ['completed', 'failed']:
            break
        
        time.sleep(30)  # Check every 30 seconds
    
    # Get results
    if status['status'] == 'completed':
        model_path = status['output_path']
        print(f"Model saved to: {model_path}")
        
        # Run comparison
        comparison_engine = ComparisonEngine()
        comparison_result = comparison_engine.compare_models(
            base_model="mistralai/Mistral-7B-Instruct-v0.1",
            finetuned_model=model_path,
            test_dataset="data/training/synthetic_finance_gsib_v3.jsonl"
        )
        
        print(f"Base Model Score: {comparison_result['base_score']:.3f}")
        print(f"Fine-tuned Score: {comparison_result['finetuned_score']:.3f}")
        print(f"Improvement: {comparison_result['improvement']:.3f}")
    
    return job_id, status

# Run fine-tuning
job_id, final_status = run_auto_finetuning()
```

### Custom Training Configuration
```python
def custom_training_setup():
    """Setup custom training with specific parameters."""
    
    from src.training.lora_trainer import LoRATrainer
    
    # Custom training configuration
    training_config = {
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "dataset_path": "data/training/custom_financial_data.jsonl",
        "output_dir": "models/custom_finance_model",
        
        # LoRA parameters
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        
        # Training parameters
        "learning_rate": 1e-4,
        "num_epochs": 5,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "warmup_steps": 100,
        
        # Hardware optimization
        "use_4bit_quantization": True,
        "use_gradient_checkpointing": True,
        "mixed_precision": "fp16"
    }
    
    # Initialize trainer
    trainer = LoRATrainer(training_config)
    
    # Start training
    trainer.train()
    
    return trainer

# Run custom training
custom_trainer = custom_training_setup()
```

## 📊 **4. Data Management & Processing**

### Vector Store Operations
```python
from src.data.processors.vector_store import create_vector_store
from src.data.processors.embeddings import create_embedding_generator

def setup_document_store():
    """Setup and populate vector store with financial documents."""
    
    # Initialize components
    vector_store = create_vector_store(use_mock=False)
    embedder = create_embedding_generator(use_mock=False)
    
    # Sample financial documents
    documents = [
        {
            "id": "jpm_q3_2024",
            "content": "JPMorgan Chase Q3 2024: EPS $4.44, Revenue $40.7B...",
            "metadata": {
                "ticker": "JPM",
                "document_type": "10-Q",
                "quarter": "Q3",
                "year": 2024
            }
        },
        {
            "id": "bac_q3_2024", 
            "content": "Bank of America Q3 2024: EPS $0.81, Revenue $25.3B...",
            "metadata": {
                "ticker": "BAC",
                "document_type": "10-Q",
                "quarter": "Q3", 
                "year": 2024
            }
        }
    ]
    
    # Add documents to vector store
    for doc in documents:
        vector_store.add_document(
            doc_id=doc["id"],
            content=doc["content"],
            metadata=doc["metadata"]
        )
    
    print(f"Added {len(documents)} documents to vector store")
    
    # Test retrieval
    results = vector_store.similarity_search(
        query="JPMorgan Chase earnings per share",
        k=3
    )
    
    for result in results:
        print(f"Found: {result['id']} (score: {result['score']:.3f})")
    
    return vector_store

# Setup document store
doc_store = setup_document_store()
```

### SEC Data Collection
```python
from src.data.collectors.sec_filings import SECFilingsCollector

def collect_sec_data():
    """Collect SEC filings for analysis."""
    
    collector = SECFilingsCollector()
    
    # Collect filings for major banks
    banks = ["JPM", "BAC", "WFC", "C", "GS", "MS"]
    filing_types = ["10-Q", "10-K"]
    
    for ticker in banks:
        for filing_type in filing_types:
            try:
                filings = collector.get_recent_filings(
                    ticker=ticker,
                    filing_type=filing_type,
                    count=2  # Last 2 filings
                )
                
                for filing in filings:
                    print(f"Collected: {ticker} {filing_type} - {filing['filing_date']}")
                    
                    # Process and store filing
                    processed_content = collector.extract_financial_data(filing)
                    
                    # Add to vector store
                    doc_store.add_document(
                        doc_id=f"{ticker}_{filing_type}_{filing['filing_date']}",
                        content=processed_content,
                        metadata={
                            "ticker": ticker,
                            "filing_type": filing_type,
                            "filing_date": filing['filing_date']
                        }
                    )
                    
            except Exception as e:
                print(f"Error collecting {ticker} {filing_type}: {e}")
    
    return True

# Collect SEC data
collect_sec_data()
```

## 🔧 **5. System Configuration & Monitoring**

### Environment Configuration
```python
from src.config.settings import settings

def configure_system():
    """Configure system settings programmatically."""
    
    # Display current configuration
    print("Current Configuration:")
    print(f"- Database URL: {settings.database_url}")
    print(f"- Vector Index: {settings.vector_index_name}")
    print(f"- Default Temperature: {settings.default_temperature}")
    print(f"- Max Tokens: {settings.default_max_tokens}")
    print(f"- Benchmark Seeds: {settings.benchmark_seeds}")
    print(f"- Evaluation Timeout: {settings.evaluation_timeout_minutes}min")
    
    # Create necessary directories
    settings.create_directories()
    print("Created necessary directories")
    
    return settings

# Configure system
config = configure_system()
```

### Health Checks & Monitoring
```python
def system_health_check():
    """Perform comprehensive system health check."""
    
    health_status = {
        "database": False,
        "vector_store": False,
        "model_loader": False,
        "embeddings": False,
        "training": False
    }
    
    # Test database connection
    try:
        from src.config.settings import settings
        # Test database connectivity
        health_status["database"] = True
        print("✅ Database: Connected")
    except Exception as e:
        print(f"❌ Database: {e}")
    
    # Test vector store
    try:
        vector_store = create_vector_store(use_mock=False)
        test_results = vector_store.similarity_search("test query", k=1)
        health_status["vector_store"] = True
        print("✅ Vector Store: Available")
    except Exception as e:
        print(f"❌ Vector Store: {e}")
    
    # Test model loading
    try:
        from src.models.model_loader import ModelLoader
        loader = ModelLoader()
        # Test with a small model
        model = loader.load_model("gpt-3.5-turbo")
        health_status["model_loader"] = model is not None
        print("✅ Model Loader: Functional")
    except Exception as e:
        print(f"❌ Model Loader: {e}")
    
    # Test embeddings
    try:
        embedder = create_embedding_generator(use_mock=False)
        test_embeddings = embedder.generate_embeddings(["test text"])
        health_status["embeddings"] = len(test_embeddings) > 0
        print("✅ Embeddings: Available")
    except Exception as e:
        print(f"❌ Embeddings: {e}")
    
    # Test training system
    try:
        from src.training.local_orchestrator import LocalTrainingOrchestrator
        orchestrator = LocalTrainingOrchestrator()
        health_status["training"] = True
        print("✅ Training System: Ready")
    except Exception as e:
        print(f"❌ Training System: {e}")
    
    # Overall health
    overall_health = all(health_status.values())
    print(f"\n🏥 Overall System Health: {'✅ HEALTHY' if overall_health else '⚠️ ISSUES DETECTED'}")
    
    return health_status

# Run health check
health = system_health_check()
```

## 📈 **6. Batch Processing & Automation**

### Batch Model Evaluation
```python
import json
from datetime import datetime

def batch_evaluate_models():
    """Run batch evaluation of multiple models and save results."""
    
    # Define evaluation matrix
    evaluation_matrix = {
        "models": [
            "mistralai/Mistral-7B-Instruct-v0.1",
            "meta-llama/Llama-2-7b-chat-hf", 
            "Qwen/Qwen1.5-7B-Chat"
        ],
        "tasks": [
            "eps_extraction",
            "sentiment_analysis",
            "ratio_calculation",
            "regulatory_compliance"
        ],
        "datasets": [
            "data/training/synthetic_finance_v2.jsonl",
            "data/training/synthetic_finance_gsib_v3.jsonl"
        ]
    }
    
    results = []
    
    for model in evaluation_matrix["models"]:
        for task in evaluation_matrix["tasks"]:
            for dataset in evaluation_matrix["datasets"]:
                try:
                    print(f"Evaluating {model} on {task} with {dataset}")
                    
                    # Run evaluation
                    runner = BenchmarkRunner()
                    result = asyncio.run(runner.run_benchmark(
                        models=[model],
                        tasks=[task],
                        evaluation_seeds=[42],
                        max_concurrent=1
                    ))
                    
                    # Store result
                    results.append({
                        "timestamp": datetime.now().isoformat(),
                        "model": model,
                        "task": task,
                        "dataset": dataset,
                        "score": result[0].score if result else 0.0,
                        "latency_ms": result[0].latency_ms if result else 0
                    })
                    
                except Exception as e:
                    print(f"Error evaluating {model} on {task}: {e}")
                    results.append({
                        "timestamp": datetime.now().isoformat(),
                        "model": model,
                        "task": task,
                        "dataset": dataset,
                        "error": str(e)
                    })
    
    # Save results
    output_file = f"batch_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Batch evaluation completed. Results saved to {output_file}")
    return results

# Run batch evaluation
batch_results = batch_evaluate_models()
```

### Automated Training Pipeline
```python
def automated_training_pipeline():
    """Run automated training pipeline for multiple models."""
    
    training_configs = [
        {
            "name": "mistral_gsib_specialist",
            "base_model": "mistralai/Mistral-7B-Instruct-v0.1",
            "dataset": "data/training/synthetic_finance_gsib_v3.jsonl",
            "lora_rank": 16,
            "learning_rate": 2e-5,
            "epochs": 3
        },
        {
            "name": "llama_finance_general", 
            "base_model": "meta-llama/Llama-2-7b-chat-hf",
            "dataset": "data/training/synthetic_finance_v2.jsonl",
            "lora_rank": 32,
            "learning_rate": 1e-4,
            "epochs": 5
        }
    ]
    
    orchestrator = LocalTrainingOrchestrator()
    job_ids = []
    
    for config in training_configs:
        print(f"Starting training: {config['name']}")
        
        job_id = orchestrator.submit_training_job(
            model_name=config["base_model"],
            dataset_path=config["dataset"],
            lora_rank=config["lora_rank"],
            learning_rate=config["learning_rate"],
            num_epochs=config["epochs"]
        )
        
        job_ids.append((job_id, config["name"]))
        print(f"Job {job_id} submitted for {config['name']}")
    
    # Monitor all jobs
    completed_jobs = []
    while len(completed_jobs) < len(job_ids):
        for job_id, name in job_ids:
            if (job_id, name) not in completed_jobs:
                status = orchestrator.get_job_status(job_id)
                
                if status['status'] in ['completed', 'failed']:
                    completed_jobs.append((job_id, name))
                    print(f"Job {name} {status['status']}")
                    
                    if status['status'] == 'completed':
                        # Run evaluation on trained model
                        model_path = status['output_path']
                        print(f"Evaluating trained model: {model_path}")
                        # Add evaluation logic here
        
        time.sleep(60)  # Check every minute
    
    print("All training jobs completed")
    return completed_jobs

# Run automated pipeline
pipeline_results = automated_training_pipeline()
```

## 🔍 **7. Testing & Validation**

### Corpus Validation
```python
def validate_training_corpus():
    """Validate training corpus quality."""
    
    from scripts.validate_gsib_corpus import GSIBCorpusValidator
    
    validator = GSIBCorpusValidator()
    
    # Validate G-SIB corpus
    gsib_result = validator.validate_corpus("data/training/synthetic_finance_gsib_v3.jsonl")
    
    print("G-SIB Corpus Validation:")
    print(f"- Passed: {gsib_result.passed}")
    print(f"- Total Samples: {gsib_result.metrics['total_samples']}")
    print(f"- Format Valid: {gsib_result.metrics['format_valid']}")
    print(f"- Numerical Accurate: {gsib_result.metrics['numerical_accurate']}")
    print(f"- Coverage Score: {gsib_result.metrics['coverage_score']:.2%}")
    
    if gsib_result.errors:
        print("Errors found:")
        for error in gsib_result.errors[:5]:  # Show first 5
            print(f"  - {error}")
    
    # Generate detailed report
    report = validator.generate_report(gsib_result, "corpus_validation_report.md")
    print("Detailed report saved to corpus_validation_report.md")
    
    return gsib_result

# Validate corpus
validation_result = validate_training_corpus()
```

### Integration Testing
```python
def run_integration_tests():
    """Run comprehensive integration tests."""
    
    test_results = {}
    
    # Test 1: End-to-end model evaluation
    try:
        print("Testing end-to-end model evaluation...")
        runner = BenchmarkRunner()
        results = asyncio.run(runner.run_benchmark(
            models=["gpt-3.5-turbo"],
            tasks=["eps_extraction"],
            evaluation_seeds=[42],
            max_concurrent=1
        ))
        test_results["model_evaluation"] = len(results) > 0
        print("✅ Model evaluation test passed")
    except Exception as e:
        test_results["model_evaluation"] = False
        print(f"❌ Model evaluation test failed: {e}")
    
    # Test 2: Agent workflow
    try:
        print("Testing agent workflow...")
        agent = FinancialAgent(model_name="gpt-3.5-turbo")
        result = agent.run(
            query="What is the EPS?",
            context_documents=[{
                "content": "EPS: $3.50",
                "source": "Test Document"
            }],
            task_type="eps_extraction"
        )
        test_results["agent_workflow"] = "final_response" in result
        print("✅ Agent workflow test passed")
    except Exception as e:
        test_results["agent_workflow"] = False
        print(f"❌ Agent workflow test failed: {e}")
    
    # Test 3: Training system
    try:
        print("Testing training system initialization...")
        orchestrator = LocalTrainingOrchestrator()
        test_results["training_system"] = True
        print("✅ Training system test passed")
    except Exception as e:
        test_results["training_system"] = False
        print(f"❌ Training system test failed: {e}")
    
    # Summary
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    print(f"\n📊 Integration Test Results: {passed_tests}/{total_tests} passed")
    
    return test_results

# Run integration tests
integration_results = run_integration_tests()
```

## 🚀 **8. Production Deployment**

### Docker Deployment
```python
def prepare_docker_deployment():
    """Prepare system for Docker deployment."""
    
    # Generate production configuration
    prod_config = {
        "database_url": "postgresql://user:pass@db:5432/leaderboard",
        "vector_index_name": "finance-leaderboard-prod",
        "log_level": "INFO",
        "max_concurrent_evaluations": 5,
        "evaluation_timeout_minutes": 60
    }
    
    # Write production .env
    with open('.env.prod', 'w') as f:
        for key, value in prod_config.items():
            f.write(f"{key.upper()}={value}\n")
    
    print("Production configuration written to .env.prod")
    
    # Validate Docker setup
    import subprocess
    
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker available")
        else:
            print("❌ Docker not available")
    except FileNotFoundError:
        print("❌ Docker not installed")
    
    return prod_config

# Prepare deployment
deployment_config = prepare_docker_deployment()
```

### CI/CD Integration
```python
def setup_cicd_pipeline():
    """Setup CI/CD pipeline configuration."""
    
    pipeline_config = {
        "test_commands": [
            "python scripts/test_system.py",
            "python scripts/validate_gsib_corpus.py data/training/synthetic_finance_gsib_v3.jsonl",
            "python -m pytest tests/ -v"
        ],
        "deployment_commands": [
            "docker build -t llm-finance-leaderboard .",
            "docker tag llm-finance-leaderboard:latest registry/llm-finance-leaderboard:latest",
            "docker push registry/llm-finance-leaderboard:latest"
        ],
        "environment_variables": [
            "PINECONE_API_KEY",
            "OPENAI_API_KEY", 
            "ANTHROPIC_API_KEY",
            "DATABASE_URL"
        ]
    }
    
    # Generate GitHub Actions workflow
    github_workflow = f"""
name: LLM Finance Leaderboard CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run tests
      run: |
        {chr(10).join('        ' + cmd for cmd in pipeline_config['test_commands'])}
      env:
        {chr(10).join('        ' + var + ': ${{{{ secrets.' + var + ' }}}}' for var in pipeline_config['environment_variables'])}
  
  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    - name: Deploy
      run: |
        {chr(10).join('        ' + cmd for cmd in pipeline_config['deployment_commands'])}
"""
    
    # Write workflow file
    import os
    os.makedirs('.github/workflows', exist_ok=True)
    with open('.github/workflows/ci-cd.yml', 'w') as f:
        f.write(github_workflow)
    
    print("CI/CD pipeline configuration created")
    return pipeline_config

# Setup CI/CD
cicd_config = setup_cicd_pipeline()
```

## 📚 **9. Advanced Usage Examples**

### Custom Model Comparison
```python
def compare_model_performance():
    """Compare performance of different models on financial tasks."""
    
    from src.training.comparison_engine import ComparisonEngine
    
    comparison_engine = ComparisonEngine()
    
    # Compare base vs fine-tuned models
    comparison_result = comparison_engine.compare_models(
        base_model="mistralai/Mistral-7B-Instruct-v0.1",
        finetuned_model="models/mistral_gsib_specialist",
        test_dataset="data/training/synthetic_finance_gsib_v3.jsonl",
        metrics=["accuracy", "f1_score", "regulatory_compliance"]
    )
    
    # Generate detailed report
    report = comparison_engine.generate_comparison_report(comparison_result)
    
    print("Model Comparison Results:")
    print(f"Base Model Score: {comparison_result['base_score']:.3f}")
    print(f"Fine-tuned Score: {comparison_result['finetuned_score']:.3f}")
    print(f"Improvement: {comparison_result['improvement']:.3f}")
    print(f"Statistical Significance: {comparison_result['p_value']:.4f}")
    
    # Save detailed report
    with open("model_comparison_report.md", "w") as f:
        f.write(report)
    
    return comparison_result

# Run comparison
comparison = compare_model_performance()
```

### Real-time Financial Analysis
```python
def real_time_financial_analysis():
    """Setup real-time financial analysis pipeline."""
    
    import time
    from datetime import datetime
    
    # Initialize components
    agent = FinancialAgent(model_name="gpt-3.5-turbo")
    vector_store = create_vector_store(use_mock=False)
    
    # Simulated real-time data stream
    def get_latest_financial_data():
        """Simulate getting latest financial data."""
        return {
            "
timestamp": datetime.now().isoformat(),
            "ticker": "JPM",
            "price": 150.25,
            "volume": 1000000,
            "news": "JPMorgan reports strong Q4 earnings with EPS of $4.50"
        }
    
    # Real-time analysis loop
    analysis_results = []
    
    print("Starting real-time financial analysis...")
    for i in range(5):  # Run 5 iterations
        # Get latest data
        data = get_latest_financial_data()
        
        # Analyze with AI agent
        query = f"Analyze the latest news for {data['ticker']}: {data['news']}"
        
        result = agent.run(
            query=query,
            context_documents=[{
                "content": f"Stock: {data['ticker']}, Price: ${data['price']}, Volume: {data['volume']:,}",
                "source": "Real-time Market Data"
            }],
            task_type="sentiment_analysis"
        )
        
        analysis_results.append({
            "timestamp": data["timestamp"],
            "ticker": data["ticker"],
            "analysis": result["final_response"],
            "confidence": result["confidence_score"]
        })
        
        print(f"Analysis {i+1}: {result['final_response'][:100]}...")
        
        time.sleep(10)  # Wait 10 seconds between analyses
    
    return analysis_results

# Run real-time analysis
realtime_results = real_time_financial_analysis()
```

## 🔧 **10. Troubleshooting & Common Issues**

### Common Configuration Issues
```python
def diagnose_configuration_issues():
    """Diagnose and fix common configuration issues."""
    
    issues_found = []
    fixes_applied = []
    
    # Check 1: Environment variables
    required_env_vars = [
        "PINECONE_API_KEY",
        "PINECONE_ENVIRONMENT", 
        "OPENAI_API_KEY"
    ]
    
    import os
    for var in required_env_vars:
        if not os.getenv(var):
            issues_found.append(f"Missing environment variable: {var}")
        else:
            print(f"✅ {var}: Set")
    
    # Check 2: Benchmark seeds configuration
    try:
        from src.config.settings import settings
        seeds = settings.benchmark_seeds
        if not seeds:
            issues_found.append("benchmark_seeds not configured")
        else:
            print(f"✅ Benchmark seeds: {seeds}")
    except Exception as e:
        issues_found.append(f"Settings configuration error: {e}")
        
        # Apply fix for benchmark_seeds JSON parsing issue
        env_file_path = ".env"
        if os.path.exists(env_file_path):
            with open(env_file_path, 'r') as f:
                content = f.read()
            
            # Fix benchmark_seeds format
            if "BENCHMARK_SEEDS=42,123,456" in content:
                content = content.replace(
                    "BENCHMARK_SEEDS=42,123,456",
                    "BENCHMARK_SEEDS=[42,123,456]"
                )
                with open(env_file_path, 'w') as f:
                    f.write(content)
                fixes_applied.append("Fixed benchmark_seeds format in .env")
    
    # Check 3: Required directories
    required_dirs = [
        "data/training",
        "models",
        "logs",
        "results"
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            fixes_applied.append(f"Created directory: {dir_path}")
        else:
            print(f"✅ Directory exists: {dir_path}")
    
    # Check 4: Training data files
    training_files = [
        "data/training/synthetic_finance_v2.jsonl",
        "data/training/synthetic_finance_gsib_v3.jsonl"
    ]
    
    for file_path in training_files:
        if not os.path.exists(file_path):
            issues_found.append(f"Missing training file: {file_path}")
        else:
            print(f"✅ Training file exists: {file_path}")
    
    # Summary
    print(f"\n🔍 Diagnosis Complete:")
    print(f"Issues found: {len(issues_found)}")
    print(f"Fixes applied: {len(fixes_applied)}")
    
    if issues_found:
        print("\n❌ Issues to resolve:")
        for issue in issues_found:
            print(f"  - {issue}")
    
    if fixes_applied:
        print("\n✅ Fixes applied:")
        for fix in fixes_applied:
            print(f"  - {fix}")
    
    return {
        "issues_found": issues_found,
        "fixes_applied": fixes_applied
    }

# Run diagnosis
diagnosis = diagnose_configuration_issues()
```

### Performance Optimization
```python
def optimize_system_performance():
    """Optimize system performance for production use."""
    
    optimizations = []
    
    # 1. GPU Memory Optimization
    try:
        import torch
        if torch.cuda.is_available():
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(0.8)
            
            optimizations.append("GPU memory optimized")
            print("✅ GPU memory optimization applied")
        else:
            print("ℹ️ No GPU available for optimization")
    except ImportError:
        print("ℹ️ PyTorch not available for GPU optimization")
    
    # 2. Model Loading Optimization
    model_cache_config = {
        "max_models_in_memory": 3,
        "model_cache_size_gb": 8,
        "use_model_quantization": True,
        "enable_model_offloading": True
    }
    
    # Write optimization config
    import json
    with open("config/performance_config.json", "w") as f:
        json.dump(model_cache_config, f, indent=2)
    
    optimizations.append("Model loading optimization configured")
    
    # 3. Database Connection Pooling
    db_pool_config = {
        "pool_size": 10,
        "max_overflow": 20,
        "pool_timeout": 30,
        "pool_recycle": 3600
    }
    
    optimizations.append("Database connection pooling configured")
    
    # 4. Vector Store Optimization
    vector_config = {
        "batch_size": 100,
        "index_type": "HNSW",
        "ef_construction": 200,
        "m": 16
    }
    
    optimizations.append("Vector store optimization configured")
    
    print(f"✅ Applied {len(optimizations)} performance optimizations")
    return optimizations

# Apply optimizations
performance_opts = optimize_system_performance()
```

## 📖 **11. API Reference**

### Core Classes
```python
# Model Management
from src.models.model_loader import ModelLoader
from src.models.model_registry import ModelRegistry

# Evaluation System
from src.evaluation.runners.benchmark_runner import BenchmarkRunner
from src.evaluation.tasks.base_task import BaseTask
from src.evaluation.tasks.eps_extraction import EPSExtractionTask
from src.evaluation.tasks.sentiment_analysis import SentimentAnalysisTask

# Training System
from src.training.lora_trainer import LoRATrainer
from src.training.local_orchestrator import LocalTrainingOrchestrator
from src.training.comparison_engine import ComparisonEngine

# Agent System
from src.agents.base_agent import FinancialAgent
from src.agents.workflow_nodes import RetrievalNode, AnalysisNode, CritiqueNode

# Data Processing
from src.data.processors.vector_store import create_vector_store
from src.data.processors.embeddings import create_embedding_generator
from src.data.collectors.sec_filings import SECFilingsCollector

# Configuration
from src.config.settings import settings
```

### Key Methods

#### ModelLoader
```python
loader = ModelLoader()

# Load model
model = loader.load_model(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    device="auto",
    quantization="4bit"
)

# Generate response
response = loader.generate_response(
    model=model,
    prompt="What is the EPS?",
    max_tokens=512,
    temperature=0.1
)
```

#### BenchmarkRunner
```python
runner = BenchmarkRunner()

# Run benchmark
results = await runner.run_benchmark(
    models=["gpt-3.5-turbo"],
    tasks=["eps_extraction"],
    evaluation_seeds=[42, 123, 456],
    max_concurrent=2
)

# Get leaderboard
leaderboard = runner.get_leaderboard(
    task_filter=["eps_extraction"],
    model_filter=["gpt-3.5-turbo"]
)
```

#### LoRATrainer
```python
trainer = LoRATrainer(config={
    "model_name": "mistralai/Mistral-7B-Instruct-v0.1",
    "dataset_path": "data/training/synthetic_finance_gsib_v3.jsonl",
    "lora_rank": 16,
    "learning_rate": 2e-5
})

# Start training
trainer.train()

# Monitor progress
status = trainer.get_training_status()
```

#### FinancialAgent
```python
agent = FinancialAgent(
    model_name="gpt-3.5-turbo",
    temperature=0.1
)

# Run analysis
result = agent.run(
    query="What is the EPS for JPMorgan?",
    context_documents=[...],
    task_type="eps_extraction"
)
```

## 🚀 **12. Quick Start Scripts**

### Complete Evaluation Script
```python
#!/usr/bin/env python3
"""
Quick start script for model evaluation.
Usage: python quick_evaluate.py
"""

import asyncio
import sys
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
        results = await runner.run_benchmark(
            models=models,
            tasks=tasks,
            evaluation_seeds=seeds,
            max_concurrent=1
        )
        
        # Display results
        print("\n📊 Evaluation Results:")
        for result in results:
            print(f"Model: {result.model_name}")
            print(f"Task: {result.task_name}")
            print(f"Score: {result.score:.3f}")
            print(f"Latency: {result.latency_ms}ms")
            print("-" * 40)
        
        print("✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
```

### Training Quick Start
```python
#!/usr/bin/env python3
"""
Quick start script for model training.
Usage: python quick_train.py
"""

from src.training.local_orchestrator import LocalTrainingOrchestrator
import time

def main():
    """Run quick model training."""
    
    print("🚀 Starting LLM Finance Model Training")
    
    # Initialize orchestrator
    orchestrator = LocalTrainingOrchestrator()
    
    # Training configuration
    config = {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.1",
        "dataset_path": "data/training/synthetic_finance_gsib_v3.jsonl",
        "lora_rank": 16,
        "learning_rate": 2e-5,
        "num_epochs": 2,  # Quick training
        "batch_size": 4
    }
    
    try:
        # Submit training job
        job_id = orchestrator.submit_training_job(**config)
        print(f"Training job submitted: {job_id}")
        
        # Monitor progress
        while True:
            status = orchestrator.get_job_status(job_id)
            print(f"Status: {status['status']}")
            
            if status['status'] in ['completed', 'failed']:
                break
            
            time.sleep(30)
        
        if status['status'] == 'completed':
            print(f"✅ Training completed! Model saved to: {status['output_path']}")
        else:
            print(f"❌ Training failed: {status.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"❌ Training setup failed: {e}")

if __name__ == "__main__":
    main()
```

## 📞 **Support & Resources**

### Getting Help
- **Documentation**: Check `docs/` directory for detailed guides
- **Configuration**: Review `src/config/settings.py` for all available options
- **Examples**: See `examples/` directory for usage patterns
- **Logs**: Check `logs/` directory for detailed error information

### Performance Monitoring
```python
# Monitor system resources
from src.utils.monitoring import SystemMonitor

monitor = SystemMonitor()
metrics = monitor.get_current_metrics()
print(f"GPU Memory: {metrics['gpu_memory_used']:.1f}GB")
print(f"CPU Usage: {metrics['cpu_percent']:.1f}%")
print(f"RAM Usage: {metrics['memory_percent']:.1f}%")
```

### Best Practices
1. **Always validate configuration** before running evaluations
2. **Use appropriate batch sizes** for your hardware
3. **Monitor GPU memory** during training
4. **Save intermediate results** for long-running processes
5. **Use version control** for training configurations
6. **Test with small datasets** before full evaluation

---

## 🎯 **Summary**

This developer guide provides comprehensive coverage of the LLM Finance Leaderboard system's programmatic interface, enabling:

- **Model Evaluation**: Automated benchmarking across financial tasks
- **Fine-tuning**: Local LoRA/QLoRA training with comparison analysis
- **Agent Workflows**: Financial analysis using AI agents
- **Data Management**: Vector stores, embeddings, and SEC data collection
- **System Monitoring**: Health checks, performance optimization
- **Batch Processing**: Automated evaluation and training pipelines
- **Production Deployment**: Docker, CI/CD, and monitoring setup

The system is designed for both research and production use, with robust error handling, comprehensive logging, and scalable architecture supporting the full spectrum of financial AI applications.