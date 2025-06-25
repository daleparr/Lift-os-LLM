# LLM Finance Leaderboard - Quick Reference

## 🚀 **Essential Commands**

### Setup & Configuration
```bash
# Environment setup
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys

# Validate system
python scripts/diagnose_system.py
```

### Model Evaluation (5 minutes)
```python
import asyncio
from src.evaluation.runners.benchmark_runner import BenchmarkRunner

# Quick evaluation
runner = BenchmarkRunner()
results = await runner.run_benchmark(
    models=["gpt-3.5-turbo"],
    tasks=["eps_extraction"],
    evaluation_seeds=[42]
)
```

### Fine-tuning (30 minutes)
```python
from src.training.local_orchestrator import LocalTrainingOrchestrator

orchestrator = LocalTrainingOrchestrator()
job_id = orchestrator.submit_training_job(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    dataset_path="data/training/synthetic_finance_gsib_v3.jsonl",
    lora_rank=16,
    num_epochs=2
)
```

### Financial Analysis
```python
from src.agents.base_agent import FinancialAgent

agent = FinancialAgent(model_name="gpt-3.5-turbo")
result = agent.run(
    query="What is the EPS for JPMorgan?",
    context_documents=[{"content": "JPM EPS: $4.44", "source": "10-Q"}],
    task_type="eps_extraction"
)
```

## 📊 **Common Use Cases**

### 1. Compare Models
```python
# Batch evaluation
models = ["gpt-3.5-turbo", "mistralai/Mistral-7B-Instruct-v0.1"]
tasks = ["eps_extraction", "sentiment_analysis"]
results = await runner.run_benchmark(models=models, tasks=tasks)
```

### 2. Train Custom Model
```python
# Submit training job
job_id = orchestrator.submit_training_job(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    dataset_path="data/training/custom_data.jsonl",
    lora_rank=32,
    learning_rate=1e-4
)

# Monitor progress
status = orchestrator.get_job_status(job_id)
```

### 3. Process Financial Documents
```python
from src.data.processors.vector_store import create_vector_store

vector_store = create_vector_store(use_mock=False)
vector_store.add_document(
    doc_id="jpm_q3_2024",
    content="JPMorgan Q3 2024: EPS $4.44...",
    metadata={"ticker": "JPM", "quarter": "Q3"}
)
```

## 🔧 **Troubleshooting**

### Common Issues
```python
# Fix configuration
from src.config.settings import settings
settings.create_directories()

# Health check
def quick_health_check():
    try:
        from src.models.model_loader import ModelLoader
        loader = ModelLoader()
        model = loader.load_model("gpt-3.5-turbo")
        return "✅ System healthy"
    except Exception as e:
        return f"❌ Issue: {e}"
```

### Environment Variables
```bash
# Required
PINECONE_API_KEY=your_key
OPENAI_API_KEY=your_key

# Fix benchmark_seeds format
BENCHMARK_SEEDS=[42,123,456]  # JSON array format
```

## 📈 **Performance Tips**

### GPU Optimization
```python
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.8)
```

### Batch Processing
```python
# Process multiple models efficiently
for model in models:
    results = await runner.run_benchmark(
        models=[model],
        tasks=tasks,
        max_concurrent=1  # Adjust based on GPU memory
    )
```

## 🎯 **Key File Locations**

- **Configuration**: `src/config/settings.py`
- **Training Data**: `data/training/synthetic_finance_gsib_v3.jsonl`
- **Models**: `models/` (after training)
- **Logs**: `logs/`
- **Results**: `results/`

## 📞 **Quick Help**

### System Status
```python
# Check system health
python scripts/diagnose_system.py

# Validate training data
python scripts/validate_gsib_corpus.py data/training/synthetic_finance_gsib_v3.jsonl
```

### Documentation
- **Full API Guide**: [`docs/DEVELOPER_API_GUIDE.md`](DEVELOPER_API_GUIDE.md)
- **Architecture**: [`LLM_Finance_Leaderboard_Architecture.md`](../LLM_Finance_Leaderboard_Architecture.md)
- **Training Config**: [`src/config/training_config.yaml`](../src/config/training_config.yaml)

---

**Need more details?** See the complete [Developer API Guide](DEVELOPER_API_GUIDE.md) for comprehensive examples and advanced usage patterns.