# Model Comparison Quick Start Guide

This guide provides step-by-step instructions for technical stakeholders to quickly start comparing LLM models on financial tasks.

**Key Insight**: You only need **ONE** HuggingFace token to compare thousands of open source models. See [HuggingFace Multi-Model Guide](HUGGINGFACE_MULTI_MODEL_GUIDE.md) for details.

## 🚀 5-Minute Quick Start

### Step 1: Setup Environment (2 minutes)
```bash
# Clone and install
git clone <repository-url>
cd llm-finance-leaderboard
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
```

### Step 2: Configure APIs (2 minutes)
Edit `.env` file with your API keys:
```bash
# REQUIRED - Get from https://www.pinecone.io/
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here

# OPTIONAL - For API models
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HUGGINGFACE_API_TOKEN=your_huggingface_token_here  # ONE token for 7,000+ models
```

### Step 3: Run First Comparison (1 minute)
```bash
# Test with mock data (no API keys needed)
python scripts/run_benchmark.py --verbose

# Or with real APIs
python scripts/run_benchmark.py --use-real-runner --verbose
```

## 📊 Model Comparison Scenarios

### Scenario 1: API Model Comparison
**Compare OpenAI vs Anthropic models:**

```bash
python scripts/run_benchmark.py \
  --models gpt35_turbo claude3_haiku \
  --tasks eps_extraction \
  --name "API Model Comparison" \
  --use-real-runner
```

**Expected Output:**
```
🚀 LLM Finance Leaderboard - Benchmark Runner
📊 Models: gpt35_turbo, claude3_haiku
📋 Tasks: eps_extraction
📝 Run Name: API Model Comparison

✅ Benchmark completed successfully!
🆔 Run ID: abc123...
⏱️  Duration: 2.34 minutes
📈 Models Evaluated: 2
📋 Tasks Completed: 1
```

### Scenario 2: Open Source Model Comparison
**Compare HuggingFace models:**

First, add models to [`src/config/models_config.yaml`](src/config/models_config.yaml):
```yaml
stock_models:
  mistral_7b:
    name: "mistralai/Mistral-7B-Instruct-v0.1"
    display_name: "Mistral 7B Instruct"
    parameters: "7B"
    quantization: "4bit"
    context_length: 8192
    cost_per_1k_tokens: 0.0
    provider: "huggingface"
    
  llama2_7b:
    name: "meta-llama/Llama-2-7b-chat-hf"
    display_name: "Llama 2 7B Chat"
    parameters: "7B"
    quantization: "4bit"
    context_length: 4096
    cost_per_1k_tokens: 0.0
    provider: "huggingface"
```

Then run comparison:
```bash
python scripts/run_benchmark.py \
  --models mistral_7b llama2_7b \
  --tasks eps_extraction \
  --name "Open Source 7B Comparison" \
  --use-real-runner
```

### Scenario 3: Mixed Model Comparison
**Compare API vs Open Source:**

```bash
python scripts/run_benchmark.py \
  --models gpt35_turbo mistral_7b finma_7b \
  --tasks eps_extraction \
  --name "API vs Open Source" \
  --use-real-runner
```

### Scenario 4: Finance-Tuned Model Evaluation
**Test finance-specific models:**

```bash
python scripts/run_benchmark.py \
  --models finma_7b fingpt_llama2_13b \
  --tasks eps_extraction \
  --name "Finance-Tuned Models" \
  --use-real-runner
```

## 🎯 Understanding Results

### Dashboard Analysis
Launch the dashboard to view results:
```bash
streamlit run streamlit_app/main.py
```

### Key Metrics to Compare

#### 1. **Final Score** (0.0 - 1.0)
- Composite score combining quality and efficiency
- Higher is better
- Formula: `0.9 × Quality + 0.1 × Efficiency`

#### 2. **Quality Scores**
- **F1 Score**: Token overlap accuracy
- **ROUGE-1/2**: Text similarity metrics
- **Exact Match**: Perfect answer matching
- **Numerical Accuracy**: Financial number precision

#### 3. **Efficiency Metrics**
- **Latency**: Response time in milliseconds
- **Cost per Task**: API cost or compute cost
- **Tokens Used**: Input/output token count

#### 4. **Success Metrics**
- **Completion Rate**: % of tasks completed successfully
- **Hallucination Rate**: % of responses with detected hallucinations

### Sample Results Interpretation

```
Model Comparison Results:
┌─────────────────┬─────────────┬─────────────┬──────────────┬─────────────┐
│ Model           │ Final Score │ Quality     │ Latency (ms) │ Cost/Task   │
├─────────────────┼─────────────┼─────────────┼──────────────┼─────────────┤
│ GPT-3.5 Turbo   │ 0.847       │ 0.891       │ 1,234        │ $0.0023     │
│ Claude 3 Haiku  │ 0.832       │ 0.876       │ 1,567        │ $0.0018     │
│ Mistral 7B      │ 0.798       │ 0.834       │ 2,890        │ $0.0000     │
│ FinMA 7B        │ 0.819       │ 0.863       │ 2,456        │ $0.0000     │
└─────────────────┴─────────────┴─────────────┴──────────────┴─────────────┘
```

**Analysis:**
- **GPT-3.5 Turbo**: Highest quality, fast, but costs money
- **Claude 3 Haiku**: Good balance of quality and cost
- **FinMA 7B**: Best open-source option for finance
- **Mistral 7B**: Slower but free, good baseline

## 🔧 Advanced Comparison Workflows

### Custom Evaluation Script

Create [`scripts/my_comparison.py`](scripts/my_comparison.py):
```python
#!/usr/bin/env python3
import asyncio
from src.evaluation.runners.benchmark_runner import create_benchmark_runner

async def compare_models():
    """Custom model comparison."""
    runner = create_benchmark_runner()
    
    # Define comparison
    models = ["gpt-3.5-turbo", "claude-3-haiku", "mistral-7b"]
    tasks = ["eps_extraction"]
    
    print("🚀 Starting custom model comparison...")
    
    result = await runner.run_benchmark(
        models=models,
        tasks=tasks,
        run_name="Custom Comparison",
        description="Comparing top 3 models for EPS extraction"
    )
    
    print(f"\n📊 Results Summary:")
    print(f"Run ID: {result.run_id}")
    print(f"Duration: {result.total_duration_minutes:.2f} minutes")
    
    # Print model rankings
    rankings = sorted(
        result.model_metrics.items(),
        key=lambda x: x[1].final_score,
        reverse=True
    )
    
    print(f"\n🏆 Model Rankings:")
    for i, (model_name, metrics) in enumerate(rankings, 1):
        print(f"{i}. {model_name}: {metrics.final_score:.3f}")
        print(f"   Quality: {metrics.overall_quality_score:.3f}")
        print(f"   Latency: {metrics.avg_latency_ms:.0f}ms")
        print(f"   Cost: ${metrics.avg_cost_per_task:.4f}")
        print()

if __name__ == "__main__":
    asyncio.run(compare_models())
```

Run it:
```bash
python scripts/my_comparison.py
```

### Batch Model Testing

Create [`scripts/batch_test.py`](scripts/batch_test.py):
```python
#!/usr/bin/env python3
import asyncio
from src.evaluation.runners.benchmark_runner import create_benchmark_runner

async def batch_test_models():
    """Test multiple model configurations."""
    runner = create_benchmark_runner()
    
    # Test configurations
    test_configs = [
        {
            "name": "API Models",
            "models": ["gpt-3.5-turbo", "claude-3-haiku"],
            "description": "Testing API-based models"
        },
        {
            "name": "7B Open Source",
            "models": ["mistral-7b", "llama2-7b"],
            "description": "Testing 7B parameter open source models"
        },
        {
            "name": "Finance Tuned",
            "models": ["finma-7b", "fingpt-llama2-13b"],
            "description": "Testing finance-specific models"
        }
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n🧪 Testing: {config['name']}")
        
        result = await runner.run_benchmark(
            models=config["models"],
            tasks=["eps_extraction"],
            run_name=config["name"],
            description=config["description"]
        )
        
        results.append(result)
        
        # Print quick summary
        best_model = max(
            result.model_metrics.items(),
            key=lambda x: x[1].final_score
        )
        print(f"✅ Best in category: {best_model[0]} ({best_model[1].final_score:.3f})")
    
    return results

if __name__ == "__main__":
    asyncio.run(batch_test_models())
```

### Performance Monitoring

Create [`scripts/monitor_performance.py`](scripts/monitor_performance.py):
```python
#!/usr/bin/env python3
import time
import psutil
import GPUtil
from src.models.model_loader import get_model_loader

def monitor_model_loading():
    """Monitor system resources during model loading."""
    loader = get_model_loader()
    
    print("📊 System Resources Before Loading:")
    print(f"CPU: {psutil.cpu_percent()}%")
    print(f"RAM: {psutil.virtual_memory().percent}%")
    
    if GPUtil.getGPUs():
        gpu = GPUtil.getGPUs()[0]
        print(f"GPU: {gpu.memoryUtil*100:.1f}%")
    
    print("\n🔄 Loading model...")
    start_time = time.time()
    
    # Load model (replace with your model)
    model = loader.load_model("mistral-7b")
    
    load_time = time.time() - start_time
    
    print(f"\n📊 System Resources After Loading:")
    print(f"CPU: {psutil.cpu_percent()}%")
    print(f"RAM: {psutil.virtual_memory().percent}%")
    print(f"Load Time: {load_time:.2f}s")
    
    if GPUtil.getGPUs():
        gpu = GPUtil.getGPUs()[0]
        print(f"GPU: {gpu.memoryUtil*100:.1f}%")

if __name__ == "__main__":
    monitor_model_loading()
```

## 🎯 Best Practices

### 1. Start Small
- Begin with 1-2 models
- Use single task (eps_extraction)
- Test with mock data first

### 2. Gradual Scaling
- Add more models incrementally
- Monitor system resources
- Use quantization for large models

### 3. Cost Management
- Start with free models (HuggingFace)
- Monitor API costs carefully
- Use batch processing for efficiency

### 4. Quality Validation
- Always review sample outputs
- Check for hallucinations
- Validate financial accuracy

### 5. Documentation
- Document model configurations
- Track evaluation results
- Note any issues or observations

## 🔍 Troubleshooting Quick Fixes

### Model Won't Load
```bash
# Check model name
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('model-name')"

# Check GPU memory
python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')"
```

### API Errors
```bash
# Test API connection
python -c "from openai import OpenAI; client = OpenAI(); print('OpenAI OK')"
python -c "from anthropic import Anthropic; client = Anthropic(); print('Anthropic OK')"
```

### Performance Issues
```bash
# Reduce batch size
export MAX_CONCURRENT_EVALUATIONS=1

# Use quantization
# Edit model config: quantization: "4bit"

# Monitor resources
htop  # CPU/RAM
nvidia-smi  # GPU
```

---

**Ready to start comparing models?**

1. ✅ Setup environment and APIs
2. ✅ Run your first comparison
3. ✅ Analyze results in dashboard
4. ✅ Scale up with more models/tasks
5. ✅ Document findings and optimize