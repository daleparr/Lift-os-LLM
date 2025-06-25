# LLM Finance Leaderboard - Technical Setup Guide

This guide provides detailed instructions for technical stakeholders on how to configure model APIs, add new models, and start comparing model performance.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [API Configuration](#api-configuration)
3. [Adding Model APIs](#adding-model-apis)
4. [HuggingFace Integration](#huggingface-integration)
5. [Model Comparison Workflow](#model-comparison-workflow)
6. [Advanced Configuration](#advanced-configuration)
7. [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Git
- 8GB+ RAM (16GB+ recommended for local models)
- GPU with 8GB+ VRAM (optional, for local models)

### 1. Initial Setup
```bash
# Clone repository
git clone <repository-url>
cd llm-finance-leaderboard

# Install dependencies
pip install -r requirements.txt

# Setup environment
python scripts/setup_environment.py

# Test system
python scripts/test_system.py
```

### 2. Basic Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys (see API Configuration section)
nano .env
```

### 3. Verify Installation
```bash
# Run system test
python scripts/test_system.py

# Launch dashboard
streamlit run streamlit_app/main.py
```

## 🔑 API Configuration

### Required APIs

#### 1. Pinecone (Vector Database) - **REQUIRED**
```bash
# Get API key from: https://www.pinecone.io/
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here
```

**Setup Steps:**
1. Create account at [pinecone.io](https://www.pinecone.io/)
2. Create a new index:
   - **Name**: `finance-leaderboard`
   - **Dimensions**: `384` (for sentence-transformers)
   - **Metric**: `cosine`
   - **Pod Type**: `p1.x1` (starter)

#### 2. OpenAI API - **OPTIONAL**
```bash
# Get API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here
```

**Supported Models:**
- `gpt-3.5-turbo-1106`
- `gpt-4-1106-preview`
- `gpt-4-turbo-preview`

#### 3. Anthropic API - **OPTIONAL**
```bash
# Get API key from: https://console.anthropic.com/
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

**Supported Models:**
- `claude-3-haiku-20240307`
- `claude-3-sonnet-20240229`
- `claude-3-opus-20240229`

#### 4. HuggingFace Token - **OPTIONAL** (One Token for All Open Source Models)
```bash
# Get token from: https://huggingface.co/settings/tokens
# ONE token gives access to 7,000+ open source models
HUGGINGFACE_API_TOKEN=your_huggingface_token_here
```

**Important**: You only need **ONE** HuggingFace token to access thousands of open source models (Llama, Mistral, Qwen, Phi, etc.). See [HuggingFace Multi-Model Guide](HUGGINGFACE_MULTI_MODEL_GUIDE.md) for details.

### Financial Data APIs (Optional)

#### FRED API (Economic Data)
```bash
# Get API key from: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY=your_fred_api_key_here
```

#### Alpha Vantage (Market Data)
```bash
# Get API key from: https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
```

## 🤖 Adding Model APIs

### 1. OpenAI Models

**Add to [`src/config/models_config.yaml`](src/config/models_config.yaml):**
```yaml
stock_models:
  gpt4_turbo:
    name: "gpt-4-turbo-preview"
    display_name: "GPT-4 Turbo"
    parameters: "175B"
    quantization: null
    context_length: 128000
    cost_per_1k_tokens: 0.01
    provider: "openai"
```

**Test the model:**
```bash
python scripts/run_benchmark.py --models gpt4_turbo --tasks eps_extraction --verbose
```

### 2. Anthropic Models

**Add to [`src/config/models_config.yaml`](src/config/models_config.yaml):**
```yaml
stock_models:
  claude3_sonnet:
    name: "claude-3-sonnet-20240229"
    display_name: "Claude 3 Sonnet"
    parameters: "200B"
    quantization: null
    context_length: 200000
    cost_per_1k_tokens: 0.003
    provider: "anthropic"
```

### 3. Custom API Integration

**Create custom model provider in [`src/models/model_loader.py`](src/models/model_loader.py):**

```python
def _load_custom_api_model(self, model_name: str, model_config: ModelConfig):
    """Load custom API model."""
    from langchain_community.llms import CustomLLM
    
    return CustomLLM(
        api_key=os.getenv("CUSTOM_API_KEY"),
        model_name=model_name,
        temperature=model_config.temperature,
        max_tokens=model_config.max_tokens
    )
```

## 🤗 HuggingFace Integration

### 1. Basic HuggingFace Model Setup

**Add model to [`src/config/models_config.yaml`](src/config/models_config.yaml):**
```yaml
stock_models:
  llama2_7b:
    name: "meta-llama/Llama-2-7b-chat-hf"
    display_name: "Llama 2 7B Chat"
    parameters: "7B"
    quantization: "4bit"  # Options: null, "4bit", "8bit"
    context_length: 4096
    cost_per_1k_tokens: 0.0
    provider: "huggingface"
```

### 2. GPU Requirements by Model Size

| Model Size | GPU Memory | Recommended GPU | Quantization |
|------------|------------|-----------------|--------------|
| 3B-7B      | 8-16GB     | RTX 4060 Ti     | 4bit         |
| 13B        | 16-24GB    | RTX 4090        | 4bit/8bit    |
| 30B-40B    | 40-80GB    | A100 40GB       | 4bit         |
| 70B+       | 80GB+      | A100 80GB       | 4bit         |

### 3. Quantization Options

**4-bit Quantization (Recommended):**
```yaml
quantization: "4bit"
# Reduces memory by ~75%, minimal quality loss
```

**8-bit Quantization:**
```yaml
quantization: "8bit"
# Reduces memory by ~50%, better quality than 4bit
```

**No Quantization:**
```yaml
quantization: null
# Full precision, maximum quality, highest memory usage
```

### 4. Fine-tuned Models (LoRA)

**Add LoRA model:**
```yaml
finance_tuned_models:
  custom_finance_llama:
    name: "meta-llama/Llama-2-7b-chat-hf"
    display_name: "Custom Finance Llama 7B"
    parameters: "7B"
    quantization: "4bit"
    context_length: 4096
    cost_per_1k_tokens: 0.0
    provider: "local"
    base_model: "meta-llama/Llama-2-7b-chat-hf"
    lora_path: "./models/finance_lora_adapter"
```

### 5. Local Model Files

**For local model files:**
```yaml
stock_models:
  local_model:
    name: "/path/to/local/model"
    display_name: "Local Custom Model"
    parameters: "7B"
    quantization: null
    context_length: 2048
    cost_per_1k_tokens: 0.0
    provider: "local"
```

## 📊 Model Comparison Workflow

### 1. Single Model Evaluation

```bash
# Test single model
python scripts/run_benchmark.py \
  --models mistral_7b \
  --tasks eps_extraction \
  --name "Mistral 7B Test" \
  --verbose
```

### 2. Multi-Model Comparison

```bash
# Compare multiple models
python scripts/run_benchmark.py \
  --models mistral_7b llama2_7b finma_7b \
  --tasks eps_extraction \
  --name "7B Model Comparison" \
  --use-real-runner \
  --verbose
```

### 3. Comprehensive Evaluation

```bash
# Full evaluation (when more tasks are implemented)
python scripts/run_benchmark.py \
  --models gpt35_turbo claude3_haiku mistral_7b \
  --tasks eps_extraction ratio_identification revenue_analysis \
  --name "Comprehensive Evaluation" \
  --use-real-runner
```

### 4. Custom Evaluation Script

**Create [`scripts/custom_evaluation.py`](scripts/custom_evaluation.py):**
```python
#!/usr/bin/env python3
import asyncio
from src.evaluation.runners.benchmark_runner import create_benchmark_runner

async def main():
    runner = create_benchmark_runner()
    
    # Define your models and tasks
    models = ["gpt-3.5-turbo", "claude-3-haiku", "mistral-7b"]
    tasks = ["eps_extraction"]
    
    result = await runner.run_benchmark(
        models=models,
        tasks=tasks,
        run_name="Custom Evaluation",
        description="Comparing API vs Open Source models"
    )
    
    print(f"Evaluation completed: {result.run_id}")
    
    # Print results
    for model_name, metrics in result.model_metrics.items():
        print(f"{model_name}: {metrics.final_score:.3f}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 5. Dashboard Analysis

**Launch Streamlit dashboard:**
```bash
streamlit run streamlit_app/main.py
```

**Dashboard Features:**
- **Leaderboard**: Real-time model rankings
- **Model Comparison**: Side-by-side performance analysis
- **Task Analysis**: Performance breakdown by complexity
- **Cost Analysis**: Cost per task and efficiency metrics

## ⚙️ Advanced Configuration

### 1. Custom Task Development

**Create new task in [`src/tasks/custom/`](src/tasks/custom/):**
```python
from ..base_task import FinancialExtractionTask
from ...data.schemas.data_models import TaskComplexity

class CustomTask(FinancialExtractionTask):
    def __init__(self):
        super().__init__(
            name="Custom Financial Task",
            description="Extract custom financial metrics",
            complexity=TaskComplexity.MEDIUM
        )
    
    def generate_prompt(self, context):
        # Your prompt generation logic
        pass
    
    def validate_response(self, response, expected):
        # Your validation logic
        pass
```

### 2. Custom Metrics

**Add to [`src/evaluation/metrics/custom_metrics.py`](src/evaluation/metrics/custom_metrics.py):**
```python
def custom_financial_metric(response: str, reference: str) -> float:
    """Custom metric for financial accuracy."""
    # Your metric implementation
    return score
```

### 3. Model Registry

**Create [`src/models/model_registry.py`](src/models/model_registry.py):**
```python
from typing import Dict, List
from ..data.schemas.data_models import ModelConfig, ModelProvider

class ModelRegistry:
    def __init__(self):
        self.models = self._load_model_configs()
    
    def register_model(self, model_config: ModelConfig):
        """Register a new model."""
        self.models[model_config.name] = model_config
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.models.keys())
```

### 4. Batch Processing

**Create [`scripts/batch_evaluation.py`](scripts/batch_evaluation.py):**
```python
#!/usr/bin/env python3
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def run_batch_evaluation():
    """Run evaluations in parallel."""
    models = ["model1", "model2", "model3"]
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for model in models:
            future = executor.submit(run_single_evaluation, model)
            futures.append(future)
        
        results = [future.result() for future in futures]
    
    return results
```

## 🔧 Troubleshooting

### Common Issues

#### 1. GPU Memory Issues
```bash
# Error: CUDA out of memory
# Solution: Use quantization
```
**Fix:** Add quantization to model config:
```yaml
quantization: "4bit"
```

#### 2. API Rate Limits
```bash
# Error: Rate limit exceeded
# Solution: Add delays between requests
```
**Fix:** Reduce concurrent evaluations:
```bash
MAX_CONCURRENT_EVALUATIONS=1
```

#### 3. Model Loading Failures
```bash
# Error: Model not found
# Solution: Check model name and access
```
**Fix:** Verify model name and HuggingFace access:
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("model-name")
```

#### 4. Pinecone Connection Issues
```bash
# Error: Pinecone connection failed
# Solution: Check API key and environment
```
**Fix:** Verify Pinecone configuration:
```python
import pinecone
pinecone.init(api_key="your-key", environment="your-env")
print(pinecone.list_indexes())
```

### Performance Optimization

#### 1. Model Caching
```python
# Cache loaded models to avoid reloading
model_loader.load_model("model-name", force_reload=False)
```

#### 2. Batch Processing
```python
# Process multiple tasks in batches
batch_size = 4
for i in range(0, len(tasks), batch_size):
    batch = tasks[i:i+batch_size]
    process_batch(batch)
```

#### 3. Memory Management
```python
# Clear GPU memory between models
import torch
torch.cuda.empty_cache()
```

### Monitoring and Logging

#### 1. Enable Debug Logging
```bash
LOG_LEVEL=DEBUG python scripts/run_benchmark.py --verbose
```

#### 2. Monitor GPU Usage
```bash
# Install nvidia-ml-py
pip install nvidia-ml-py3

# Monitor in Python
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
info = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(f"GPU Memory: {info.used/1024**3:.1f}GB / {info.total/1024**3:.1f}GB")
```

#### 3. Performance Profiling
```python
import time
import psutil

def profile_evaluation():
    start_time = time.time()
    start_memory = psutil.virtual_memory().used
    
    # Run evaluation
    result = run_evaluation()
    
    end_time = time.time()
    end_memory = psutil.virtual_memory().used
    
    print(f"Duration: {end_time - start_time:.2f}s")
    print(f"Memory used: {(end_memory - start_memory)/1024**3:.2f}GB")
```

## 📞 Support

### Getting Help

1. **Check logs**: `logs/leaderboard.log`
2. **Run system test**: `python scripts/test_system.py`
3. **Check configuration**: Verify `.env` file
4. **Review documentation**: This guide and architecture docs

### Common Commands

```bash
# System health check
python scripts/test_system.py

# Quick model test
python scripts/run_benchmark.py --models gpt-3.5-turbo --tasks eps_extraction

# Dashboard launch
streamlit run streamlit_app/main.py

# Environment setup
python scripts/setup_environment.py --verbose
```

---

**Next Steps:**
1. Configure your API keys in `.env`
2. Run the system test to verify setup
3. Start with a simple model comparison
4. Explore the dashboard for detailed analysis
5. Add your custom models and tasks as needed