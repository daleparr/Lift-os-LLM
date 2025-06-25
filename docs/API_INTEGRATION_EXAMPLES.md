# API Integration Examples

This document provides practical code examples for integrating different model APIs and HuggingFace models with the LLM Finance Leaderboard.

**Key Insight**: You only need **ONE** HuggingFace token to access 7,000+ open source models. See [HuggingFace Multi-Model Guide](HUGGINGFACE_MULTI_MODEL_GUIDE.md) for details.

## 📋 Table of Contents

1. [OpenAI Integration](#openai-integration)
2. [Anthropic Integration](#anthropic-integration)
3. [HuggingFace Integration](#huggingface-integration)
4. [Custom API Integration](#custom-api-integration)
5. [Model Configuration Examples](#model-configuration-examples)
6. [Testing and Validation](#testing-and-validation)

## 🤖 OpenAI Integration

### 1. Basic Setup

**Environment Configuration:**
```bash
# .env file
OPENAI_API_KEY=sk-your-openai-api-key-here
```

**Model Configuration in [`src/config/models_config.yaml`](src/config/models_config.yaml):**
```yaml
stock_models:
  gpt35_turbo:
    name: "gpt-3.5-turbo-1106"
    display_name: "GPT-3.5 Turbo"
    parameters: "20B"
    quantization: null
    context_length: 16385
    cost_per_1k_tokens: 0.001
    provider: "openai"
    temperature: 0.1
    max_tokens: 2048

  gpt4_turbo:
    name: "gpt-4-turbo-preview"
    display_name: "GPT-4 Turbo"
    parameters: "175B"
    quantization: null
    context_length: 128000
    cost_per_1k_tokens: 0.01
    provider: "openai"
    temperature: 0.1
    max_tokens: 2048

  gpt4_vision:
    name: "gpt-4-vision-preview"
    display_name: "GPT-4 Vision"
    parameters: "175B"
    quantization: null
    context_length: 128000
    cost_per_1k_tokens: 0.01
    provider: "openai"
    temperature: 0.1
    max_tokens: 2048
```

### 2. Testing OpenAI Models

**Quick Test Script:**
```python
#!/usr/bin/env python3
"""Test OpenAI model integration."""

import asyncio
from src.evaluation.runners.benchmark_runner import create_benchmark_runner

async def test_openai_models():
    """Test OpenAI models."""
    runner = create_benchmark_runner()
    
    models = ["gpt35_turbo", "gpt4_turbo"]
    
    result = await runner.run_benchmark(
        models=models,
        tasks=["eps_extraction"],
        run_name="OpenAI Model Test",
        description="Testing OpenAI GPT models"
    )
    
    print("OpenAI Model Results:")
    for model_name, metrics in result.model_metrics.items():
        print(f"{model_name}:")
        print(f"  Final Score: {metrics.final_score:.3f}")
        print(f"  Quality: {metrics.overall_quality_score:.3f}")
        print(f"  Avg Cost: ${metrics.avg_cost_per_task:.4f}")
        print(f"  Latency: {metrics.avg_latency_ms:.0f}ms")
        print()

if __name__ == "__main__":
    asyncio.run(test_openai_models())
```

### 3. Cost Monitoring

**Track OpenAI API Costs:**
```python
#!/usr/bin/env python3
"""Monitor OpenAI API costs."""

import openai
from datetime import datetime, timedelta

def get_openai_usage():
    """Get OpenAI API usage statistics."""
    client = openai.OpenAI()
    
    # Get usage for last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    try:
        # Note: This is a placeholder - actual OpenAI usage API may differ
        usage = client.usage.retrieve(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        print(f"OpenAI Usage (Last 30 days):")
        print(f"Total Requests: {usage.total_requests}")
        print(f"Total Tokens: {usage.total_tokens:,}")
        print(f"Estimated Cost: ${usage.total_cost:.2f}")
        
    except Exception as e:
        print(f"Could not retrieve usage: {e}")

if __name__ == "__main__":
    get_openai_usage()
```

## 🧠 Anthropic Integration

### 1. Basic Setup

**Environment Configuration:**
```bash
# .env file
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here
```

**Model Configuration:**
```yaml
stock_models:
  claude3_haiku:
    name: "claude-3-haiku-20240307"
    display_name: "Claude 3 Haiku"
    parameters: "20B"
    quantization: null
    context_length: 200000
    cost_per_1k_tokens: 0.00025
    provider: "anthropic"
    temperature: 0.1
    max_tokens: 2048

  claude3_sonnet:
    name: "claude-3-sonnet-20240229"
    display_name: "Claude 3 Sonnet"
    parameters: "200B"
    quantization: null
    context_length: 200000
    cost_per_1k_tokens: 0.003
    provider: "anthropic"
    temperature: 0.1
    max_tokens: 2048

  claude3_opus:
    name: "claude-3-opus-20240229"
    display_name: "Claude 3 Opus"
    parameters: "400B"
    quantization: null
    context_length: 200000
    cost_per_1k_tokens: 0.015
    provider: "anthropic"
    temperature: 0.1
    max_tokens: 2048
```

### 2. Testing Anthropic Models

**Test Script:**
```python
#!/usr/bin/env python3
"""Test Anthropic model integration."""

import asyncio
from src.evaluation.runners.benchmark_runner import create_benchmark_runner

async def test_anthropic_models():
    """Test Anthropic Claude models."""
    runner = create_benchmark_runner()
    
    models = ["claude3_haiku", "claude3_sonnet"]
    
    result = await runner.run_benchmark(
        models=models,
        tasks=["eps_extraction"],
        run_name="Anthropic Model Test",
        description="Testing Anthropic Claude models"
    )
    
    print("Anthropic Model Results:")
    for model_name, metrics in result.model_metrics.items():
        print(f"{model_name}:")
        print(f"  Final Score: {metrics.final_score:.3f}")
        print(f"  Quality: {metrics.overall_quality_score:.3f}")
        print(f"  Avg Cost: ${metrics.avg_cost_per_task:.4f}")
        print(f"  Context Length: 200K tokens")
        print()

if __name__ == "__main__":
    asyncio.run(test_anthropic_models())
```

## 🤗 HuggingFace Integration

### 1. Basic Setup

**Environment Configuration:**
```bash
# .env file
HUGGINGFACE_API_TOKEN=hf_your-huggingface-token-here
```

**Model Configuration Examples:**

#### Small Models (3-7B parameters)
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

  phi3_mini:
    name: "microsoft/Phi-3-mini-128k-instruct"
    display_name: "Phi-3 Mini 128K"
    parameters: "3.8B"
    quantization: null
    context_length: 128000
    cost_per_1k_tokens: 0.0
    provider: "huggingface"

  qwen_7b:
    name: "Qwen/Qwen1.5-7B-Chat"
    display_name: "Qwen 1.5 7B Chat"
    parameters: "7B"
    quantization: "4bit"
    context_length: 32768
    cost_per_1k_tokens: 0.0
    provider: "huggingface"
```

#### Medium Models (13B parameters)
```yaml
stock_models:
  llama2_13b:
    name: "meta-llama/Llama-2-13b-chat-hf"
    display_name: "Llama 2 13B Chat"
    parameters: "13B"
    quantization: "4bit"
    context_length: 4096
    cost_per_1k_tokens: 0.0
    provider: "huggingface"

  vicuna_13b:
    name: "lmsys/vicuna-13b-v1.5"
    display_name: "Vicuna 13B v1.5"
    parameters: "13B"
    quantization: "4bit"
    context_length: 4096
    cost_per_1k_tokens: 0.0
    provider: "huggingface"
```

#### Large Models (30B+ parameters)
```yaml
stock_models:
  falcon_40b:
    name: "tiiuae/falcon-40b-instruct"
    display_name: "Falcon 40B Instruct"
    parameters: "40B"
    quantization: "4bit"
    context_length: 2048
    cost_per_1k_tokens: 0.0
    provider: "huggingface"

  llama2_70b:
    name: "meta-llama/Llama-2-70b-chat-hf"
    display_name: "Llama 2 70B Chat"
    parameters: "70B"
    quantization: "4bit"
    context_length: 4096
    cost_per_1k_tokens: 0.0
    provider: "huggingface"
```

### 2. Finance-Specific Models

```yaml
finance_tuned_models:
  finma_7b:
    name: "ChanceFocus/finma-7b-nlp"
    display_name: "FinMA 7B"
    parameters: "7B"
    quantization: "4bit"
    context_length: 8192
    cost_per_1k_tokens: 0.0
    provider: "huggingface"
    base_model: "mistralai/Mistral-7B-Instruct-v0.1"

  fingpt_llama2:
    name: "FinGPT/fingpt-mt_llama2-7b_lora"
    display_name: "FinGPT Llama2 7B LoRA"
    parameters: "7B"
    quantization: "4bit"
    context_length: 4096
    cost_per_1k_tokens: 0.0
    provider: "huggingface"
    base_model: "meta-llama/Llama-2-7b-chat-hf"
    lora_path: "FinGPT/fingpt-mt_llama2-7b_lora"
```

### 3. Testing HuggingFace Models

**GPU Memory Check:**
```python
#!/usr/bin/env python3
"""Check GPU memory before loading models."""

import torch

def check_gpu_memory():
    """Check available GPU memory."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"GPU {i}: {props.name}")
            print(f"  Total Memory: {memory_gb:.1f} GB")
            
            # Check current usage
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            cached = torch.cuda.memory_reserved(i) / (1024**3)
            print(f"  Allocated: {allocated:.1f} GB")
            print(f"  Cached: {cached:.1f} GB")
            print(f"  Available: {memory_gb - cached:.1f} GB")
            print()
    else:
        print("No CUDA GPUs available")

if __name__ == "__main__":
    check_gpu_memory()
```

**Model Loading Test:**
```python
#!/usr/bin/env python3
"""Test HuggingFace model loading."""

import torch
from src.models.model_loader import get_model_loader
from src.data.schemas.data_models import ModelConfig, ModelProvider

def test_huggingface_model():
    """Test loading a HuggingFace model."""
    loader = get_model_loader()
    
    # Test model configuration
    model_config = ModelConfig(
        name="mistralai/Mistral-7B-Instruct-v0.1",
        display_name="Mistral 7B Test",
        provider=ModelProvider.HUGGINGFACE,
        parameters="7B",
        context_length=8192,
        cost_per_1k_tokens=0.0,
        quantization="4bit",
        temperature=0.1,
        max_tokens=512
    )
    
    print("🔄 Loading HuggingFace model...")
    print(f"Model: {model_config.name}")
    print(f"Quantization: {model_config.quantization}")
    
    try:
        # Check compatibility first
        compatibility = loader.check_model_compatibility(model_config)
        print(f"Compatible: {compatibility['compatible']}")
        
        if compatibility['warnings']:
            print("Warnings:")
            for warning in compatibility['warnings']:
                print(f"  - {warning}")
        
        if compatibility['compatible']:
            # Load the model
            model = loader.load_model("mistral_7b_test", model_config)
            print("✅ Model loaded successfully!")
            
            # Get model info
            info = loader.get_model_info("mistral_7b_test")
            print(f"Model Info: {info}")
            
        else:
            print("❌ Model not compatible with current system")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")

if __name__ == "__main__":
    test_huggingface_model()
```

### 4. Batch Model Comparison

**Compare Multiple HuggingFace Models:**
```python
#!/usr/bin/env python3
"""Compare multiple HuggingFace models."""

import asyncio
from src.evaluation.runners.benchmark_runner import create_benchmark_runner

async def compare_huggingface_models():
    """Compare different HuggingFace models."""
    runner = create_benchmark_runner()
    
    # Test different model sizes
    model_groups = {
        "Small Models (3-7B)": ["mistral_7b", "phi3_mini", "qwen_7b"],
        "Medium Models (13B)": ["llama2_13b"],
        "Finance Models": ["finma_7b"]
    }
    
    all_results = {}
    
    for group_name, models in model_groups.items():
        print(f"\n🧪 Testing {group_name}...")
        
        try:
            result = await runner.run_benchmark(
                models=models,
                tasks=["eps_extraction"],
                run_name=f"HF {group_name}",
                description=f"Testing {group_name}"
            )
            
            all_results[group_name] = result
            
            # Print group results
            print(f"✅ {group_name} Results:")
            for model_name, metrics in result.model_metrics.items():
                print(f"  {model_name}: {metrics.final_score:.3f}")
                
        except Exception as e:
            print(f"❌ Error testing {group_name}: {e}")
    
    # Overall comparison
    print(f"\n🏆 Overall Best Models:")
    all_models = []
    for result in all_results.values():
        for model_name, metrics in result.model_metrics.items():
            all_models.append((model_name, metrics.final_score))
    
    all_models.sort(key=lambda x: x[1], reverse=True)
    for i, (model_name, score) in enumerate(all_models[:5], 1):
        print(f"{i}. {model_name}: {score:.3f}")

if __name__ == "__main__":
    asyncio.run(compare_huggingface_models())
```

## 🔌 Custom API Integration

### 1. Adding a New API Provider

**Example: Cohere Integration**

**Step 1: Add to model loader [`src/models/model_loader.py`](src/models/model_loader.py):**
```python
def _load_cohere_model(self, model_name: str, model_config: ModelConfig):
    """Load Cohere model."""
    try:
        import cohere
        from langchain_community.llms import Cohere
        
        if not os.getenv("COHERE_API_KEY"):
            raise ValueError("Cohere API key not configured")
        
        return Cohere(
            model=model_name,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            cohere_api_key=os.getenv("COHERE_API_KEY")
        )
    except ImportError:
        raise ImportError("cohere package not installed. Install with: pip install cohere")
```

**Step 2: Update provider detection:**
```python
def _detect_model_provider(self, model_name: str) -> ModelProvider:
    """Detect model provider from model name."""
    model_name_lower = model_name.lower()
    
    if "gpt" in model_name_lower or "openai" in model_name_lower:
        return ModelProvider.OPENAI
    elif "claude" in model_name_lower or "anthropic" in model_name_lower:
        return ModelProvider.ANTHROPIC
    elif "cohere" in model_name_lower:
        return ModelProvider.COHERE  # Add to enum
    elif "/" in model_name or any(org in model_name_lower for org in ["meta", "mistral", "microsoft"]):
        return ModelProvider.HUGGINGFACE
    else:
        return ModelProvider.LOCAL
```

**Step 3: Add model configuration:**
```yaml
stock_models:
  cohere_command:
    name: "command"
    display_name: "Cohere Command"
    parameters: "52B"
    quantization: null
    context_length: 4096
    cost_per_1k_tokens: 0.0015
    provider: "cohere"
```

### 2. Local API Server Integration

**Example: Ollama Integration**

```python
def _load_ollama_model(self, model_name: str, model_config: ModelConfig):
    """Load Ollama local model."""
    try:
        from langchain_community.llms import Ollama
        
        return Ollama(
            model=model_name,
            base_url="http://localhost:11434",  # Default Ollama port
            temperature=model_config.temperature
        )
    except ImportError:
        raise ImportError("langchain-community package required for Ollama")
```

**Configuration:**
```yaml
stock_models:
  ollama_llama2:
    name: "llama2:7b"
    display_name: "Ollama Llama 2 7B"
    parameters: "7B"
    quantization: null
    context_length: 4096
    cost_per_1k_tokens: 0.0
    provider: "ollama"
```

## 📊 Model Configuration Examples

### 1. Performance-Optimized Configurations

**For Speed (Low Latency):**
```yaml
speed_optimized:
  phi3_mini_fast:
    name: "microsoft/Phi-3-mini-128k-instruct"
    display_name: "Phi-3 Mini (Speed)"
    parameters: "3.8B"
    quantization: "4bit"
    context_length: 4096  # Reduced for speed
    temperature: 0.0      # Deterministic
    max_tokens: 512       # Shorter responses
    provider: "huggingface"
```

**For Quality (High Accuracy):**
```yaml
quality_optimized:
  gpt4_quality:
    name: "gpt-4-turbo-preview"
    display_name: "GPT-4 (Quality)"
    parameters: "175B"
    context_length: 128000
    temperature: 0.1      # Low but not zero
    max_tokens: 4096      # Longer responses
    cost_per_1k_tokens: 0.01
    provider: "openai"
```

**For Cost Efficiency:**
```yaml
cost_optimized:
  mistral_7b_efficient:
    name: "mistralai/Mistral-7B-Instruct-v0.1"
    display_name: "Mistral 7B (Cost Efficient)"
    parameters: "7B"
    quantization: "4bit"
    context_length: 8192
    temperature: 0.1
    max_tokens: 1024      # Moderate length
    cost_per_1k_tokens: 0.0
    provider: "huggingface"
```

### 2. Task-Specific Configurations

**For Numerical Tasks:**
```yaml
numerical_tasks:
  gpt35_numerical:
    name: "gpt-3.5-turbo-1106"
    display_name: "GPT-3.5 (Numerical)"
    temperature: 0.0      # Deterministic for numbers
    max_tokens: 512       # Short, precise answers
    provider: "openai"
```

**For Analysis Tasks:**
```yaml
analysis_tasks:
  claude3_analysis:
    name: "claude-3-sonnet-20240229"
    display_name: "Claude 3 (Analysis)"
    temperature: 0.2      # Slightly creative
    max_tokens: 2048      # Longer explanations
    context_length: 200000  # Large context for analysis
    provider: "anthropic"
```

## 🧪 Testing and Validation

### 1. API Connection Test

```python
#!/usr/bin/env python3
"""Test all API connections."""

import os
from dotenv import load_dotenv

load_dotenv()

def test_openai():
    """Test OpenAI API connection."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        print("✅ OpenAI API: Connected")
        return True
    except Exception as e:
        print(f"❌ OpenAI API: {e}")
        return False

def test_anthropic():
    """Test Anthropic API connection."""
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=5,
            messages=[{"role": "user", "content": "Hello"}]
        )
        print("✅ Anthropic API: Connected")
        return True
    except Exception as e:
        print(f"❌ Anthropic API: {e}")
        return False

def test_huggingface():
    """Test HuggingFace API connection."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.getenv("HUGGINGFACE_API_TOKEN"))
        
        # Test by getting user info
        user = api.whoami()
        print(f"✅ HuggingFace API: Connected as {user['name']}")
        return True
    except Exception as e:
        print(f"❌ HuggingFace API: {e}")
        return False

def test_pinecone():
    """Test Pinecone connection."""
    try:
        import pinecone
        from pinecone import Pinecone
        
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        indexes = pc.list_indexes()
        print(f"✅ Pinecone: Connected, {len(indexes)} indexes")
        return True
    except Exception as e:
        print(f"❌ Pinecone: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing API Connections...")
    print("=" * 40)
    
    results = {
        "OpenAI": test_openai(),
        "Anthropic": test_anthropic(),
        "HuggingFace": test_huggingface(),
        "Pinecone": test_pinecone()
    }
    
    print("=" * 40)
    connected = sum(results.values())
    total = len(results)
    print(f"📊 Summary: {connected}/{total} APIs connected")
    
    if connected == total:
        print("🎉 All APIs ready for model comparison!")
    else:
        print("⚠️  Some APIs need configuration")
```

### 2. Model Performance Test

```python
#!/usr/bin/env python3
"""Test model performance across different configurations."""

import asyncio
import time
from src.evaluation.runners.benchmark_runner import create_benchmark_runner

async def performance_test():
    """Test model performance."""
    runner = create_benchmark_runner()
    
    # Test configurations
    test_cases = [
        {
            "name": "Speed Test",
            "models": ["phi3_mini", "gpt35_turbo"],
            "description": "Testing fastest models"
        },
        {
            "name": "Quality Test", 
            "models": ["gpt4_turbo", "claude3_sonnet"],
            "description": "Testing highest quality models"
        },
        {
            "name": "Cost Test",
            "models": ["mistral_7b", "finma_7b"],
            "description": "Testing most cost-effective models"
        }
    ]
    
    results = {}
    
    for test_case in test_cases:
        print(f"\n🧪 Running {test_case['name']}...")
        start_time = time.time()
        
        try:
            result = await runner.run_benchmark(
                models=test_case["models"],
                tasks=["eps_extraction"],
                run_name=test_case["name"],
                description=test_case["description"]
            )
            
            duration = time.time() - start_time
            results[test_case["name"]] = {
                "result": result,
                "duration": duration
            }
            
            print(f"✅ {test_case['name']} completed in {duration:.1f}s")
            
        except Exception as e:
            print(f"❌ {test_case['name']} failed: {e}")
    
    # Performance summary
    print(f"\n📊 Performance Summary:")
    print("=" * 50)
    
    for test_name, data in results.items():
        result = data["result"]
        duration = data["duration"]
        
        print(f"\n{test_name}:")
        print(f"  Duration: {duration:.1f}s")
        
        for model_name, metrics in result.model_metrics.items():
            print(f"  {model_name}:")
            print(f"    Score: {metrics.final_score:.3f}")
            print(f"    Latency: {metrics.avg_latency_ms:.0f}ms")
            print(f"    Cost: ${metrics.avg_cost_per_task:.4f}")

if __name__ == "__main__":
    asyncio.run(performance_test())
```

---

**Ready to integrate your models?**

1. ✅ Choose your API providers
2. ✅ Configure API keys and model settings
3. ✅ Test connections and model loading
4. ✅ Run performance comparisons
5. ✅ Analyze results and optimize configurations