# HuggingFace Multi-Model Integration Guide

## 🎯 Key Point: One API Token, Multiple Models

**You only need ONE HuggingFace API token to access thousands of open source models.** The HuggingFace integration acts as a universal gateway to the entire model ecosystem.

## 🔑 Single API Setup

### Environment Configuration
```bash
# .env file - Only need ONE token
HUGGINGFACE_API_TOKEN=hf_your-single-token-here
```

### What This Single Token Gives You Access To
- **7,000+ Language Models** on HuggingFace Hub
- **All Major Model Families**: Llama, Mistral, Qwen, Phi, Falcon, etc.
- **Finance-Specific Models**: FinMA, FinGPT, custom fine-tunes
- **Different Sizes**: 1B to 70B+ parameters
- **Various Formats**: Base models, instruction-tuned, chat models

## 📊 Multi-Model Configuration Examples

### Single Configuration File for Multiple Models
**In [`src/config/models_config.yaml`](src/config/models_config.yaml):**

```yaml
# ALL these models use the SAME HuggingFace token
stock_models:
  # 7B Models
  mistral_7b:
    name: "mistralai/Mistral-7B-Instruct-v0.1"
    display_name: "Mistral 7B Instruct"
    parameters: "7B"
    provider: "huggingface"
    
  llama2_7b:
    name: "meta-llama/Llama-2-7b-chat-hf"
    display_name: "Llama 2 7B Chat"
    parameters: "7B"
    provider: "huggingface"
    
  qwen_7b:
    name: "Qwen/Qwen1.5-7B-Chat"
    display_name: "Qwen 1.5 7B Chat"
    parameters: "7B"
    provider: "huggingface"
    
  phi3_mini:
    name: "microsoft/Phi-3-mini-128k-instruct"
    display_name: "Phi-3 Mini 128K"
    parameters: "3.8B"
    provider: "huggingface"
    
  # 13B Models
  llama2_13b:
    name: "meta-llama/Llama-2-13b-chat-hf"
    display_name: "Llama 2 13B Chat"
    parameters: "13B"
    provider: "huggingface"
    
  vicuna_13b:
    name: "lmsys/vicuna-13b-v1.5"
    display_name: "Vicuna 13B v1.5"
    parameters: "13B"
    provider: "huggingface"
    
  # Large Models
  falcon_40b:
    name: "tiiuae/falcon-40b-instruct"
    display_name: "Falcon 40B Instruct"
    parameters: "40B"
    provider: "huggingface"

finance_tuned_models:
  # Finance-specific models - SAME token
  finma_7b:
    name: "ChanceFocus/finma-7b-nlp"
    display_name: "FinMA 7B"
    parameters: "7B"
    provider: "huggingface"
    
  fingpt_llama2:
    name: "FinGPT/fingpt-mt_llama2-7b_lora"
    display_name: "FinGPT Llama2 7B"
    parameters: "7B"
    provider: "huggingface"
```

## 🚀 Multi-Model Comparison Examples

### Compare 10+ Models with Single API Token
```bash
# Compare multiple model families
python scripts/run_benchmark.py \
  --models mistral_7b llama2_7b qwen_7b phi3_mini llama2_13b finma_7b \
  --tasks eps_extraction \
  --name "Multi-Model Open Source Comparison" \
  --use-real-runner
```

### Model Family Comparison
```python
#!/usr/bin/env python3
"""Compare different model families using single HF token."""

import asyncio
from src.evaluation.runners.benchmark_runner import create_benchmark_runner

async def compare_model_families():
    """Compare different open source model families."""
    runner = create_benchmark_runner()
    
    # All these use the SAME HuggingFace token
    model_families = {
        "7B Models": [
            "mistral_7b",      # Mistral family
            "llama2_7b",       # Meta family  
            "qwen_7b",         # Alibaba family
            "phi3_mini"        # Microsoft family
        ],
        "13B Models": [
            "llama2_13b",
            "vicuna_13b"
        ],
        "Finance Models": [
            "finma_7b",
            "fingpt_llama2"
        ]
    }
    
    for family_name, models in model_families.items():
        print(f"\n🧪 Testing {family_name}...")
        
        result = await runner.run_benchmark(
            models=models,
            tasks=["eps_extraction"],
            run_name=f"HF {family_name}",
            description=f"Comparing {family_name} using single HF token"
        )
        
        print(f"✅ {family_name} Results:")
        for model_name, metrics in result.model_metrics.items():
            print(f"  {model_name}: {metrics.final_score:.3f}")

if __name__ == "__main__":
    asyncio.run(compare_model_families())
```

## 💡 Why One Token is Sufficient

### 1. **Unified Access**
- HuggingFace Hub uses a single authentication system
- One token grants access to all public models
- No need for separate API keys per model

### 2. **Cost Efficiency**
- **Free Tier**: Most models can be downloaded and run locally
- **No Per-Model Charges**: Unlike API providers
- **One-Time Setup**: Configure once, use everywhere

### 3. **Flexibility**
- **Easy Model Switching**: Just change the model name
- **Rapid Experimentation**: Test dozens of models quickly
- **No Vendor Lock-in**: Access to entire open source ecosystem

## 🔄 Adding New Models (Same Token)

### Step 1: Find Model on HuggingFace Hub
Visit [huggingface.co/models](https://huggingface.co/models) and find your model.

### Step 2: Add to Configuration
```yaml
# Add any HuggingFace model using the same token
new_model:
  name: "organization/model-name"  # From HuggingFace Hub
  display_name: "Your Display Name"
  parameters: "7B"  # Approximate size
  provider: "huggingface"
  quantization: "4bit"  # Optional optimization
```

### Step 3: Test Immediately
```bash
python scripts/run_benchmark.py --models new_model --tasks eps_extraction
```

## 📊 Comparison: API vs HuggingFace Integration

| Aspect | API Providers (OpenAI/Anthropic) | HuggingFace |
|--------|-----------------------------------|-------------|
| **Number of Tokens** | One per provider | **One for all models** |
| **Model Access** | Limited to provider's models | **7,000+ models** |
| **Cost** | Pay per token/request | **Free (local compute)** |
| **Setup Complexity** | Simple API calls | Model download + GPU |
| **Customization** | Limited | **Full control** |
| **Privacy** | Data sent to provider | **Local processing** |

## 🎯 Recommended Multi-Model Strategy

### Phase 1: Start with Popular Models (Same Token)
```yaml
starter_models:
  mistral_7b:
    name: "mistralai/Mistral-7B-Instruct-v0.1"
  llama2_7b:
    name: "meta-llama/Llama-2-7b-chat-hf"
  phi3_mini:
    name: "microsoft/Phi-3-mini-128k-instruct"
```

### Phase 2: Add Finance-Specific Models (Same Token)
```yaml
finance_models:
  finma_7b:
    name: "ChanceFocus/finma-7b-nlp"
  fingpt_llama2:
    name: "FinGPT/fingpt-mt_llama2-7b_lora"
```

### Phase 3: Scale to Larger Models (Same Token)
```yaml
large_models:
  llama2_13b:
    name: "meta-llama/Llama-2-13b-chat-hf"
  falcon_40b:
    name: "tiiuae/falcon-40b-instruct"
```

## 🔧 Optimization Tips for Multi-Model Testing

### 1. **Batch Model Loading**
```python
# Load multiple models efficiently
models_to_test = ["mistral_7b", "llama2_7b", "qwen_7b"]
for model in models_to_test:
    # Same token used for all
    result = await runner.run_benchmark(models=[model], tasks=["eps_extraction"])
```

### 2. **Memory Management**
```python
# Unload models between tests to save GPU memory
from src.models.model_loader import get_model_loader

loader = get_model_loader()
for model in models_to_test:
    # Test model
    result = test_model(model)
    # Free memory for next model
    loader.unload_model(model)
```

### 3. **Quantization for More Models**
```yaml
# Use quantization to fit more models in GPU memory
model_config:
  quantization: "4bit"  # Reduces memory by ~75%
  # Allows testing larger models or more models simultaneously
```

## ✅ Summary

**Answer: No, you only need ONE HuggingFace API token to test multiple open source models.**

### Key Benefits:
- ✅ **Single Token** → Access to 7,000+ models
- ✅ **Cost Effective** → Free local inference
- ✅ **Easy Scaling** → Add models by just changing configuration
- ✅ **Full Control** → Local processing, quantization, fine-tuning
- ✅ **Rapid Comparison** → Test dozens of models quickly

### Recommended Approach:
1. **Get ONE HuggingFace token** (free account)
2. **Configure multiple models** in YAML file
3. **Run comparative benchmarks** across model families
4. **Scale up gradually** based on GPU memory and requirements

This approach maximizes model diversity while minimizing API complexity and costs.