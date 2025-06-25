# Auto Fine-tuning Feature - Quick Start Guide

This guide helps you get started with the auto fine-tuning feature that allows you to compare base models against fine-tuned variants.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install training dependencies
pip install transformers peft datasets torch bitsandbytes accelerate

# Install monitoring dependencies  
pip install GPUtil psutil nvidia-ml-py3

# Install visualization dependencies
pip install plotly pandas
```

### 2. Prepare Training Data

The system includes a sample synthetic financial dataset:

```bash
# Dataset is already included at:
data/training/synthetic_finance_v2.jsonl
```

### 3. Launch the Dashboard

```bash
streamlit run streamlit_app/main.py
```

Navigate to the **🔧 Auto Fine-tuning** tab.

## 📋 How It Works

### Two-Option Approach

1. **Base Model (Standard)** - Evaluates the out-of-the-box model
2. **Also Fine-tune and Compare** - Adds fine-tuned variant for side-by-side comparison

### User Workflow

```
Model Selection → Choose Evaluation Mode → Configure Training → Start Job → Monitor Progress → View Results
```

## 🔧 Features

### Model Selection
- Support for 7B-13B parameter models
- Automatic hardware requirement detection
- Resource availability checking

### Training Configuration
- **Method**: LoRA/QLoRA for memory efficiency
- **Dataset**: Synthetic financial corpus
- **Monitoring**: Real-time GPU usage, temperature, power
- **Estimates**: Training time and resource consumption

### Results & Comparison
- Side-by-side performance metrics
- Improvement percentages by task
- ROI analysis (performance gain vs training cost)
- Downloadable reports

## 📊 Example Results

```
Model Performance Comparison
                    Base Model    Fine-tuned    Improvement
EPS Extraction        0.78          0.92         +18%
Sentiment Analysis    0.71          0.88         +24%  
Revenue Analysis      0.65          0.84         +29%
Overall Score         0.71          0.88         +24%

Training Time: 3.2 hours
Power Consumption: 1.8 kWh
Fine-tuning ROI: +24% performance improvement
```

## 🖥️ Hardware Requirements

### Minimum Requirements
- **7B models**: RTX 4090 24GB or equivalent
- **13B models**: A100 40GB or equivalent
- **Storage**: 50GB+ free space
- **Power**: 850W+ PSU

### Recommended Setup
- NVIDIA GPU with 24GB+ VRAM
- NVMe SSD for fast model loading
- Adequate cooling for sustained training
- Stable power supply

## 🔍 Testing

Test the system components:

```bash
python scripts/test_auto_finetuning.py
```

This will verify:
- GPU detection and monitoring
- Dataset loading
- Configuration validation
- Component integration

## 📁 File Structure

```
src/training/
├── __init__.py
├── lora_trainer.py          # Core LoRA training implementation
├── local_orchestrator.py    # Job queue and management
├── resource_manager.py      # GPU monitoring and allocation
└── comparison_engine.py     # Model comparison and evaluation

streamlit_app/components/
├── model_selector.py        # Model selection UI
├── comparison_results.py    # Results visualization
└── training_monitor.py      # Progress monitoring

data/training/
└── synthetic_finance_v2.jsonl  # Training dataset

configs/
└── training_config.yaml    # Training parameters
```

## ⚙️ Configuration

### Training Parameters

Edit `src/config/training_config.yaml`:

```yaml
lora_config:
  r: 16                    # LoRA rank
  lora_alpha: 32          # LoRA scaling
  lora_dropout: 0.1       # Dropout rate

training_args:
  num_train_epochs: 3
  learning_rate: 2e-4
  per_device_train_batch_size: 4
  max_steps: 1000
```

### Hardware Limits

```yaml
hardware_requirements:
  min_vram_gb: 16
  max_power_watts: 450
  max_temp_celsius: 85
```

## 🚨 Troubleshooting

### Common Issues

**GPU Not Detected**
```bash
# Install GPU monitoring
pip install GPUtil nvidia-ml-py3

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

**Insufficient Memory**
- Reduce batch size in training config
- Use QLoRA instead of LoRA
- Try smaller model (7B instead of 13B)

**Training Fails**
- Check dataset format (JSONL with instruction/input/output)
- Verify sufficient disk space
- Monitor GPU temperature

**Slow Training**
- Use NVMe SSD for model storage
- Increase batch size if memory allows
- Check GPU utilization in monitoring

## 📈 Performance Tips

### Optimization Strategies

1. **Memory Optimization**
   - Use 4-bit quantization (QLoRA)
   - Enable gradient checkpointing
   - Optimize batch size for your GPU

2. **Speed Optimization**
   - Use mixed precision (FP16)
   - Increase batch size within memory limits
   - Use fast storage (NVMe SSD)

3. **Quality Optimization**
   - Increase training epochs for better convergence
   - Tune learning rate for your dataset
   - Use larger LoRA rank for complex tasks

## 🔗 Integration

### API Endpoints

The system exposes REST APIs for programmatic access:

```python
# Submit training job
POST /api/training/submit
{
  "model_name": "mistralai/Mistral-7B-Instruct-v0.1",
  "dataset_path": "data/training/synthetic_finance_v2.jsonl"
}

# Check job status
GET /api/training/status/{job_id}

# Get comparison results
GET /api/models/compare/{base_model_id}/{finetuned_model_id}
```

### Programmatic Usage

```python
from src.training.local_orchestrator import LocalTrainingOrchestrator
from src.training.comparison_engine import ModelComparisonEngine

# Start training
orchestrator = LocalTrainingOrchestrator()
orchestrator.start()

job_id = orchestrator.submit_training_job(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    dataset_path="data/training/synthetic_finance_v2.jsonl"
)

# Compare models
comparison_engine = ModelComparisonEngine()
comparison = comparison_engine.compare_models(
    base_model_name="mistralai/Mistral-7B-Instruct-v0.1",
    finetuned_model_path="./models/auto_finetuned/job_123"
)

print(f"Overall improvement: {comparison.overall_improvement:.1%}")
```

## 📚 Next Steps

1. **Experiment with Different Models**: Try various 7B and 13B models
2. **Custom Datasets**: Replace synthetic data with your own financial corpus
3. **Hyperparameter Tuning**: Optimize LoRA parameters for your use case
4. **Production Deployment**: Scale up with multiple GPUs and distributed training

## 🤝 Contributing

To extend the auto fine-tuning feature:

1. **Add New Training Methods**: Implement additional fine-tuning approaches
2. **Improve Monitoring**: Add more detailed resource tracking
3. **Enhance UI**: Create better visualization components
4. **Optimize Performance**: Implement distributed training support

## 📄 License

This auto fine-tuning feature is part of the LLM Finance Leaderboard project and follows the same license terms.