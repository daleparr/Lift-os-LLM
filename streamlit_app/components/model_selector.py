"""
Model selector component for choosing base model and fine-tuning options.
"""

import streamlit as st
from typing import Dict, Any, List
import yaml
from pathlib import Path

from src.config.settings import settings


class ModelSelector:
    """Component for model selection and fine-tuning configuration."""
    
    def __init__(self):
        """Initialize model selector."""
        self.available_models = self._load_available_models()
        self.training_config = self._load_training_config()
    
    def _load_available_models(self) -> Dict[str, Any]:
        """Load available models from configuration."""
        try:
            config_path = Path("src/config/models_config.yaml")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                return config.get("stock_models", {})
            else:
                # Fallback models
                return {
                    "mistral_7b": {
                        "name": "mistralai/Mistral-7B-Instruct-v0.1",
                        "display_name": "Mistral 7B Instruct",
                        "parameters": "7B"
                    },
                    "llama2_7b": {
                        "name": "meta-llama/Llama-2-7b-chat-hf", 
                        "display_name": "Llama 2 7B Chat",
                        "parameters": "7B"
                    }
                }
        except Exception as e:
            st.error(f"Failed to load models: {e}")
            return {}
    
    def _load_training_config(self) -> Dict[str, Any]:
        """Load training configuration."""
        try:
            config_path = Path("src/config/training_config.yaml")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                return {}
        except Exception as e:
            st.warning(f"Failed to load training config: {e}")
            return {}
    
    def render(self) -> Dict[str, Any]:
        """Render the model selector component."""
        
        st.subheader("🤖 Model Configuration")
        
        # Model selection
        model_options = {}
        for key, config in self.available_models.items():
            display_name = f"{config['display_name']} ({config['parameters']})"
            model_options[display_name] = config['name']
        
        if not model_options:
            st.error("No models available. Please check configuration.")
            return {}
        
        selected_display = st.selectbox(
            "Select Base Model",
            options=list(model_options.keys()),
            help="Choose the base model for evaluation"
        )
        
        base_model = model_options[selected_display]
        
        # Fine-tuning option
        st.markdown("---")
        st.subheader("📈 Evaluation Mode")
        
        finetune_option = st.radio(
            "Choose evaluation approach:",
            options=[
                "Base Model (Standard evaluation)",
                "Also Fine-tune and Compare"
            ],
            help="Base model evaluation is always performed. Fine-tuning adds comparison analysis."
        )
        
        # Fine-tuning configuration (shown when enabled)
        dataset_config = None
        training_estimates = None
        
        if finetune_option == "Also Fine-tune and Compare":
            st.info("🔧 Fine-tuning will create a domain-specific variant for performance comparison")
            
            with st.expander("Fine-tuning Configuration", expanded=True):
                
                # Dataset selection
                available_datasets = self.training_config.get("datasets", {})
                if available_datasets:
                    dataset_options = {
                        config["name"]: key 
                        for key, config in available_datasets.items()
                    }
                    
                    selected_dataset_name = st.selectbox(
                        "Training Dataset",
                        options=list(dataset_options.keys()),
                        help="Synthetic financial dataset for domain adaptation"
                    )
                    
                    dataset_key = dataset_options[selected_dataset_name]
                    dataset_config = available_datasets[dataset_key]
                else:
                    st.warning("No training datasets configured")
                    dataset_config = {
                        "name": "Synthetic Finance Corpus v2",
                        "path": "data/training/synthetic_finance_v2.jsonl"
                    }
                
                # Training method
                method = st.selectbox(
                    "Fine-tuning Method",
                    options=["LoRA", "QLoRA"],
                    index=0,
                    help="LoRA: Low-Rank Adaptation, QLoRA: Quantized LoRA"
                )
                
                # Hardware and time estimates
                col1, col2 = st.columns(2)
                
                with col1:
                    # Estimate based on model size
                    model_size = self._extract_model_size(base_model)
                    estimated_time = self._estimate_training_time(model_size)
                    
                    st.metric(
                        "Estimated Time", 
                        f"{estimated_time:.1f} hours",
                        help="Approximate training duration"
                    )
                    
                    st.metric(
                        "GPU Required",
                        self._get_gpu_requirement(model_size),
                        help="Minimum GPU memory requirement"
                    )
                
                with col2:
                    power_consumption = estimated_time * 0.45  # 450W average
                    
                    st.metric(
                        "Power Usage",
                        f"{power_consumption:.1f} kWh",
                        help="Estimated power consumption"
                    )
                    
                    st.metric(
                        "Method",
                        method,
                        help="Training approach"
                    )
                
                # Advanced options
                with st.expander("Advanced Options"):
                    learning_rate = st.number_input(
                        "Learning Rate",
                        min_value=1e-5,
                        max_value=1e-3,
                        value=2e-4,
                        format="%.0e"
                    )
                    
                    num_epochs = st.slider(
                        "Number of Epochs",
                        min_value=1,
                        max_value=10,
                        value=3
                    )
                    
                    batch_size = st.selectbox(
                        "Batch Size",
                        options=[1, 2, 4, 8],
                        index=2
                    )
                
                training_estimates = {
                    "estimated_time_hours": estimated_time,
                    "power_consumption_kwh": power_consumption,
                    "gpu_requirement": self._get_gpu_requirement(model_size),
                    "method": method,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size
                }
        
        # Resource check
        if finetune_option == "Also Fine-tune and Compare":
            self._show_resource_check(base_model)
        
        return {
            "base_model": base_model,
            "base_model_display": selected_display,
            "finetune_enabled": finetune_option == "Also Fine-tune and Compare",
            "dataset_config": dataset_config,
            "training_estimates": training_estimates
        }
    
    def _extract_model_size(self, model_name: str) -> str:
        """Extract model size from model name."""
        model_name_lower = model_name.lower()
        if "7b" in model_name_lower:
            return "7B"
        elif "13b" in model_name_lower:
            return "13B"
        elif "3b" in model_name_lower:
            return "3B"
        else:
            return "7B"  # Default
    
    def _estimate_training_time(self, model_size: str) -> float:
        """Estimate training time in hours."""
        time_estimates = {
            "3B": 1.5,
            "7B": 3.0,
            "13B": 6.0
        }
        return time_estimates.get(model_size, 3.0)
    
    def _get_gpu_requirement(self, model_size: str) -> str:
        """Get GPU requirement for model size."""
        gpu_requirements = {
            "3B": "RTX 4060 Ti 16GB",
            "7B": "RTX 4090 24GB", 
            "13B": "A100 40GB"
        }
        return gpu_requirements.get(model_size, "RTX 4090 24GB")
    
    def _show_resource_check(self, model_name: str) -> None:
        """Show resource availability check."""
        
        st.markdown("---")
        st.subheader("🔍 Resource Check")
        
        # Mock resource check - in real implementation, this would check actual resources
        model_size = self._extract_model_size(model_name)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # GPU availability
            gpu_available = True  # Mock check
            if gpu_available:
                st.success("✅ GPU Available")
            else:
                st.error("❌ No GPU Detected")
        
        with col2:
            # Memory check
            memory_sufficient = True  # Mock check
            if memory_sufficient:
                st.success("✅ Memory Sufficient")
            else:
                st.error("❌ Insufficient Memory")
        
        with col3:
            # Thermal check
            thermal_ok = True  # Mock check
            if thermal_ok:
                st.success("✅ Thermal OK")
            else:
                st.warning("⚠️ High Temperature")
        
        if not (gpu_available and memory_sufficient and thermal_ok):
            st.error("⚠️ Resource requirements not met. Fine-tuning may fail.")
            st.info("Consider using a smaller model or upgrading hardware.")