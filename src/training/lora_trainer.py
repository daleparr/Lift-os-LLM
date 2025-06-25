"""
LoRA trainer for fine-tuning models on financial datasets.
"""

import os
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from loguru import logger
import yaml

try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        TrainingArguments, Trainer,
        BitsAndBytesConfig
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset, load_dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers/peft not available. Install with: pip install transformers peft datasets")

from ..data.schemas.data_models import TrainingConfig, TrainingJob
from ..config.settings import settings


class LoRATrainer:
    """LoRA trainer for fine-tuning models locally."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize LoRA trainer."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and peft libraries are required for training")
        
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Load training configuration
        self.training_config = self._load_training_config()
        
        logger.info(f"Initialized LoRATrainer for {config.model_name}")
        logger.info(f"Using device: {self.device}")
    
    def _load_training_config(self) -> Dict[str, Any]:
        """Load training configuration from YAML."""
        config_path = Path("src/config/training_config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            logger.warning("Training config not found, using defaults")
            return {}
    
    def prepare_model_and_tokenizer(self) -> None:
        """Prepare model and tokenizer for training."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Quantization config
        if self.config.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            quantization_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quantization_config,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.config.use_fp16 else torch.float32,
        )
        
        # Prepare model for training
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("Model and tokenizer prepared successfully")
    
    def prepare_dataset(self) -> Dataset:
        """Prepare training dataset."""
        logger.info(f"Loading dataset from: {self.config.dataset_path}")
        
        if not os.path.exists(self.config.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.config.dataset_path}")
        
        # Load JSONL dataset
        data = []
        with open(self.config.dataset_path, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        # Convert to HuggingFace dataset
        dataset = Dataset.from_list(data)
        
        # Tokenize dataset
        def tokenize_function(examples):
            # Format as instruction-following
            texts = []
            for i in range(len(examples['instruction'])):
                instruction = examples['instruction'][i]
                input_text = examples.get('input', [''] * len(examples['instruction']))[i]
                output = examples['output'][i]
                
                if input_text:
                    text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                else:
                    text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                
                texts.append(text)
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=2048,
                return_tensors=None,
            )
            
            # Set labels for causal LM
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        
        logger.info(f"Dataset prepared with {len(tokenized_dataset)} samples")
        return tokenized_dataset
    
    def train(self, job: TrainingJob) -> str:
        """Train the model with LoRA."""
        try:
            logger.info(f"Starting training job: {job.job_id}")
            
            # Prepare model and dataset
            self.prepare_model_and_tokenizer()
            dataset = self.prepare_dataset()
            
            # Split dataset
            train_size = int(0.9 * len(dataset))
            train_dataset = dataset.select(range(train_size))
            eval_dataset = dataset.select(range(train_size, len(dataset)))
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                gradient_accumulation_steps=4,
                warmup_steps=100,
                max_steps=self.config.max_steps,
                learning_rate=self.config.learning_rate,
                fp16=self.config.use_fp16,
                logging_steps=10,
                save_steps=100,
                eval_steps=100,
                save_total_limit=3,
                remove_unused_columns=False,
                push_to_hub=False,
                report_to="none",
                dataloader_pin_memory=False,
            )
            
            # Initialize trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
            )
            
            # Start training
            start_time = time.time()
            self.trainer.train()
            training_time = time.time() - start_time
            
            # Save model
            output_path = os.path.join(self.config.output_dir, f"lora_adapter_{job.job_id}")
            self.trainer.model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Model saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def validate_model(self, model_path: str) -> Dict[str, float]:
        """Validate the trained model."""
        logger.info(f"Validating model: {model_path}")
        
        try:
            # Load the fine-tuned model
            from peft import PeftModel
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            
            model = PeftModel.from_pretrained(base_model, model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Simple validation - generate a response
            test_prompt = "### Instruction:\nWhat is the EPS for a company with net income of $100M and 50M shares outstanding?\n\n### Response:\n"
            
            inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Validation response: {response}")
            
            # Return basic metrics
            return {
                "validation_success": 1.0,
                "response_length": len(response),
                "contains_calculation": 1.0 if "$2.00" in response or "2.00" in response else 0.0
            }
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return {"validation_success": 0.0}