#!/usr/bin/env python3
"""
Comprehensive Custom Model Training Pipeline
- Streams large datasets (LAION, custom data)
- LoRA training for efficiency
- Supports multiple resolutions
- Saves deployable models
"""

import os
import json
import torch
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL
)
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomModelTrainer:
    def __init__(
        self,
        base_model: str = "runwayml/stable-diffusion-v1-5",
        use_sdxl: bool = False,
        resolution: int = 512,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 1e-4,
        num_epochs: int = 1,
        save_steps: int = 500,
        max_train_steps: Optional[int] = None,
        mixed_precision: str = "fp16",
        output_dir: str = "custom_model_outputs"
    ):
        self.base_model = base_model
        self.use_sdxl = use_sdxl
        self.resolution = resolution
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.save_steps = save_steps
        self.max_train_steps = max_train_steps
        self.mixed_precision = mixed_precision
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with="tensorboard",
            project_dir=str(self.output_dir / "logs")
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
    def load_datasets(self, dataset_configs: Dict[str, Any]):
        """Load and combine multiple datasets with streaming support"""
        datasets = []
        
        for config in dataset_configs:
            dataset_type = config.get("type", "huggingface")
            
            if dataset_type == "huggingface":
                # Load HuggingFace dataset (LAION, etc.)
                dataset = load_dataset(
                    config["name"],
                    split=config.get("split", "train"),
                    streaming=config.get("streaming", True),
                    cache_dir=config.get("cache_dir", None)
                )
                
                # Apply transformations if specified
                if "transform" in config:
                    dataset = dataset.map(config["transform"])
                    
            elif dataset_type == "imagefolder":
                # Load local image folder dataset
                dataset = load_dataset(
                    "imagefolder",
                    data_dir=config["data_dir"],
                    split="train",
                    streaming=False  # Local datasets usually not streaming
                )
                
            elif dataset_type == "custom":
                # Custom dataset loading logic
                dataset = self._load_custom_dataset(config)
                
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")
            
            # Apply filtering if specified
            if "filter" in config:
                dataset = dataset.filter(config["filter"])
            
            datasets.append(dataset)
        
        # Combine all datasets
        if len(datasets) > 1:
            combined_dataset = concatenate_datasets(datasets)
        else:
            combined_dataset = datasets[0]
            
        logger.info(f"Loaded {len(combined_dataset)} total samples")
        return combined_dataset
    
    def _load_custom_dataset(self, config: Dict[str, Any]) -> Dataset:
        """Load custom dataset from various formats"""
        data_dir = Path(config["data_dir"])
        
        if not data_dir.exists():
            raise ValueError(f"Dataset directory not found: {data_dir}")
        
        # Support for different custom dataset formats
        if (data_dir / "metadata.jsonl").exists():
            # JSONL format
            return load_dataset("json", data_files=str(data_dir / "metadata.jsonl"))
        elif (data_dir / "captions.txt").exists():
            # Simple captions format
            return self._load_captions_dataset(data_dir)
        else:
            # Assume imagefolder format
            return load_dataset("imagefolder", data_dir=str(data_dir))
    
    def _load_captions_dataset(self, data_dir: Path) -> Dataset:
        """Load dataset with captions.txt file"""
        captions_file = data_dir / "captions.txt"
        images_dir = data_dir / "images"
        
        if not captions_file.exists():
            raise ValueError(f"Captions file not found: {captions_file}")
        
        # Read captions
        captions = {}
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '|' in line:
                    filename, caption = line.strip().split('|', 1)
                    captions[filename] = caption
        
        # Create dataset
        data = []
        for img_file in images_dir.glob("*.jpg"):
            if img_file.name in captions:
                data.append({
                    "image": str(img_file),
                    "text": captions[img_file.name]
                })
        
        return Dataset.from_list(data)
    
    def setup_model(self):
        """Setup the model with LoRA configuration"""
        logger.info(f"Loading base model: {self.base_model}")
        
        if self.use_sdxl:
            # Load SDXL pipeline
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                variant="fp16"
            )
            self.unet = self.pipeline.unet
            self.vae = self.pipeline.vae
            self.text_encoder = self.pipeline.text_encoder
            self.text_encoder_2 = self.pipeline.text_encoder_2
            self.tokenizer = self.pipeline.tokenizer
            self.tokenizer_2 = self.pipeline.tokenizer_2
        else:
            # Load SD v1.5 pipeline
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16
            )
            self.unet = self.pipeline.unet
            self.vae = self.pipeline.vae
            self.text_encoder = self.pipeline.text_encoder
            self.tokenizer = self.pipeline.tokenizer
        
        # Freeze VAE and text encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        if self.use_sdxl:
            self.text_encoder_2.requires_grad_(False)
        
        # Setup LoRA configuration
        lora_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,  # Alpha scaling
            target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,  # Use CAUSAL_LM for better compatibility
        )
        
        # Apply LoRA to UNet
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-08
        )
        
        # Setup scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_train_steps or 1000
        )
        
        # Prepare for distributed training
        self.unet, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.unet, self.optimizer, self.lr_scheduler
        )
        
        logger.info("Model setup completed")
    
    def preprocess_data(self, examples):
        """Preprocess data for training"""
        # Tokenize text
        if self.use_sdxl:
            # SDXL uses two text encoders
            tokenized = self.tokenizer(
                examples["text"],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            tokenized_2 = self.tokenizer_2(
                examples["text"],
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            examples["input_ids"] = tokenized.input_ids
            examples["input_ids_2"] = tokenized_2.input_ids
        else:
            tokenized = self.tokenizer(
                examples["text"],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            examples["input_ids"] = tokenized.input_ids
        
        return examples
    
    def training_step(self, batch):
        """Single training step"""
        # Get latents
        latents = batch["latents"]
        
        # Sample noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        
        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Get text embeddings
        if self.use_sdxl:
            encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]
            encoder_hidden_states_2 = self.text_encoder_2(batch["input_ids_2"])[0]
            # Combine embeddings for SDXL
            encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_2], dim=-1)
        else:
            encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]
        
        # Predict noise
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states
        ).sample
        
        # Calculate loss
        loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
        
        return loss
    
    def train(self, dataset):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Setup noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.base_model, subfolder="scheduler")
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        # Prepare dataloader
        dataloader = self.accelerator.prepare(dataloader)
        
        # Training loop
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            progress_bar = self.accelerator.init_progress_bar(
                dataloader,
                total=len(dataloader),
                desc=f"Epoch {epoch}"
            )
            
            for step, batch in enumerate(dataloader):
                # Skip steps if needed
                if self.max_train_steps and self.global_step >= self.max_train_steps:
                    break
                
                with self.accelerator.accumulate(self.unet):
                    loss = self.training_step(batch)
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                # Log progress
                if self.global_step % 10 == 0:
                    self.accelerator.log(
                        {
                            "train_loss": loss.detach().float(),
                            "lr": self.lr_scheduler.get_last_lr()[0],
                            "epoch": epoch,
                            "step": self.global_step,
                        }
                    )
                
                # Save checkpoint
                if self.global_step % self.save_steps == 0:
                    self.save_checkpoint()
                
                self.global_step += 1
                progress_bar.update(1)
            
            progress_bar.close()
        
        # Final save
        self.save_checkpoint()
        logger.info("Training completed!")
    
    def save_checkpoint(self):
        """Save model checkpoint"""
        # Create checkpoint directory
        checkpoint_dir = self.output_dir / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save LoRA weights
        self.unet.save_pretrained(checkpoint_dir)
        
        # Save training info
        training_info = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "base_model": self.base_model,
            "use_sdxl": self.use_sdxl,
            "resolution": self.resolution,
            "learning_rate": self.learning_rate,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(checkpoint_dir / "training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_dir}")
    
    def create_deployable_model(self, checkpoint_path: str, output_path: str):
        """Create a deployable model from checkpoint"""
        logger.info(f"Creating deployable model from {checkpoint_path}")
        
        # Load base pipeline
        if self.use_sdxl:
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                variant="fp16"
            )
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16
            )
        
        # Load LoRA weights
        pipeline.unet = PeftModel.from_pretrained(pipeline.unet, checkpoint_path)
        
        # Save deployable model
        pipeline.save_pretrained(output_path)
        logger.info(f"Deployable model saved: {output_path}")

def main():
    """Main training function"""
    
    # Configuration
    config = {
        "base_model": "runwayml/stable-diffusion-v1-5",  # or "stabilityai/stable-diffusion-xl-base-1.0" for SDXL
        "use_sdxl": False,  # Set to True for SDXL
        "resolution": 512,  # 512 for SD v1.5, 1024 for SDXL
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-4,
        "num_epochs": 1,
        "save_steps": 500,
        "max_train_steps": 1000,  # Set to None for unlimited
        "mixed_precision": "fp16",
        "output_dir": "custom_model_outputs"
    }
    
    # Dataset configurations
    dataset_configs = [
        {
            "type": "huggingface",
            "name": "laion/laion2B-en",
            "split": "train",
            "streaming": True,
            "filter": lambda x: len(x["text"]) > 10 and len(x["text"]) < 200  # Filter by caption length
        },
        # Add your custom dataset here
        # {
        #     "type": "imagefolder",
        #     "data_dir": "path/to/your/custom/dataset"
        # }
    ]
    
    # Initialize trainer
    trainer = CustomModelTrainer(**config)
    
    # Load datasets
    dataset = trainer.load_datasets(dataset_configs)
    
    # Setup model
    trainer.setup_model()
    
    # Start training
    trainer.train(dataset)
    
    # Create deployable model
    latest_checkpoint = max(
        trainer.output_dir.glob("checkpoint-*"),
        key=lambda x: int(x.name.split("-")[1])
    )
    trainer.create_deployable_model(
        str(latest_checkpoint),
        str(trainer.output_dir / "deployable_model")
    )
    
    logger.info("Training pipeline completed!")

if __name__ == "__main__":
    main() 