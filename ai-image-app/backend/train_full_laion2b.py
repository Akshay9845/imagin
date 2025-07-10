#!/usr/bin/env python3
"""
Full LAION-2B Training Pipeline
Trains on the complete 2 billion image dataset for production deployment
"""

import os
import json
import torch
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import gc

# Set MPS environment variables for Mac
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, Dataset
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import threading
from queue import Queue
import psutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('full_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FullLAION2BTrainer:
    def __init__(self, config_path: str = "full_training_config.json"):
        self.config = self.load_config(config_path)
        self.device = self.setup_device()
        self.setup_accelerator()
        
        # Training state
        self.global_step = 0
        self.total_samples_processed = 0
        self.start_time = time.time()
        self.best_loss = float('inf')
        
        # Create output directories
        self.setup_directories()
        
        logger.info(f"🚀 Full LAION-2B Training Setup Complete")
        logger.info(f"📊 Device: {self.device}")
        logger.info(f"🎯 Target: {self.config['training']['total_samples']:,} samples")
        
    def setup_device(self) -> str:
        """Setup device with MPS support for Mac"""
        if torch.backends.mps.is_available():
            device = "mps"
            logger.info("✅ MPS (Apple Silicon GPU) available and enabled")
        elif torch.cuda.is_available():
            device = "cuda"
            logger.info("✅ CUDA GPU available")
        else:
            device = "cpu"
            logger.warning("⚠️ Using CPU - training will be very slow!")
        
        return device
    
    def setup_accelerator(self):
        """Setup accelerator for distributed training"""
        self.accelerator = Accelerator(
            mixed_precision="fp16" if self.device == "cuda" else "no",
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
            log_with="tensorboard" if self.config["advanced"]["enable_tensorboard"] else None,
            project_dir="training_logs"
        )
    
    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            "full_training_outputs",
            "full_training_outputs/checkpoints",
            "full_training_outputs/final_model",
            "training_logs",
            "static"
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(exist_ok=True)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration"""
        if not Path(config_path).exists():
            self.create_default_config(config_path)
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    
    def create_default_config(self, config_path: str):
        """Create default configuration for full LAION-2B training"""
        config = {
            "training": {
                "base_model": "runwayml/stable-diffusion-v1-5",
                "resolution": 512,
                "batch_size": 2,
                "gradient_accumulation_steps": 8,
                "learning_rate": 5e-5,
                "total_samples": 2000000000,  # 2B samples
                "samples_per_epoch": 1000000,  # 1M samples per epoch
                "save_steps": 10000,
                "eval_steps": 5000,
                "logging_steps": 100,
                "mixed_precision": "fp16",
                "max_grad_norm": 1.0,
                "warmup_steps": 10000,
                "lr_scheduler": "cosine"
            },
            "lora": {
                "r": 32,
                "lora_alpha": 64,
                "target_modules": [
                    "to_q", "to_k", "to_v", "to_out.0",
                    "ff.net.0.proj", "ff.net.2"
                ],
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": "CAUSAL_LM"
            },
            "datasets": [
                {
                    "type": "huggingface",
                    "name": "laion/laion2B-en",
                    "split": "train",
                    "streaming": True,
                    "filter": "caption_length_10_200",
                    "shuffle_buffer": 10000
                }
            ],
            "advanced": {
                "enable_wandb": False,
                "enable_tensorboard": True,
                "gradient_checkpointing": True,
                "use_8bit_adam": False,
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_weight_decay": 1e-2,
                "adam_epsilon": 1e-08,
                "max_grad_norm": 1.0,
                "warmup_steps": 10000,
                "lr_scheduler": "cosine",
                "resume_from_checkpoint": None,
                "max_memory_usage": 0.8
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"📝 Created default config: {config_path}")
    
    def load_models(self):
        """Load base models and setup LoRA"""
        logger.info("📦 Loading base models...")
        
        # Load base model
        model_id = self.config["training"]["base_model"]
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Get components
        self.unet = self.pipeline.unet
        self.text_encoder = self.pipeline.text_encoder
        self.tokenizer = self.pipeline.tokenizer
        self.vae = self.pipeline.vae
        
        # Move to device
        self.unet.to(self.device)
        self.text_encoder.to(self.device)
        self.vae.to(self.device)
        
        # Setup LoRA
        lora_config = LoraConfig(
            r=self.config["lora"]["r"],
            lora_alpha=self.config["lora"]["lora_alpha"],
            target_modules=self.config["lora"]["target_modules"],
            lora_dropout=self.config["lora"]["lora_dropout"],
            bias=self.config["lora"]["bias"],
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()
        
        # Setup optimizer and scheduler
        self.setup_optimizer()
        
        logger.info("✅ Models loaded successfully")
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        # Optimizer
        self.optimizer = AdamW(
            self.unet.parameters(),
            lr=self.config["training"]["learning_rate"],
            betas=(self.config["advanced"]["adam_beta1"], self.config["advanced"]["adam_beta2"]),
            weight_decay=self.config["advanced"]["adam_weight_decay"],
            eps=self.config["advanced"]["adam_epsilon"]
        )
        
        # Scheduler
        total_steps = self.config["training"]["total_samples"] // (
            self.config["training"]["batch_size"] * self.config["training"]["gradient_accumulation_steps"]
        )
        
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-6
        )
        
        logger.info(f"📈 Optimizer setup: {total_steps:,} total steps")
    
    def prepare_dataset(self):
        """Prepare LAION-2B dataset with streaming"""
        logger.info("📚 Loading LAION-2B dataset...")
        
        dataset_config = self.config["datasets"][0]
        
        try:
            # Load dataset with streaming
            self.dataset = load_dataset(
                dataset_config["name"],
                split=dataset_config["split"],
                streaming=dataset_config["streaming"]
            )
            
            # Apply filters
            if "filter" in dataset_config:
                self.dataset = self.dataset.filter(
                    lambda x: len(x["text"]) >= 10 and len(x["text"]) <= 200
                )
            
            # Shuffle if specified
            if dataset_config.get("shuffle_buffer", 0) > 0:
                self.dataset = self.dataset.shuffle(
                    buffer_size=dataset_config["shuffle_buffer"]
                )
            
            logger.info(f"✅ Dataset loaded: {dataset_config['name']}")
            logger.info(f"📊 Streaming dataset ready")
            
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            raise
    
    def preprocess_batch(self, batch):
        """Preprocess a batch of data"""
        processed_batch = {
            "images": [],
            "texts": [],
            "input_ids": []
        }
        
        for item in batch:
            try:
                # Process image
                image = self.preprocess_image(item["url"])
                if image is None:
                    continue
                
                # Process text
                text = item["text"][:200]  # Limit text length
                input_ids = self.tokenize_text(text)
                
                processed_batch["images"].append(image)
                processed_batch["texts"].append(text)
                processed_batch["input_ids"].append(input_ids)
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to process item: {e}")
                continue
        
        if not processed_batch["images"]:
            return None
        
        # Convert to tensors
        processed_batch["images"] = torch.stack(processed_batch["images"])
        processed_batch["input_ids"] = torch.stack(processed_batch["input_ids"])
        
        return processed_batch
    
    def preprocess_image(self, image_url: str) -> Optional[torch.Tensor]:
        """Preprocess image from URL"""
        try:
            response = requests.get(image_url, timeout=5)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            
            # Resize to training resolution
            resolution = self.config["training"]["resolution"]
            image = image.resize((resolution, resolution))
            
            # Convert to tensor
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            return image_tensor
            
        except Exception as e:
            return None
    
    def tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize text for training"""
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        return tokens.input_ids.squeeze(0)
    
    def training_step(self, batch):
        """Single training step"""
        # Get batch data
        images = batch["images"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        
        # Encode text
        with torch.no_grad():
            text_embeddings = self.text_encoder(input_ids)[0]
        
        # Prepare latents
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215
        
        # Add noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (latents.shape[0],), device=self.device)
        noisy_latents = self.add_noise(latents, noise, timesteps)
        
        # Predict noise
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
        
        # Calculate loss
        loss = F.mse_loss(noise_pred, noise, reduction="mean")
        
        return loss
    
    def add_noise(self, latents, noise, timesteps):
        """Add noise to latents"""
        alpha = 1.0 - timesteps / 1000.0
        alpha = alpha.view(-1, 1, 1, 1)
        return alpha * latents + (1 - alpha) * noise
    
    def save_checkpoint(self, step: int, loss: float):
        """Save training checkpoint"""
        checkpoint_dir = f"full_training_outputs/checkpoints/step_{step}"
        Path(checkpoint_dir).mkdir(exist_ok=True)
        
        # Save LoRA weights
        self.unet.save_pretrained(checkpoint_dir)
        
        # Save training state
        training_state = {
            "step": step,
            "loss": loss,
            "total_samples": self.total_samples_processed,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.lr_scheduler.state_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(f"{checkpoint_dir}/training_state.json", 'w') as f:
            json.dump(training_state, f, indent=2)
        
        logger.info(f"💾 Checkpoint saved: {checkpoint_dir}")
    
    def log_progress(self, step: int, loss: float, samples_per_sec: float):
        """Log training progress"""
        elapsed_time = time.time() - self.start_time
        eta = (self.config["training"]["total_samples"] - self.total_samples_processed) / samples_per_sec
        
        progress = (self.total_samples_processed / self.config["training"]["total_samples"]) * 100
        
        logger.info(
            f"📊 Step {step:,} | "
            f"Loss: {loss:.4f} | "
            f"Samples: {self.total_samples_processed:,}/{self.config['training']['total_samples']:,} ({progress:.2f}%) | "
            f"Speed: {samples_per_sec:.1f} samples/sec | "
            f"ETA: {timedelta(seconds=int(eta))}"
        )
    
    def train(self):
        """Main training loop"""
        logger.info("🚀 Starting full LAION-2B training...")
        
        # Load models and dataset
        self.load_models()
        self.prepare_dataset()
        
        # Prepare accelerator
        self.unet, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.unet, self.optimizer, self.lr_scheduler
        )
        
        # Training loop
        batch_size = self.config["training"]["batch_size"]
        grad_accum_steps = self.config["training"]["gradient_accumulation_steps"]
        
        dataset_iter = iter(self.dataset)
        accumulated_loss = 0
        step_start_time = time.time()
        
        try:
            while self.total_samples_processed < self.config["training"]["total_samples"]:
                # Collect batch
                batch_data = []
                for _ in range(batch_size):
                    try:
                        item = next(dataset_iter)
                        batch_data.append(item)
                    except StopIteration:
                        # Restart dataset iterator
                        dataset_iter = iter(self.dataset)
                        item = next(dataset_iter)
                        batch_data.append(item)
                
                # Preprocess batch
                batch = self.preprocess_batch(batch_data)
                if batch is None:
                    continue
                
                # Training step
                loss = self.training_step(batch)
                loss = loss / grad_accum_steps
                
                # Backward pass
                self.accelerator.backward(loss)
                accumulated_loss += loss.item()
                
                # Gradient accumulation
                if (self.global_step + 1) % grad_accum_steps == 0:
                    # Gradient clipping
                    self.accelerator.clip_grad_norm_(self.unet.parameters(), self.config["training"]["max_grad_norm"])
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # Update counters
                    self.total_samples_processed += batch_size * grad_accum_steps
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config["training"]["logging_steps"] == 0:
                        avg_loss = accumulated_loss / grad_accum_steps
                        samples_per_sec = batch_size * grad_accum_steps / (time.time() - step_start_time)
                        self.log_progress(self.global_step, avg_loss, samples_per_sec)
                        
                        # Reset for next logging period
                        accumulated_loss = 0
                        step_start_time = time.time()
                    
                    # Save checkpoint
                    if self.global_step % self.config["training"]["save_steps"] == 0:
                        self.save_checkpoint(self.global_step, avg_loss)
                    
                    # Memory management
                    if self.global_step % 100 == 0:
                        self.cleanup_memory()
                
        except KeyboardInterrupt:
            logger.info("⚠️ Training interrupted by user")
        except Exception as e:
            logger.error(f"❌ Training error: {e}")
            raise
        finally:
            self.save_final_model()
    
    def cleanup_memory(self):
        """Clean up memory"""
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
        
        gc.collect()
    
    def save_final_model(self):
        """Save final trained model"""
        logger.info("💾 Saving final model...")
        
        final_dir = "full_training_outputs/final_model"
        Path(final_dir).mkdir(exist_ok=True)
        
        # Save LoRA weights
        self.unet.save_pretrained(final_dir)
        
        # Save training info
        training_info = {
            "total_steps": self.global_step,
            "total_samples": self.total_samples_processed,
            "final_loss": self.best_loss,
            "training_time": time.time() - self.start_time,
            "config": self.config,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(f"{final_dir}/training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"✅ Final model saved: {final_dir}")

def main():
    """Main function"""
    logger.info("🎯 Starting Full LAION-2B Training Pipeline")
    
    # Check system resources
    memory_gb = psutil.virtual_memory().total / (1024**3)
    logger.info(f"💻 System memory: {memory_gb:.1f} GB")
    
    if memory_gb < 16:
        logger.warning("⚠️ Low memory system detected. Consider reducing batch size.")
    
    # Start training
    trainer = FullLAION2BTrainer()
    trainer.train()

if __name__ == "__main__":
    main() 