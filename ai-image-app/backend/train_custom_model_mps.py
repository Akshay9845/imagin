#!/usr/bin/env python3
"""
MPS-Compatible LoRA Training Script for Mac M3
Optimized for LAION-2B and custom datasets
"""

import os
import json
import torch
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Set MPS environment variables
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MPSLoRATrainer:
    def __init__(self, config_path: str = "training_config.json"):
        self.config = self.load_config(config_path)
        self.device = self.setup_device()
        self.accelerator = Accelerator(
            mixed_precision="fp16" if self.device == "cuda" else "no",
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"]
        )
        
        logger.info(f"🚀 Using device: {self.device}")
        logger.info(f"📊 Training config: {json.dumps(self.config, indent=2)}")
        
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
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    
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
        
        # Get UNet and Text Encoder
        self.unet = self.pipeline.unet
        self.text_encoder = self.pipeline.text_encoder
        self.tokenizer = self.pipeline.tokenizer
        self.vae = self.pipeline.vae
        
        # Move models to device
        self.unet.to(self.device)
        self.text_encoder.to(self.device)
        self.vae.to(self.device)
        
        # Setup LoRA configuration
        lora_config = LoraConfig(
            r=self.config["lora"]["r"],
            lora_alpha=self.config["lora"]["lora_alpha"],
            target_modules=self.config["lora"]["target_modules"],
            lora_dropout=self.config["lora"]["lora_dropout"],
            bias=self.config["lora"]["bias"],
            task_type=TaskType.FEATURE_EXTRACTION,  # Fixed for SD
        )
        
        # Apply LoRA to UNet
        self.unet = get_peft_model(self.unet, lora_config)
        
        # Print trainable parameters
        self.unet.print_trainable_parameters()
        
        logger.info("✅ Models loaded and LoRA applied successfully")
    
    def prepare_dataset(self):
        """Prepare LAION-2B dataset"""
        logger.info("📚 Loading LAION-2B dataset...")
        
        dataset_config = self.config["datasets"][0]
        dataset_name = dataset_config["name"]
        
        try:
            # Load dataset with streaming
            self.dataset = load_dataset(
                dataset_name,
                split=dataset_config["split"],
                streaming=dataset_config["streaming"]
            )
            
            # Apply filter if specified
            if "filter" in dataset_config:
                self.dataset = self.dataset.filter(
                    lambda x: len(x["text"]) >= 10 and len(x["text"]) <= 200
                )
            
            logger.info(f"✅ Dataset loaded: {dataset_name}")
            logger.info(f"📊 Dataset size: {len(self.dataset) if not dataset_config['streaming'] else 'Streaming'}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            raise
    
    def preprocess_image(self, image_url: str) -> torch.Tensor:
        """Preprocess image from URL"""
        try:
            response = requests.get(image_url, timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            
            # Resize to training resolution
            resolution = self.config["training"]["resolution"]
            image = image.resize((resolution, resolution))
            
            # Convert to tensor
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            return image_tensor
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load image {image_url}: {e}")
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
        images = batch["image"].to(self.device)
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
        # Simple linear noise schedule
        alpha = 1.0 - timesteps / 1000.0
        alpha = alpha.view(-1, 1, 1, 1)
        return alpha * latents + (1 - alpha) * noise
    
    def train(self):
        """Main training loop"""
        logger.info("🎯 Starting training...")
        
        # Setup optimizer and scheduler
        optimizer = AdamW(
            self.unet.parameters(),
            lr=self.config["training"]["learning_rate"],
            betas=(0.9, 0.999),
            weight_decay=1e-2
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config["training"]["max_train_steps"]
        )
        
        # Prepare for training
        self.unet.train()
        
        # Training loop
        progress_bar = tqdm(range(self.config["training"]["max_train_steps"]))
        total_loss = 0
        
        for step in progress_bar:
            try:
                # Get batch from dataset
                batch_data = next(iter(self.dataset))
                
                # Preprocess batch
                image_tensor = self.preprocess_image(batch_data["URL"])
                if image_tensor is None:
                    continue
                
                input_ids = self.tokenize_text(batch_data["TEXT"])
                
                # Create batch
                batch = {
                    "image": image_tensor.unsqueeze(0),
                    "input_ids": input_ids.unsqueeze(0)
                }
                
                # Training step
                loss = self.training_step(batch)
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (step + 1) % self.config["training"]["gradient_accumulation_steps"] == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item()
                
                # Update progress bar
                avg_loss = total_loss / (step + 1)
                progress_bar.set_description(f"Loss: {avg_loss:.4f}")
                
                # Save checkpoint
                if (step + 1) % self.config["training"]["save_steps"] == 0:
                    self.save_checkpoint(step + 1, avg_loss)
                
                # Log training info
                self.log_training_info(step + 1, avg_loss, True)
                
            except Exception as e:
                logger.error(f"❌ Training step failed: {e}")
                continue
        
        # Save final model
        self.save_final_model()
        logger.info("✅ Training completed!")
    
    def save_checkpoint(self, step: int, loss: float):
        """Save training checkpoint"""
        checkpoint_dir = Path("lora_weights")
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save LoRA weights
        checkpoint_path = checkpoint_dir / f"lora_step_{step}"
        self.unet.save_pretrained(checkpoint_path)
        
        # Save training info
        training_info = {
            "step": step,
            "loss": loss,
            "device": self.device,
            "lora_success": True,
            "timestamp": datetime.now().isoformat()
        }
        
        info_path = checkpoint_path / "lora_info.json"
        with open(info_path, 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def save_final_model(self):
        """Save final trained model"""
        final_path = Path("custom_model_outputs")
        final_path.mkdir(exist_ok=True)
        
        # Save LoRA weights
        self.unet.save_pretrained(final_path / "lora_model")
        
        # Save training config
        with open(final_path / "training_config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"🎉 Final model saved: {final_path}")
    
    def log_training_info(self, step: int, loss: float, success: bool):
        """Log training information"""
        log_dir = Path("training_logs")
        log_dir.mkdir(exist_ok=True)
        
        training_info = {
            "step": step,
            "loss": loss,
            "device": self.device,
            "lora_success": success,
            "timestamp": datetime.now().isoformat(),
            "config": self.config
        }
        
        log_file = log_dir / f"training_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w') as f:
            json.dump(training_info, f, indent=2)

def main():
    """Main training function"""
    try:
        # Initialize trainer
        trainer = MPSLoRATrainer()
        
        # Load models and setup LoRA
        trainer.load_models()
        
        # Prepare dataset
        trainer.prepare_dataset()
        
        # Start training
        trainer.train()
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 