#!/usr/bin/env python3
"""
Working LAION-2B Training Script
Clean implementation with correct SD + LoRA training flow
"""

import os
import json
import torch
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Set MPS environment variables for Mac
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkingLAION2BTrainer:
    def __init__(self):
        self.device = self.setup_device()
        self.config = self.create_config()
        
        # Training state
        self.global_step = 0
        self.total_samples_processed = 0
        
        # Create output directories
        Path("working_training_outputs").mkdir(exist_ok=True)
        Path("working_training_outputs/checkpoints").mkdir(exist_ok=True)
        
    def setup_device(self):
        """Setup device with MPS support"""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def create_config(self):
        """Create training configuration"""
        return {
            "base_model": "runwayml/stable-diffusion-v1-5",
            "resolution": 512,
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1e-4,
            "total_samples": 400,  # Small test
            "save_steps": 50,
            "logging_steps": 10,
            "lora_r": 16,
            "lora_alpha": 32,
        }
    
    def load_models(self):
        """Load models and apply LoRA to UNet only"""
        logger.info("📦 Loading models...")
        
        # Load pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            self.config["base_model"],
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        self.unet = pipe.unet
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.config["base_model"], subfolder="scheduler"
        )
        
        # Apply LoRA to UNet only
        lora_config = LoraConfig(
            r=self.config["lora_r"],
            lora_alpha=self.config["lora_alpha"],
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()
        
        # Move to device
        self.unet.to(self.device)
        self.text_encoder.to(self.device)
        self.vae.to(self.device)
        
        # Setup optimizer
        self.optimizer = AdamW(self.unet.parameters(), lr=self.config["learning_rate"])
        
        logger.info("✅ Models loaded successfully")
    
    def load_dataset(self):
        """Load LAION-2B dataset"""
        logger.info("📚 Loading LAION-2B dataset...")
        
        self.dataset = load_dataset(
            "laion/laion2B-en",
            split="train",
            streaming=True
        ).shuffle(buffer_size=1000)
        
        logger.info("✅ Dataset loaded")
    
    def is_valid_sample(self, sample):
        """Check if sample is valid"""
        try:
            # Normalize keys
            sample_lower = {k.lower(): v for k, v in sample.items()}
            
            # Get text and URL
            text = sample_lower.get("text") or sample_lower.get("caption")
            url = sample_lower.get("url") or sample_lower.get("image_url")
            
            if not text or not url:
                return False
            
            if len(text) < 5 or len(text) > 200:
                return False
            
            if not isinstance(url, str) or not url.startswith(('http://', 'https://')):
                return False
            
            return True
        except:
            return False
    
    def normalize_sample(self, sample):
        """Normalize sample keys"""
        sample_lower = {k.lower(): v for k, v in sample.items()}
        return {
            "text": sample_lower.get("text") or sample_lower.get("caption"),
            "url": sample_lower.get("url") or sample_lower.get("image_url")
        }
    
    def preprocess_image(self, image_url):
        """Download and preprocess image"""
        try:
            response = requests.get(image_url, timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image = image.resize((self.config["resolution"], self.config["resolution"]))
            
            # Convert to tensor
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            return image_tensor
        except:
            return None
    
    def training_step(self, batch):
        """Single training step with correct SD flow"""
        images = batch["images"].to(self.device)
        captions = batch["texts"]
        
        # 1. Tokenize captions
        inputs = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.device)
        
        input_ids = inputs.input_ids
        
        # 2. Get CLIP text embeddings
        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(input_ids)[0]
        
        # 3. Encode images into latent space
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215  # Scale factor for SD
        
        # 4. Add noise to latents
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (latents.shape[0],), device=self.device
        ).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # 5. Predict noise (✅ CORRECT: no input_ids to UNet)
        noise_pred = self.unet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states
        ).sample
        
        # 6. Compute loss
        loss = F.mse_loss(noise_pred, noise, reduction="mean")
        
        return loss
    
    def train(self):
        """Main training loop"""
        logger.info("🚀 Starting training...")
        
        self.load_models()
        self.load_dataset()
        
        dataset_iter = iter(self.dataset)
        
        while self.total_samples_processed < self.config["total_samples"]:
            try:
                # Collect batch
                batch_images = []
                batch_texts = []
                
                for _ in range(self.config["batch_size"]):
                    # Get next sample
                    try:
                        sample = next(dataset_iter)
                    except StopIteration:
                        dataset_iter = iter(self.dataset)
                        sample = next(dataset_iter)
                    
                    # Validate and process
                    if not self.is_valid_sample(sample):
                        continue
                    
                    normalized = self.normalize_sample(sample)
                    image = self.preprocess_image(normalized["url"])
                    
                    if image is not None:
                        batch_images.append(image)
                        batch_texts.append(normalized["text"][:200])
                
                if not batch_images:
                    continue
                
                # Create batch
                batch = {
                    "images": torch.stack(batch_images),
                    "texts": batch_texts
                }
                
                # Training step
                loss = self.training_step(batch)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update counters
                self.total_samples_processed += len(batch_images)
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config["logging_steps"] == 0:
                    logger.info(
                        f"📊 Step {self.global_step} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Samples: {self.total_samples_processed}/{self.config['total_samples']}"
                    )
                
                # Save checkpoint
                if self.global_step % self.config["save_steps"] == 0:
                    self.save_checkpoint()
                
            except Exception as e:
                logger.error(f"Training error: {e}")
                continue
        
        logger.info("✅ Training completed!")
        self.save_final_model()
    
    def save_checkpoint(self):
        """Save checkpoint"""
        checkpoint_dir = f"working_training_outputs/checkpoints/step_{self.global_step}"
        Path(checkpoint_dir).mkdir(exist_ok=True)
        self.unet.save_pretrained(checkpoint_dir)
        logger.info(f"💾 Checkpoint saved: step_{self.global_step}")
    
    def save_final_model(self):
        """Save final model"""
        final_dir = "working_training_outputs/final_model"
        Path(final_dir).mkdir(exist_ok=True)
        self.unet.save_pretrained(final_dir)
        logger.info("✅ Final model saved")

def main():
    trainer = WorkingLAION2BTrainer()
    trainer.train()

if __name__ == "__main__":
    main() 