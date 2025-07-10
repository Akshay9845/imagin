#!/usr/bin/env python3
"""
LAION-2B Training Script WITHOUT LoRA
Test base SD training flow to isolate the input_ids issue
"""

import os
import torch
import logging
from pathlib import Path

# Set MPS environment variables for Mac
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from diffusers import StableDiffusionPipeline, DDPMScheduler
from datasets import load_dataset
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NoLoRATrainer:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.global_step = 0
        self.total_samples_processed = 0
        
        Path("no_lora_outputs").mkdir(exist_ok=True)
        
    def load_models(self):
        """Load models WITHOUT LoRA"""
        logger.info("📦 Loading models (no LoRA)...")
        
        # Load pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        self.unet = pipe.unet
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
        )
        
        # Move to device (NO LoRA)
        self.unet.to(self.device)
        self.text_encoder.to(self.device)
        self.vae.to(self.device)
        
        # Make UNet trainable
        self.unet.train()
        for param in self.unet.parameters():
            param.requires_grad = True
        
        # Optimizer
        self.optimizer = AdamW(self.unet.parameters(), lr=1e-5)
        
        logger.info("✅ Models loaded (no LoRA)")
    
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
            sample_lower = {k.lower(): v for k, v in sample.items()}
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
            image = image.resize((512, 512))
            
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            return image_tensor
        except:
            return None
    
    def training_step(self, batch):
        """Single training step WITHOUT LoRA"""
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
        latents = latents * 0.18215
        
        # 4. Add noise to latents
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (latents.shape[0],), device=self.device
        ).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # 5. Predict noise (BASE UNet - no LoRA)
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
        logger.info("🚀 Starting training (no LoRA)...")
        
        self.load_models()
        self.load_dataset()
        
        dataset_iter = iter(self.dataset)
        target_samples = 100  # Small test
        
        while self.total_samples_processed < target_samples:
            try:
                # Get sample
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
                
                if image is None:
                    continue
                
                # Create batch
                batch = {
                    "images": image.unsqueeze(0),
                    "texts": [normalized["text"][:200]]
                }
                
                # Training step
                loss = self.training_step(batch)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update counters
                self.total_samples_processed += 1
                self.global_step += 1
                
                # Logging
                if self.global_step % 10 == 0:
                    logger.info(
                        f"📊 Step {self.global_step} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Samples: {self.total_samples_processed}/{target_samples}"
                    )
                
            except Exception as e:
                logger.error(f"Training error: {e}")
                import traceback
                traceback.print_exc()
                break
        
        logger.info("✅ Training completed!")

def main():
    trainer = NoLoRATrainer()
    trainer.train()

if __name__ == "__main__":
    main() 