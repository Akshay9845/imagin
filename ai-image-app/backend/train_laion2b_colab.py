#!/usr/bin/env python3
"""
LAION-2B Training Script for Google Colab
Optimized for cloud GPUs (T4, A100, H100) with maximum speed
"""

import os
import torch
import logging
from pathlib import Path
import time
import json
from datetime import datetime

# Install required packages for Colab
def install_requirements():
    """Install required packages for Colab"""
    import subprocess
    import sys
    
    packages = [
        "diffusers",
        "transformers", 
        "accelerate",
        "peft",
        "datasets",
        "xformers",  # For faster attention
        "webdataset",  # For efficient data loading
        "torchvision",
        "pillow",
        "requests"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
        except:
            pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ColabLAION2BTrainer:
    def __init__(self):
        """Initialize trainer optimized for Colab GPUs"""
        self.setup_device()
        
        # Speed optimizations for cloud GPUs
        self.batch_size = 8  # Larger batch size for powerful GPUs
        self.image_size = 512  # Full resolution for better quality
        self.max_text_length = 77
        self.target_samples = 5000  # More samples for better training
        
        # Training state
        self.global_step = 0
        self.total_samples_processed = 0
        self.start_time = time.time()
        
        # Create output directories
        Path("colab_training_outputs").mkdir(exist_ok=True)
        Path("colab_training_outputs/checkpoints").mkdir(exist_ok=True)
        
        logger.info(f"🚀 Colab LAION-2B Trainer initialized on {self.device}")
        logger.info(f"⚡ GPU optimizations: batch_size={self.batch_size}, image_size={self.image_size}")
        
    def setup_device(self):
        """Setup device for Colab GPU"""
        if torch.cuda.is_available():
            self.device = "cuda"
            # Enable memory efficient attention
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            logger.info(f"✅ CUDA GPU available: {torch.cuda.get_device_name()}")
            logger.info(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.device = "cpu"
            logger.warning("⚠️ No CUDA GPU found, using CPU (very slow!)")
    
    def load_models(self):
        """Load models optimized for Colab GPUs"""
        logger.info("📦 Loading models for Colab...")
        
        # Load Stable Diffusion pipeline with optimizations
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,  # Use FP16 for speed
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True
        )
        
        # Extract components
        self.unet = pipe.unet
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        
        # Load noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
        )
        
        # Apply LoRA with optimized settings for cloud GPUs
        lora_config = LoraConfig(
            r=32,  # Higher rank for better quality
            lora_alpha=64,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.1,
            bias="none",
        )
        
        # Apply LoRA to UNet
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()
        
        # Move models to device
        self.unet.to(self.device)
        self.text_encoder.to(self.device)
        self.vae.to(self.device)
        
        # Enable gradient checkpointing for memory efficiency
        self.unet.enable_gradient_checkpointing()
        
        # Setup optimizer with higher learning rate for faster convergence
        self.optimizer = AdamW(self.unet.parameters(), lr=2e-4, weight_decay=0.01)
        
        # Setup mixed precision training
        self.scaler = GradScaler()
        
        logger.info("✅ Models loaded and optimized for Colab GPU")
        logger.info(f"⚡ Mixed precision: enabled")
        logger.info(f"⚡ Gradient checkpointing: enabled")
    
    def load_dataset(self):
        """Load LAION-2B dataset with streaming"""
        logger.info("📚 Loading LAION-2B dataset...")
        
        self.dataset = load_dataset(
            "laion/laion2B-en",
            split="train",
            streaming=True
        ).shuffle(buffer_size=10000)  # Larger buffer for better shuffling
        
        logger.info("✅ LAION-2B dataset loaded")
    
    def is_valid_sample(self, sample):
        """Validate sample data (optimized)"""
        try:
            sample_lower = {k.lower(): v for k, v in sample.items()}
            text = sample_lower.get("text") or sample_lower.get("caption")
            url = sample_lower.get("url") or sample_lower.get("image_url")
            
            # Quick validation
            return bool(text and url and isinstance(url, str) and url.startswith(('http://', 'https://')))
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
            image = image.resize((self.image_size, self.image_size))
            
            # Convert to tensor [C, H, W] normalized to [0, 1]
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            return image_tensor
        except:
            return None
    
    def preprocess_images_parallel(self, image_urls):
        """Download and preprocess multiple images in parallel"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:  # More workers for cloud
            futures = [executor.submit(self.preprocess_image, url) for url in image_urls]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        return [r for r in results if r is not None]
    
    def training_step(self, images, captions):
        """Single training step with mixed precision"""
        
        # Move images to device
        images = images.to(self.device, dtype=torch.float16)
        
        # Tokenize captions
        text_inputs = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        
        # Get text embeddings
        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(input_ids)[0]
        
        # Encode images to latent space
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215
        
        # Add noise to latents
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=self.device
        ).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # UNet forward pass with mixed precision
        with autocast():
            noise_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states
            ).sample
            loss = F.mse_loss(noise_pred, noise, reduction="mean")
        
        return loss
    
    def train(self):
        """Main training loop optimized for Colab"""
        logger.info("🚀 Starting Colab LAION-2B training...")
        
        # Load models and dataset
        self.load_models()
        self.load_dataset()
        
        dataset_iter = iter(self.dataset)
        
        while self.total_samples_processed < self.target_samples:
            try:
                # Collect valid samples for batch
                batch_samples = []
                batch_urls = []
                batch_texts = []
                
                # Collect multiple samples at once
                for _ in range(self.batch_size * 3):  # Get extra samples
                    try:
                        sample = next(dataset_iter)
                    except StopIteration:
                        dataset_iter = iter(self.dataset)
                        sample = next(dataset_iter)
                    
                    # Validate sample
                    if self.is_valid_sample(sample):
                        normalized = self.normalize_sample(sample)
                        batch_samples.append(normalized)
                        batch_urls.append(normalized["url"])
                        batch_texts.append(normalized["text"][:self.max_text_length])
                    
                    # Stop if we have enough samples
                    if len(batch_samples) >= self.batch_size:
                        break
                
                # Process images in parallel
                if batch_urls:
                    batch_images = self.preprocess_images_parallel(batch_urls[:self.batch_size])
                    batch_texts = batch_texts[:len(batch_images)]
                else:
                    batch_images = []
                
                # Skip if no valid images in batch
                if not batch_images:
                    continue
                
                # Stack images into batch tensor
                batch_images_tensor = torch.stack(batch_images)
                
                # Training step
                loss = self.training_step(batch_images_tensor, batch_texts)
                
                # Backward pass with mixed precision
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Update counters
                self.total_samples_processed += len(batch_images)
                self.global_step += 1
                
                # Logging
                if self.global_step % 2 == 0:  # More frequent logging
                    elapsed_time = time.time() - self.start_time
                    samples_per_sec = self.total_samples_processed / elapsed_time if elapsed_time > 0 else 0
                    logger.info(
                        f"📊 Step {self.global_step} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Samples: {self.total_samples_processed}/{self.target_samples} | "
                        f"Speed: {samples_per_sec:.1f} samples/sec"
                    )
                
                # Save checkpoint
                if self.global_step % 100 == 0:
                    self.save_checkpoint()
                
            except Exception as e:
                logger.error(f"Training error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        logger.info("✅ Training completed!")
        self.save_final_model()
    
    def save_checkpoint(self):
        """Save training checkpoint"""
        checkpoint_dir = f"colab_training_outputs/checkpoints/step_{self.global_step}"
        Path(checkpoint_dir).mkdir(exist_ok=True)
        
        # Save LoRA weights
        self.unet.save_pretrained(checkpoint_dir)
        
        logger.info(f"💾 Checkpoint saved: step_{self.global_step}")
    
    def save_final_model(self):
        """Save final trained model"""
        final_dir = "colab_training_outputs/final_model"
        Path(final_dir).mkdir(exist_ok=True)
        
        # Save final LoRA weights
        self.unet.save_pretrained(final_dir)
        
        # Save training info
        training_info = {
            "total_steps": self.global_step,
            "total_samples": self.total_samples_processed,
            "training_time": time.time() - self.start_time,
            "timestamp": datetime.now().isoformat(),
            "device": self.device,
            "batch_size": self.batch_size,
            "image_size": self.image_size
        }
        
        with open(f"{final_dir}/training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info("✅ Final model saved to colab_training_outputs/final_model")

def main():
    """Main function"""
    # Install requirements first
    install_requirements()
    
    # Import after installation
    global StableDiffusionPipeline, DDPMScheduler, LoraConfig, get_peft_model
    global load_dataset, AdamW, autocast, GradScaler, F, np, Image, requests, BytesIO, concurrent
    
    from diffusers import StableDiffusionPipeline, DDPMScheduler
    from peft import LoraConfig, get_peft_model
    from datasets import load_dataset
    from torch.optim import AdamW
    from torch.cuda.amp import autocast, GradScaler
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    import requests
    from io import BytesIO
    import concurrent.futures
    
    trainer = ColabLAION2BTrainer()
    trainer.train()

if __name__ == "__main__":
    main() 