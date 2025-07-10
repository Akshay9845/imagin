#!/usr/bin/env python3
"""
FINAL Working LAION-2B Training Script
Complete implementation with correct SD + LoRA training flow
NO input_ids errors - properly handles text encoding
"""

import os
import torch
import logging
from pathlib import Path
import time

# Set MPS environment variables for Mac
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model, TaskType
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

class FinalLAION2BTrainer:
    def __init__(self):
        self.device = self.setup_device()
        self.global_step = 0
        self.total_samples_processed = 0
        
        # Create output directories
        Path("final_training_outputs").mkdir(exist_ok=True)
        Path("final_training_outputs/checkpoints").mkdir(exist_ok=True)
        
        logger.info(f"🚀 Final LAION-2B Trainer initialized on {self.device}")
        
    def setup_device(self):
        """Setup device with MPS support"""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def load_models(self):
        """Load models and apply LoRA correctly"""
        logger.info("📦 Loading models...")
        
        # Load Stable Diffusion pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
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
        
        # Apply LoRA to UNet only (CORRECT target modules for SD)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # SD UNet modules
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,  # Correct for UNet
        )
        
        # Apply LoRA to UNet
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()
        
        # Move models to device
        self.unet.to(self.device)
        self.text_encoder.to(self.device)
        self.vae.to(self.device)
        
        # Setup optimizer for UNet parameters only
        self.optimizer = AdamW(self.unet.parameters(), lr=1e-4)
        
        logger.info("✅ Models loaded and LoRA applied to UNet only")
    
    def load_dataset(self):
        """Load LAION-2B dataset with streaming"""
        logger.info("📚 Loading LAION-2B dataset...")
        
        self.dataset = load_dataset(
            "laion/laion2B-en",
            split="train",
            streaming=True
        ).shuffle(buffer_size=1000)
        
        logger.info("✅ LAION-2B dataset loaded")
    
    def is_valid_sample(self, sample):
        """Validate sample data"""
        try:
            # Normalize keys to lowercase
            sample_lower = {k.lower(): v for k, v in sample.items()}
            
            # Extract text and URL with fallbacks
            text = sample_lower.get("text") or sample_lower.get("caption")
            url = sample_lower.get("url") or sample_lower.get("image_url")
            
            # Validation checks
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
            
            # Convert to tensor [C, H, W] normalized to [0, 1]
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            return image_tensor
        except:
            return None
    
    def training_step(self, images, captions):
        """Single training step with CORRECT text encoding flow"""
        
        # Move images to device
        images = images.to(self.device)
        
        # ✅ STEP 1: Tokenize captions (input_ids used HERE for text encoder)
        text_inputs = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        
        # ✅ STEP 2: Get text embeddings from CLIP text encoder
        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(input_ids)[0]
        
        # ✅ STEP 3: Encode images to latent space
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215  # SD scaling factor
        
        # ✅ STEP 4: Add noise to latents
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=self.device
        ).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # ✅ STEP 5: UNet forward pass (CORRECT: no input_ids!)
        noise_pred = self.unet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states
        ).sample
        
        # ✅ STEP 6: Compute loss
        loss = F.mse_loss(noise_pred, noise, reduction="mean")
        
        return loss
    
    def train(self):
        """Main training loop"""
        logger.info("🚀 Starting final LAION-2B training...")
        
        # Load models and dataset
        self.load_models()
        self.load_dataset()
        
        # Training parameters
        target_samples = 400  # Test with small number first
        batch_size = 1
        
        dataset_iter = iter(self.dataset)
        
        while self.total_samples_processed < target_samples:
            try:
                # Collect valid samples for batch
                batch_images = []
                batch_texts = []
                
                for _ in range(batch_size):
                    # Get next sample
                    try:
                        sample = next(dataset_iter)
                    except StopIteration:
                        dataset_iter = iter(self.dataset)
                        sample = next(dataset_iter)
                    
                    # Validate and process sample
                    if not self.is_valid_sample(sample):
                        continue
                    
                    normalized = self.normalize_sample(sample)
                    image = self.preprocess_image(normalized["url"])
                    
                    if image is not None:
                        batch_images.append(image)
                        batch_texts.append(normalized["text"][:200])  # Limit text length
                
                # Skip if no valid images in batch
                if not batch_images:
                    continue
                
                # Stack images into batch tensor
                batch_images_tensor = torch.stack(batch_images)
                
                # Training step with correct flow
                loss = self.training_step(batch_images_tensor, batch_texts)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update counters
                self.total_samples_processed += len(batch_images)
                self.global_step += 1
                
                # Logging
                if self.global_step % 10 == 0:
                    logger.info(
                        f"📊 Step {self.global_step} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Samples: {self.total_samples_processed}/{target_samples}"
                    )
                
                # Save checkpoint
                if self.global_step % 50 == 0:
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
        checkpoint_dir = f"final_training_outputs/checkpoints/step_{self.global_step}"
        Path(checkpoint_dir).mkdir(exist_ok=True)
        
        # Save LoRA weights
        self.unet.save_pretrained(checkpoint_dir)
        
        logger.info(f"💾 Checkpoint saved: step_{self.global_step}")
    
    def save_final_model(self):
        """Save final trained model"""
        final_dir = "final_training_outputs/final_model"
        Path(final_dir).mkdir(exist_ok=True)
        
        # Save final LoRA weights
        self.unet.save_pretrained(final_dir)
        
        logger.info("✅ Final model saved to final_training_outputs/final_model")

def main():
    """Main function"""
    trainer = FinalLAION2BTrainer()
    trainer.train()

if __name__ == "__main__":
    main() 