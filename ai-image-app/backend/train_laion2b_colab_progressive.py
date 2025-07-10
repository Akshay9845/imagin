#!/usr/bin/env python3
"""
Progressive LAION-2B Training Script for Google Colab
Smart training approach: Start small, test quality, scale up
Optimized for Colab Free with practical sample sizes
"""

import os
import torch
import logging
from pathlib import Path
import time
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProgressiveLAION2BTrainer:
    def __init__(self):
        """Initialize progressive trainer for Colab Free"""
        self.setup_device()
        
        # Progressive training phases
        self.training_phases = [
            {"name": "Phase 1 - Quick Test", "samples": 1000, "time": "5 minutes", "purpose": "Basic testing"},
            {"name": "Phase 2 - Quality Training", "samples": 5000, "time": "15 minutes", "purpose": "Good quality"},
            {"name": "Phase 3 - Enhanced Training", "samples": 10000, "time": "30 minutes", "purpose": "Very good quality"},
            {"name": "Phase 4 - Professional", "samples": 50000, "time": "2-3 hours", "purpose": "Professional quality"}
        ]
        
        # Current phase settings
        self.current_phase = 0
        self.batch_size = 8
        self.image_size = 512
        self.max_text_length = 77
        
        # Training state
        self.global_step = 0
        self.total_samples_processed = 0
        self.start_time = time.time()
        self.phase_start_time = time.time()
        
        # Quality metrics
        self.loss_history = []
        self.quality_threshold = 0.02  # Good loss threshold
        
        # Create output directories
        Path("progressive_training_outputs").mkdir(exist_ok=True)
        Path("progressive_training_outputs/checkpoints").mkdir(exist_ok=True)
        Path("progressive_training_outputs/phases").mkdir(exist_ok=True)
        
        logger.info(f"🚀 Progressive LAION-2B Trainer initialized on {self.device}")
        logger.info(f"📊 Training Phases: {len(self.training_phases)} phases planned")
        
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
        logger.info("📦 Loading models for progressive training...")
        
        from diffusers import StableDiffusionPipeline, DDPMScheduler
        from peft import LoraConfig, get_peft_model
        from torch.optim import AdamW
        from torch.cuda.amp import GradScaler
        
        # Load Stable Diffusion pipeline with optimizations
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
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
        
        # Apply LoRA with progressive settings
        lora_config = LoraConfig(
            r=16,  # Start with moderate rank
            lora_alpha=32,
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
        
        # Setup optimizer
        self.optimizer = AdamW(self.unet.parameters(), lr=2e-4, weight_decay=0.01)
        
        # Setup mixed precision training
        self.scaler = GradScaler()
        
        logger.info("✅ Models loaded and optimized for progressive training")
    
    def load_dataset(self):
        """Load LAION-2B dataset with streaming"""
        logger.info("📚 Loading LAION-2B dataset...")
        
        from datasets import load_dataset
        
        self.dataset = load_dataset(
            "laion/laion2B-en",
            split="train",
            streaming=True
        ).shuffle(buffer_size=10000)
        
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
            import requests
            from PIL import Image
            from io import BytesIO
            import numpy as np
            
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
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(self.preprocess_image, url) for url in image_urls]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        return [r for r in results if r is not None]
    
    def training_step(self, images, captions):
        """Single training step with mixed precision"""
        import torch.nn.functional as F
        from torch.cuda.amp import autocast
        
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
    
    def evaluate_quality(self):
        """Evaluate model quality and decide next phase"""
        if len(self.loss_history) < 10:
            return "continue"  # Need more data
        
        recent_losses = self.loss_history[-10:]  # Last 10 losses
        avg_loss = sum(recent_losses) / len(recent_losses)
        
        logger.info(f"📊 Quality Assessment:")
        logger.info(f"   Recent average loss: {avg_loss:.4f}")
        logger.info(f"   Quality threshold: {self.quality_threshold:.4f}")
        
        if avg_loss < self.quality_threshold:
            logger.info("✅ Quality threshold met! Ready for next phase.")
            return "proceed"
        else:
            logger.info("⚠️ Quality threshold not met. Continue current phase.")
            return "continue"
    
    def save_phase_checkpoint(self):
        """Save checkpoint for current phase"""
        phase = self.training_phases[self.current_phase]
        phase_dir = f"progressive_training_outputs/phases/{phase['name'].replace(' ', '_').replace('-', '_')}"
        Path(phase_dir).mkdir(exist_ok=True)
        
        # Save LoRA weights
        self.unet.save_pretrained(phase_dir)
        
        # Save phase info
        phase_info = {
            "phase_name": phase["name"],
            "samples_processed": self.total_samples_processed,
            "global_step": self.global_step,
            "phase_time": time.time() - self.phase_start_time,
            "average_loss": sum(self.loss_history[-10:]) / len(self.loss_history[-10:]) if self.loss_history else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(f"{phase_dir}/phase_info.json", 'w') as f:
            json.dump(phase_info, f, indent=2)
        
        logger.info(f"💾 Phase checkpoint saved: {phase['name']}")
    
    def train_phase(self):
        """Train for current phase"""
        phase = self.training_phases[self.current_phase]
        target_samples = phase["samples"]
        
        logger.info(f"🚀 Starting {phase['name']}")
        logger.info(f"📊 Target: {target_samples:,} samples")
        logger.info(f"⏱️ Expected time: {phase['time']}")
        logger.info(f"🎯 Purpose: {phase['purpose']}")
        
        # Reset phase timer
        self.phase_start_time = time.time()
        
        # Load models and dataset if first phase
        if self.current_phase == 0:
            self.load_models()
            self.load_dataset()
        
        dataset_iter = iter(self.dataset)
        phase_samples_processed = 0
        
        while phase_samples_processed < target_samples:
            try:
                # Collect valid samples for batch
                batch_samples = []
                batch_urls = []
                batch_texts = []
                
                # Collect multiple samples at once
                for _ in range(self.batch_size * 3):
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
                samples_in_batch = len(batch_images)
                self.total_samples_processed += samples_in_batch
                phase_samples_processed += samples_in_batch
                self.global_step += 1
                
                # Track loss history
                self.loss_history.append(loss.item())
                if len(self.loss_history) > 100:  # Keep last 100 losses
                    self.loss_history.pop(0)
                
                # Logging
                if self.global_step % 5 == 0:
                    elapsed_time = time.time() - self.phase_start_time
                    samples_per_sec = phase_samples_processed / elapsed_time if elapsed_time > 0 else 0
                    recent_loss = sum(self.loss_history[-5:]) / len(self.loss_history[-5:])
                    
                    logger.info(
                        f"📊 Phase {self.current_phase + 1} | "
                        f"Step {self.global_step} | "
                        f"Loss: {recent_loss:.4f} | "
                        f"Phase Samples: {phase_samples_processed:,}/{target_samples:,} | "
                        f"Speed: {samples_per_sec:.1f} samples/sec"
                    )
                
                # Save checkpoint every 50 steps
                if self.global_step % 50 == 0:
                    self.save_phase_checkpoint()
                
                # Quality check every 100 steps
                if self.global_step % 100 == 0:
                    quality_decision = self.evaluate_quality()
                    if quality_decision == "proceed" and self.current_phase < len(self.training_phases) - 1:
                        logger.info("🎯 Quality target reached! Moving to next phase...")
                        break
                
            except Exception as e:
                logger.error(f"Training error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save final phase checkpoint
        self.save_phase_checkpoint()
        
        # Phase completion summary
        phase_time = time.time() - self.phase_start_time
        logger.info(f"✅ {phase['name']} completed!")
        logger.info(f"⏱️ Phase time: {phase_time/60:.1f} minutes")
        logger.info(f"📊 Total samples processed: {self.total_samples_processed:,}")
        
        return phase_samples_processed >= target_samples
    
    def train_progressive(self):
        """Main progressive training loop"""
        logger.info("🚀 Starting Progressive LAION-2B Training...")
        logger.info("📋 Training Strategy:")
        for i, phase in enumerate(self.training_phases):
            logger.info(f"   {i+1}. {phase['name']}: {phase['samples']:,} samples ({phase['time']})")
        
        for phase_idx in range(len(self.training_phases)):
            self.current_phase = phase_idx
            phase = self.training_phases[phase_idx]
            
            logger.info("\n" + "="*60)
            logger.info(f"🎯 PHASE {phase_idx + 1}/{len(self.training_phases)}: {phase['name']}")
            logger.info("="*60)
            
            # Train current phase
            phase_completed = self.train_phase()
            
            if not phase_completed:
                logger.warning(f"⚠️ Phase {phase_idx + 1} did not complete. Stopping training.")
                break
            
            # Ask user if they want to continue to next phase
            if phase_idx < len(self.training_phases) - 1:
                logger.info("\n💡 Phase {} completed successfully!".format(phase_idx + 1))
                logger.info(f"📊 Next phase: {self.training_phases[phase_idx + 1]['name']}")
                logger.info(f"⏱️ Expected time: {self.training_phases[phase_idx + 1]['time']}")
                
                # In Colab, we'll auto-continue, but log the decision
                logger.info("🔄 Auto-continuing to next phase...")
                logger.info("💡 You can stop training anytime by interrupting the kernel")
        
        logger.info("\n🎉 Progressive training completed!")
        self.save_final_model()
    
    def save_final_model(self):
        """Save final trained model"""
        final_dir = "progressive_training_outputs/final_model"
        Path(final_dir).mkdir(exist_ok=True)
        
        # Save final LoRA weights
        self.unet.save_pretrained(final_dir)
        
        # Save comprehensive training info
        training_info = {
            "total_phases_completed": self.current_phase + 1,
            "total_steps": self.global_step,
            "total_samples": self.total_samples_processed,
            "total_training_time": time.time() - self.start_time,
            "final_average_loss": sum(self.loss_history[-10:]) / len(self.loss_history[-10:]) if self.loss_history else 0,
            "device": self.device,
            "batch_size": self.batch_size,
            "image_size": self.image_size,
            "phases": self.training_phases[:self.current_phase + 1],
            "timestamp": datetime.now().isoformat()
        }
        
        with open(f"{final_dir}/training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info("✅ Final model saved to progressive_training_outputs/final_model")
        logger.info(f"📊 Training Summary:")
        logger.info(f"   Total phases: {self.current_phase + 1}")
        logger.info(f"   Total samples: {self.total_samples_processed:,}")
        logger.info(f"   Total time: {(time.time() - self.start_time)/60:.1f} minutes")
        logger.info(f"   Final loss: {training_info['final_average_loss']:.4f}")

def main():
    """Main function"""
    trainer = ProgressiveLAION2BTrainer()
    trainer.train_progressive()

if __name__ == "__main__":
    main() 