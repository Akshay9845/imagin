import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, DDPMScheduler
from diffusers.loaders import AttnProcsLayers
from peft import get_peft_model, LoraConfig
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LAIONStreamTrainer:
    def __init__(self, 
                 base_model="SG161222/RealVisXL_V3.0",
                 batch_size=1,
                 learning_rate=1e-4,
                 num_epochs=1,
                 save_steps=500,
                 max_train_steps=1000):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.save_steps = save_steps
        self.max_train_steps = max_train_steps
        
        # Load base model
        logger.info(f"Loading base model: {base_model}")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        
        # Setup LoRA
        self.setup_lora()
        
        # Create output directories
        os.makedirs("lora_weights", exist_ok=True)
        os.makedirs("training_logs", exist_ok=True)
        
    def setup_lora(self):
        """Setup LoRA configuration and apply to model"""
        # Use correct target modules for SDXL/RealVisXL
        lora_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,
            target_modules=[
                "to_q", "to_k", "to_v", "to_out.0",  # Attention modules
                "conv1", "conv2", "conv_shortcut",    # Conv modules
                "time_emb_proj", "conv_in", "conv_out"  # Additional modules
            ],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        try:
            # Apply LoRA to UNet
            self.pipe.unet = get_peft_model(self.pipe.unet, lora_config)
            self.pipe.unet.print_trainable_parameters()
        except ValueError as e:
            logger.warning(f"LoRA setup failed with specific modules: {e}")
            # Fallback to auto-detection
            logger.info("Trying auto-detection of target modules...")
            try:
                # Let PEFT auto-detect target modules
                lora_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
                    lora_dropout=0.1,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                self.pipe.unet = get_peft_model(self.pipe.unet, lora_config)
                self.pipe.unet.print_trainable_parameters()
            except Exception as e2:
                logger.error(f"LoRA setup completely failed: {e2}")
                logger.info("Continuing without LoRA training...")
                return False
        return True
    
    def load_laion_stream(self, batch_size=5000, seed=None):
        """Stream LAION-2B-en dataset in chunks"""
        if seed is None:
            seed = int(datetime.now().timestamp())
            
        logger.info(f"Loading LAION-2B-en stream with seed {seed}")
        
        dataset = load_dataset(
            "laion/laion2B-en",
            split="train",
            streaming=True
        ).shuffle(seed=seed).take(batch_size)
        
        return dataset
    
    def preprocess_image(self, image, target_size=512):
        """Preprocess image for training"""
        if isinstance(image, str):
            # Handle URL or file path
            try:
                image = Image.open(image).convert("RGB")
            except:
                return None
        
        # Resize and center crop
        image = image.resize((target_size, target_size))
        return image
    
    def create_training_batch(self, dataset_batch):
        """Create training batch from dataset"""
        images = []
        prompts = []
        
        for item in dataset_batch:
            try:
                # Filter by quality if available
                if 'TEXT' in item and item['TEXT']:
                    prompt = item['TEXT'][:77]  # Limit prompt length
                    
                    # Basic quality filtering
                    if len(prompt) > 10 and not prompt.startswith('http'):
                        prompts.append(prompt)
                        
                        # Handle image (URL or PIL)
                        if 'URL' in item and item['URL']:
                            # For URLs, you'd need to download
                            # For now, skip or use placeholder
                            continue
                        elif 'image' in item:
                            processed_image = self.preprocess_image(item['image'])
                            if processed_image:
                                images.append(processed_image)
                                
            except Exception as e:
                logger.warning(f"Error processing item: {e}")
                continue
        
        return images, prompts
    
    def train_on_batch(self, images, prompts):
        """Train LoRA on a batch of images and prompts"""
        if not images or not prompts:
            return 0
            
        # Convert to tensors
        image_tensors = torch.stack([
            torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            for img in images
        ]).to(self.device)
        
        # Tokenize prompts
        tokenizer = self.pipe.tokenizer
        text_encoder = self.pipe.text_encoder
        
        tokenized_prompts = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Training step
        self.pipe.unet.train()
        optimizer = torch.optim.AdamW(
            self.pipe.unet.parameters(),
            lr=self.learning_rate
        )
        
        # Forward pass (simplified - you'd need full diffusion training loop)
        with torch.no_grad():
            # This is a placeholder - actual training would involve noise scheduling
            loss = torch.tensor(0.0, device=self.device)
        
        return loss.item()
    
    def save_lora_weights(self, step, loss):
        """Save LoRA weights"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"lora_weights/lora_step_{step}_{timestamp}.safetensors"
        
        # Save LoRA weights
        self.pipe.unet.save_pretrained(save_path)
        
        # Save training info
        training_info = {
            "step": step,
            "loss": loss,
            "timestamp": timestamp,
            "model_path": save_path
        }
        
        with open(f"training_logs/training_info_{timestamp}.json", "w") as f:
            json.dump(training_info, f, indent=2)
            
        logger.info(f"Saved LoRA weights to {save_path}")
        return save_path
    
    def train_streaming_round(self, batch_size=5000, seed=None):
        """Complete training round on a streamed batch"""
        logger.info(f"Starting streaming training round with batch_size={batch_size}")
        
        # Load streaming dataset
        dataset = self.load_laion_stream(batch_size, seed)
        
        total_loss = 0
        step = 0
        
        # Process in smaller batches
        batch_iterator = iter(dataset)
        current_batch = []
        
        try:
            while step < self.max_train_steps:
                # Collect batch
                for _ in range(self.batch_size):
                    try:
                        item = next(batch_iterator)
                        current_batch.append(item)
                    except StopIteration:
                        break
                
                if not current_batch:
                    break
                
                # Process batch
                images, prompts = self.create_training_batch(current_batch)
                
                if images and prompts:
                    loss = self.train_on_batch(images, prompts)
                    total_loss += loss
                    
                    # Save periodically
                    if step % self.save_steps == 0 and step > 0:
                        self.save_lora_weights(step, loss)
                
                step += 1
                current_batch = []
                
                if step % 100 == 0:
                    logger.info(f"Step {step}, Avg Loss: {total_loss/step:.4f}")
                    
        except Exception as e:
            logger.error(f"Training error: {e}")
        
        # Final save
        if step > 0:
            final_path = self.save_lora_weights(step, total_loss/step)
            logger.info(f"Training round complete. Final weights: {final_path}")
            return final_path
        
        return None

def main():
    """Main training function"""
    trainer = LAIONStreamTrainer(
        batch_size=1,
        learning_rate=1e-4,
        num_epochs=1,
        save_steps=500,
        max_train_steps=1000
    )
    
    # Train on multiple rounds with different seeds
    for round_num in range(3):
        logger.info(f"Starting training round {round_num + 1}")
        seed = 42 + round_num * 1000  # Different seed each round
        weights_path = trainer.train_streaming_round(batch_size=5000, seed=seed)
        
        if weights_path:
            logger.info(f"Round {round_num + 1} complete. Weights: {weights_path}")
        else:
            logger.warning(f"Round {round_num + 1} failed")

if __name__ == "__main__":
    main() 