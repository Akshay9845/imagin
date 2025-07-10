#!/usr/bin/env python3
"""
Generate Images with Custom LoRA Model
Properly loads and uses the trained LoRA weights
"""

import os
import torch
import json
from pathlib import Path
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_custom_pipeline():
    """Load pipeline with custom LoRA weights"""
    logger.info("🔧 Setting up custom LoRA pipeline...")
    
    # Setup device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"🔧 Using device: {device}")
    
    # Load base pipeline
    logger.info("📦 Loading base pipeline...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Find LoRA weights
    lora_weights_path = Path("lora_weights/lora_step_50_20250707_231755")
    if not lora_weights_path.exists():
        logger.error("❌ LoRA weights not found")
        return None
    
    logger.info(f"🔗 Loading LoRA weights from: {lora_weights_path}")
    
    # Load LoRA weights properly
    try:
        # Load the LoRA configuration
        with open(lora_weights_path / "adapter_config.json", 'r') as f:
            lora_config = json.load(f)
        
        logger.info(f"📋 LoRA config: {lora_config}")
        
        # Apply LoRA to UNet
        pipeline.unet = PeftModel.from_pretrained(
            pipeline.unet,
            lora_weights_path
        )
        
        # Move to device
        pipeline = pipeline.to(device)
        
        logger.info("✅ Custom LoRA pipeline loaded successfully!")
        return pipeline
        
    except Exception as e:
        logger.error(f"❌ Failed to load LoRA: {e}")
        return None

def generate_images(pipeline, prompts, output_dir="custom_generated"):
    """Generate images with custom LoRA model"""
    logger.info("🎨 Generating images with custom LoRA...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate images
    for i, prompt in enumerate(prompts):
        logger.info(f"🎯 Generating {i+1}/{len(prompts)}: {prompt}")
        
        try:
            # Generate image
            image = pipeline(
                prompt=prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                height=512,
                width=512
            ).images[0]
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"custom_lora_{i+1}_{timestamp}.png"
            filepath = output_path / filename
            image.save(filepath)
            
            logger.info(f"💾 Saved: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate image {i+1}: {e}")
            continue
    
    logger.info(f"✅ Generated {len(prompts)} images in {output_path}")

def main():
    """Main generation function"""
    logger.info("🚀 Starting custom LoRA image generation...")
    
    # Test prompts
    test_prompts = [
        "A beautiful sunset over mountains, high quality, detailed, masterpiece",
        "A cute robot playing with a cat, digital art, vibrant colors, 4k",
        "A futuristic city skyline at night, neon lights, cinematic, ultra detailed",
        "A magical forest with glowing mushrooms, fantasy art, ethereal lighting",
        "A steampunk airship flying through clouds, detailed machinery, golden hour"
    ]
    
    # Load custom pipeline
    pipeline = load_custom_pipeline()
    if pipeline is None:
        logger.error("❌ Failed to load custom pipeline")
        return
    
    # Generate images
    generate_images(pipeline, test_prompts)
    
    logger.info("🎉 Custom LoRA generation completed!")
    logger.info("📁 Check the 'custom_generated' folder for your images")

if __name__ == "__main__":
    main() 