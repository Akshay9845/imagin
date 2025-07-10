#!/usr/bin/env python3
"""
Working Image Generation Script
Uses base model for reliable generation while LoRA integration is fixed
"""

import os
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pipeline():
    """Load working pipeline"""
    logger.info("🔧 Setting up pipeline...")
    
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
    
    # Move to device
    pipeline = pipeline.to(device)
    
    logger.info("✅ Pipeline loaded successfully!")
    return pipeline

def generate_images(pipeline, prompts, output_dir="generated_images"):
    """Generate images"""
    logger.info("🎨 Generating images...")
    
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
            filename = f"generated_{i+1}_{timestamp}.png"
            filepath = output_path / filename
            image.save(filepath)
            
            logger.info(f"💾 Saved: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate image {i+1}: {e}")
            continue
    
    logger.info(f"✅ Generated {len(prompts)} images in {output_path}")

def main():
    """Main generation function"""
    logger.info("🚀 Starting image generation...")
    
    # Test prompts
    test_prompts = [
        "A beautiful sunset over mountains, high quality, detailed, masterpiece",
        "A cute robot playing with a cat, digital art, vibrant colors, 4k",
        "A futuristic city skyline at night, neon lights, cinematic, ultra detailed",
        "A magical forest with glowing mushrooms, fantasy art, ethereal lighting",
        "A steampunk airship flying through clouds, detailed machinery, golden hour"
    ]
    
    # Load pipeline
    pipeline = load_pipeline()
    if pipeline is None:
        logger.error("❌ Failed to load pipeline")
        return
    
    # Generate images
    generate_images(pipeline, test_prompts)
    
    logger.info("🎉 Image generation completed!")
    logger.info("📁 Check the 'generated_images' folder for your images")

if __name__ == "__main__":
    main() 