#!/usr/bin/env python3
"""
Test Custom LoRA Model
Verifies that the trained LoRA model works correctly
"""

import os
import torch
import json
from pathlib import Path
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_custom_model():
    """Test the custom LoRA model"""
    logger.info("🧪 Testing custom LoRA model...")
    
    # Check if custom model exists
    custom_model_path = Path("custom_model_outputs/lora_model")
    lora_weights_path = Path("lora_weights/lora_step_50_20250707_231755")
    
    if custom_model_path.exists():
        model_path = custom_model_path
    elif lora_weights_path.exists():
        model_path = lora_weights_path
        logger.info(f"✅ Using existing LoRA weights: {model_path}")
    else:
        logger.error("❌ Custom model not found. Please train the model first.")
        return False
    
    # Setup device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"🔧 Using device: {device}")
    
    try:
        # Load base pipeline
        logger.info("📦 Loading base pipeline...")
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Load LoRA weights
        logger.info("🔗 Loading LoRA weights...")
        pipeline.unet = PeftModel.from_pretrained(
            pipeline.unet,
            model_path
        )
        
        # Move to device
        pipeline = pipeline.to(device)
        
        # Test generation
        logger.info("🎨 Testing image generation...")
        test_prompts = [
            "A beautiful sunset over mountains, high quality, detailed",
            "A cute robot playing with a cat, digital art, vibrant colors",
            "A futuristic city skyline at night, neon lights, cinematic"
        ]
        
        for i, prompt in enumerate(test_prompts):
            logger.info(f"🎯 Generating: {prompt}")
            
            image = pipeline(
                prompt=prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                height=512,
                width=512
            ).images[0]
            
            # Save test image
            output_path = f"test_custom_model_{i+1}.png"
            image.save(output_path)
            logger.info(f"💾 Saved: {output_path}")
        
        logger.info("✅ Custom model test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def check_training_logs():
    """Check training logs for success indicators"""
    logger.info("📊 Checking training logs...")
    
    log_dir = Path("training_logs")
    if not log_dir.exists():
        logger.warning("⚠️ No training logs found")
        return False
    
    # Find latest log
    log_files = list(log_dir.glob("training_info_*.json"))
    if not log_files:
        logger.warning("⚠️ No training log files found")
        return False
    
    latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
    
    try:
        with open(latest_log, 'r') as f:
            log_data = json.load(f)
        
        logger.info(f"📋 Latest training log: {latest_log.name}")
        logger.info(f"   Step: {log_data.get('step', 'N/A')}")
        logger.info(f"   Loss: {log_data.get('loss', 'N/A')}")
        logger.info(f"   Device: {log_data.get('device', 'N/A')}")
        logger.info(f"   LoRA Success: {log_data.get('lora_success', 'N/A')}")
        
        return log_data.get('lora_success', False)
        
    except Exception as e:
        logger.error(f"❌ Failed to read training log: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting custom model tests...")
    
    # Check training logs
    training_success = check_training_logs()
    
    if not training_success:
        logger.warning("⚠️ Training may not have completed successfully")
    
    # Test custom model
    model_success = test_custom_model()
    
    if model_success:
        logger.info("🎉 All tests passed! Your custom model is working correctly.")
        logger.info("🚀 You can now:")
        logger.info("   1. Use the model in your Flask API")
        logger.info("   2. Generate HD images with generate_hd.py")
        logger.info("   3. Deploy to production")
    else:
        logger.error("❌ Tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 