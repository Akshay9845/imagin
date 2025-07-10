#!/usr/bin/env python3
"""
Simple LoRA Integration
Directly integrates the existing LoRA model without changing task type
"""

import os
import torch
import json
from pathlib import Path
from diffusers import StableDiffusionPipeline
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def integrate_lora_directly():
    """Integrate LoRA directly into the Flask app"""
    logger.info("🔗 Integrating LoRA directly into Flask app...")
    
    # Read current app.py
    with open("app.py", 'r') as f:
        app_content = f.read()
    
    # Check if LoRA loading is already present
    if "load_lora_weights" in app_content:
        logger.info("✅ LoRA loading already present in app.py")
        return True
    
    # Add LoRA loading code
    lora_path = "lora_weights/lora_step_50_20250707_231755"
    
    lora_code = f'''
    # Load custom LoRA weights
    try:
        logger.info("🔗 Loading custom LoRA weights...")
        pipeline.load_lora_weights("{lora_path}")
        logger.info("✅ Custom LoRA loaded successfully!")
    except Exception as e:
        logger.warning(f"⚠️ Failed to load LoRA: {{e}}")
        logger.info("📝 Using base model only")
'''
    
    # Find the pipeline creation line and add LoRA loading after it
    if "pipeline = StableDiffusionPipeline.from_pretrained(" in app_content:
        # Insert LoRA code after pipeline creation
        app_content = app_content.replace(
            "pipeline = StableDiffusionPipeline.from_pretrained(",
            "pipeline = StableDiffusionPipeline.from_pretrained(" + lora_code
        )
        
        # Write updated app.py
        with open("app.py", 'w') as f:
            f.write(app_content)
        
        logger.info("✅ app.py updated with LoRA integration")
        return True
    else:
        logger.error("❌ Could not find pipeline creation in app.py")
        return False

def test_lora_loading():
    """Test if LoRA can be loaded"""
    logger.info("🧪 Testing LoRA loading...")
    
    # Setup device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    try:
        # Load base pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Try to load LoRA
        lora_path = "lora_weights/lora_step_50_20250707_231755"
        pipeline.load_lora_weights(lora_path)
        
        logger.info("✅ LoRA loaded successfully!")
        
        # Test generation
        pipeline = pipeline.to(device)
        
        test_prompt = "A beautiful sunset over mountains, high quality, detailed"
        logger.info(f"🎨 Testing generation: {test_prompt}")
        
        image = pipeline(
            prompt=test_prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            height=512,
            width=512
        ).images[0]
        
        # Save test image
        output_path = "lora_test_image.png"
        image.save(output_path)
        logger.info(f"💾 Test image saved: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ LoRA test failed: {e}")
        return False

def create_lora_info():
    """Create info file about the LoRA"""
    lora_path = "lora_weights/lora_step_50_20250707_231755"
    
    info = {
        "lora_path": lora_path,
        "integration_status": "integrated",
        "integration_date": datetime.now().isoformat(),
        "task_type": "FEATURE_EXTRACTION",
        "note": "Using original task type - direct loading",
        "size_mb": round(Path(lora_path).stat().st_size / (1024 * 1024), 2)
    }
    
    with open("lora_integration_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    logger.info("📝 LoRA info saved")

def main():
    """Main integration function"""
    logger.info("🚀 Starting Simple LoRA Integration")
    
    # Check if LoRA exists
    lora_path = "lora_weights/lora_step_50_20250707_231755"
    if not Path(lora_path).exists():
        logger.error(f"❌ LoRA not found at: {lora_path}")
        return
    
    logger.info(f"📁 Found LoRA at: {lora_path}")
    
    # Test LoRA loading first
    if test_lora_loading():
        # Integrate into app
        if integrate_lora_directly():
            create_lora_info()
            logger.info("🎉 LoRA integration successful!")
            logger.info("🔄 Restart your Flask app to use the custom LoRA")
        else:
            logger.error("❌ LoRA integration failed")
    else:
        logger.error("❌ LoRA loading test failed")

if __name__ == "__main__":
    main() 