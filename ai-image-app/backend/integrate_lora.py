#!/usr/bin/env python3
"""
Integrate Existing LoRA Model
Fixes task type compatibility and integrates the trained LoRA into the generation pipeline
"""

import os
import torch
import json
from pathlib import Path
from diffusers import StableDiffusionPipeline
from peft import PeftModel, LoraConfig
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_lora_config(lora_path: str):
    """Fix the LoRA configuration for SD compatibility"""
    logger.info(f"🔧 Fixing LoRA config at: {lora_path}")
    
    config_path = Path(lora_path) / "adapter_config.json"
    
    # Read current config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Fix task type
    old_task_type = config.get("task_type", "FEATURE_EXTRACTION")
    config["task_type"] = "CAUSAL_LM"
    
    # Save fixed config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"✅ Fixed task_type: {old_task_type} → CAUSAL_LM")
    return config

def create_compatible_lora(lora_path: str, output_path: str):
    """Create a compatible version of the LoRA"""
    logger.info(f"🔄 Creating compatible LoRA from: {lora_path}")
    
    # Setup device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Load base pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Load original LoRA config
    with open(Path(lora_path) / "adapter_config.json", 'r') as f:
        original_config = json.load(f)
    
    # Create new LoRA config with correct task type
    lora_config = LoraConfig(
        r=original_config["r"],
        lora_alpha=original_config["lora_alpha"],
        target_modules=original_config["target_modules"],
        lora_dropout=original_config["lora_dropout"],
        bias=original_config["bias"],
        task_type="CAUSAL_LM"  # Fixed task type
    )
    
    # Apply LoRA to UNet
    pipeline.unet = PeftModel.from_pretrained(
        pipeline.unet,
        lora_path,
        config=lora_config
    )
    
    # Save compatible version
    compatible_path = Path(output_path)
    compatible_path.mkdir(exist_ok=True)
    
    pipeline.unet.save_pretrained(compatible_path)
    
    # Save info
    info = {
        "original_path": lora_path,
        "compatible_path": str(compatible_path),
        "task_type": "CAUSAL_LM",
        "fixed_at": datetime.now().isoformat(),
        "original_config": original_config
    }
    
    with open(compatible_path / "integration_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"✅ Compatible LoRA saved to: {compatible_path}")
    return str(compatible_path)

def integrate_lora_into_app():
    """Integrate LoRA into the main Flask app"""
    logger.info("🔗 Integrating LoRA into Flask app...")
    
    # Paths
    original_lora = "lora_weights/lora_step_50_20250707_231755"
    compatible_lora = "lora_weights/lora_compatible"
    
    if not Path(original_lora).exists():
        logger.error(f"❌ LoRA not found at: {original_lora}")
        return False
    
    # Create compatible version
    compatible_path = create_compatible_lora(original_lora, compatible_lora)
    
    # Update app.py to use the compatible LoRA
    update_app_with_lora(compatible_path)
    
    logger.info("✅ LoRA integration complete!")
    return True

def update_app_with_lora(lora_path: str):
    """Update app.py to include LoRA loading"""
    logger.info(f"📝 Updating app.py with LoRA: {lora_path}")
    
    # Read current app.py
    with open("app.py", 'r') as f:
        app_content = f.read()
    
    # Add LoRA loading code if not present
    if "load_lora_weights" not in app_content:
        # Find where pipeline is loaded
        if "StableDiffusionPipeline.from_pretrained" in app_content:
            # Add LoRA loading after pipeline creation
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
            
            # Insert LoRA code after pipeline creation
            app_content = app_content.replace(
                "pipeline = StableDiffusionPipeline.from_pretrained(",
                "pipeline = StableDiffusionPipeline.from_pretrained(" + lora_code
            )
    
    # Write updated app.py
    with open("app.py", 'w') as f:
        f.write(app_content)
    
    logger.info("✅ app.py updated with LoRA integration")

def test_lora_integration():
    """Test the integrated LoRA model"""
    logger.info("🧪 Testing LoRA integration...")
    
    # Setup device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Load base pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Load compatible LoRA
    compatible_lora = "lora_weights/lora_compatible"
    if Path(compatible_lora).exists():
        try:
            pipeline.load_lora_weights(compatible_lora)
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
    else:
        logger.error(f"❌ Compatible LoRA not found at: {compatible_lora}")
        return False

def main():
    """Main integration function"""
    logger.info("🚀 Starting LoRA Integration")
    
    # Check if LoRA exists
    lora_path = "lora_weights/lora_step_50_20250707_231755"
    if not Path(lora_path).exists():
        logger.error(f"❌ LoRA not found at: {lora_path}")
        return
    
    logger.info(f"📁 Found LoRA at: {lora_path}")
    
    # Integrate LoRA
    if integrate_lora_into_app():
        # Test integration
        if test_lora_integration():
            logger.info("🎉 LoRA integration successful!")
            logger.info("🔄 Restart your Flask app to use the custom LoRA")
        else:
            logger.error("❌ LoRA test failed")
    else:
        logger.error("❌ LoRA integration failed")

if __name__ == "__main__":
    main() 