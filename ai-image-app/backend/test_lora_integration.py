#!/usr/bin/env python3
"""
Test LoRA Integration
Tests the integrated LoRA model in the generate_fast module
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate_fast import generate_image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_lora_generation():
    """Test image generation with LoRA"""
    logger.info("🧪 Testing LoRA integration...")
    
    test_prompts = [
        "A beautiful sunset over mountains, high quality, detailed",
        "A cute robot playing with a cat, digital art",
        "A futuristic city skyline at night, neon lights"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        try:
            logger.info(f"🎨 Generating image {i}/3: {prompt}")
            image_path = generate_image(prompt)
            logger.info(f"✅ Generated: {image_path}")
        except Exception as e:
            logger.error(f"❌ Failed to generate image {i}: {e}")
            return False
    
    logger.info("🎉 All LoRA tests passed!")
    return True

if __name__ == "__main__":
    test_lora_generation() 