#!/usr/bin/env python3
"""
Quick Fix for LAION-2B Training
Fixes the 'text' key error in dataset processing
"""

import os
import torch
from datasets import load_dataset
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dataset_structure():
    """Test the LAION-2B dataset structure"""
    logger.info("🔍 Testing LAION-2B dataset structure...")
    
    try:
        # Load dataset
        dataset = load_dataset("laion/laion2B-en", split="train", streaming=True)
        
        # Get first few samples
        for i, sample in enumerate(dataset):
            if i >= 5:  # Test first 5 samples
                break
                
            logger.info(f"Sample {i} keys: {list(sample.keys())}")
            
            # Test different ways to access text
            text_variants = [
                sample.get("text"),
                sample.get("TEXT"),
                sample.get("caption"),
                sample.get("CAPTION"),
                sample.get("title"),
                sample.get("TITLE")
            ]
            
            logger.info(f"Text variants: {text_variants}")
            
            # Test URL access
            url_variants = [
                sample.get("url"),
                sample.get("URL"),
                sample.get("image_url"),
                sample.get("IMAGE_URL")
            ]
            
            logger.info(f"URL variants: {url_variants}")
            logger.info("-" * 50)
            
    except Exception as e:
        logger.error(f"❌ Dataset test failed: {e}")

def create_fixed_training_script():
    """Create a fixed training script with proper key handling"""
    fixed_code = '''
def process_sample(sample):
    """Process a LAION-2B sample with proper key handling"""
    try:
        # Normalize keys to lowercase
        sample_lower = {k.lower(): v for k, v in sample.items()}
        
        # Try multiple possible text keys
        text = (sample_lower.get("text") or 
                sample_lower.get("caption") or 
                sample_lower.get("title") or 
                "")
        
        # Try multiple possible URL keys
        url = (sample_lower.get("url") or 
               sample_lower.get("image_url") or 
               sample_lower.get("image") or 
               "")
        
        # Validate
        if not text or not url:
            return None
            
        if len(text) < 5 or len(text) > 200:
            return None
            
        return {"text": text, "url": url}
        
    except Exception as e:
        logger.warning(f"Failed to process sample: {e}")
        return None

# Use this in your training loop:
for sample in dataset:
    processed = process_sample(sample)
    if processed is None:
        continue
    
    # Now you can safely use processed["text"] and processed["url"]
    text = processed["text"]
    url = processed["url"]
'''
    
    with open("fixed_sample_processing.py", "w") as f:
        f.write(fixed_code)
    
    logger.info("✅ Fixed sample processing code saved to: fixed_sample_processing.py")

def main():
    """Main function"""
    logger.info("🔧 Quick Fix for LAION-2B Training")
    
    # Test dataset structure
    test_dataset_structure()
    
    # Create fixed training code
    create_fixed_training_script()
    
    logger.info("✅ Quick fix complete!")
    logger.info("📝 Use the fixed_sample_processing.py code in your training loop")

if __name__ == "__main__":
    main() 