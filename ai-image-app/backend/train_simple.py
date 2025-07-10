#!/usr/bin/env python3
"""
Simple training script to test LAION-2B-en streaming
"""

import os
import torch
from datasets import load_dataset
from diffusers import StableDiffusionPipeline
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_laion_streaming():
    """Test LAION-2B-en streaming without complex training"""
    print("🧪 Testing LAION-2B-en Streaming")
    print("=" * 40)
    
    try:
        # Load a small sample from LAION-2B-en
        print("📥 Loading LAION-2B-en dataset...")
        dataset = load_dataset(
            "laion/laion2B-en",
            split="train",
            streaming=True
        ).shuffle(seed=42).take(10)  # Just 10 samples for testing
        
        print("✅ Dataset loaded successfully!")
        
        # Test iterating through the dataset
        count = 0
        for item in dataset:
            count += 1
            if 'TEXT' in item and item['TEXT']:
                print(f"📝 Sample {count}: {item['TEXT'][:100]}...")
            else:
                print(f"📝 Sample {count}: [No text]")
            
            if count >= 5:  # Show first 5 samples
                break
        
        print(f"\n✅ Successfully processed {count} samples from LAION-2B-en")
        print("🎉 LAION streaming is working!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_model_loading():
    """Test if we can load a smaller model for training"""
    print("\n🤖 Testing Model Loading")
    print("=" * 40)
    
    try:
        print("📥 Loading Stable Diffusion v1.5...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        print("✅ Model loaded successfully!")
        print(f"📊 Device: {pipe.device}")
        print(f"📊 Model size: {sum(p.numel() for p in pipe.parameters()) / 1e6:.1f}M parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Simple Training Test")
    print("=" * 50)
    
    # Test LAION streaming
    laion_ok = test_laion_streaming()
    
    # Test model loading
    model_ok = test_model_loading()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   LAION Streaming: {'✅' if laion_ok else '❌'}")
    print(f"   Model Loading:   {'✅' if model_ok else '❌'}")
    
    if laion_ok and model_ok:
        print("\n🎉 All tests passed! Ready for training.")
        print("\nNext steps:")
        print("1. Run: python train_lora_stream.py")
        print("2. Or use the web interface at: http://localhost:3000")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main() 