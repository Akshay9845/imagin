#!/usr/bin/env python3
"""
Training script using LAION-400M (publicly available)
"""

import os
import torch
from datasets import load_dataset
from diffusers import StableDiffusionPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_laion400m():
    """Test with LAION-400M dataset"""
    print("🧪 Testing LAION-400M Dataset")
    print("=" * 40)
    
    try:
        # Use LAION-400M which is publicly available
        print("📥 Loading LAION-400M dataset...")
        dataset = load_dataset(
            "laion/laion400m", 
            split="train",
            streaming=True
        ).shuffle(seed=42).take(10)  # Just 10 samples
        
        print("✅ Dataset loaded successfully!")
        
        # Show some samples
        count = 0
        for item in dataset:
            count += 1
            if 'TEXT' in item and item['TEXT']:
                print(f"📝 Sample {count}: {item['TEXT'][:100]}...")
            else:
                print(f"📝 Sample {count}: [No text]")
            
            if count >= 5:  # Show first 5
                break
        
        print(f"\n✅ Successfully processed {count} samples from LAION-400M")
        print("🎉 LAION-400M streaming is working!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_model_training():
    """Test basic model functionality"""
    print("\n🤖 Testing Model Training Setup")
    print("=" * 40)
    
    try:
        print("📥 Loading model...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        print("✅ Model loaded!")
        print(f"📊 Device: {pipe.device}")
        
        # Test basic generation
        print("🎨 Testing basic generation...")
        prompt = "a simple red circle"
        image = pipe(prompt, num_inference_steps=5).images[0]  # Fast test
        
        # Save test image
        os.makedirs("static", exist_ok=True)
        test_path = "static/test_laion400m.png"
        image.save(test_path)
        print(f"✅ Test image saved: {test_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function"""
    print("🚀 LAION-400M Training Test")
    print("=" * 50)
    
    # Test LAION-400M
    dataset_ok = test_laion400m()
    
    # Test model training
    model_ok = test_model_training()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   LAION-400M: {'✅' if dataset_ok else '❌'}")
    print(f"   Model Training: {'✅' if model_ok else '❌'}")
    
    if dataset_ok and model_ok:
        print("\n🎉 Ready for LAION-400M training!")
        print("\nNext steps:")
        print("1. Run: python train_lora_stream.py (with LAION-400M)")
        print("2. Or use web interface: http://localhost:3000")
        print("3. For LAION-2B-en: Request access at https://huggingface.co/datasets/laion/laion2B-en")
    else:
        print("\n⚠️  Some tests failed.")

if __name__ == "__main__":
    main() 