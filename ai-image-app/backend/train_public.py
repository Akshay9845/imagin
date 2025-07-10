#!/usr/bin/env python3
"""
Training script using public datasets
"""

import os
import torch
from datasets import load_dataset
from diffusers import StableDiffusionPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_public_dataset():
    """Test with a public dataset"""
    print("🧪 Testing Public Dataset")
    print("=" * 40)
    
    try:
        # Use a public dataset instead of LAION
        print("📥 Loading public dataset...")
        dataset = load_dataset(
            "poloclub/diffusiondb", 
            "2m_random_1k",  # Smaller subset
            split="train"
        ).select(range(10))  # Just 10 samples
        
        print("✅ Dataset loaded successfully!")
        
        # Show some samples
        for i, item in enumerate(dataset):
            if i >= 3:  # Show first 3
                break
            print(f"📝 Sample {i+1}: {item.get('prompt', 'No prompt')[:100]}...")
        
        print(f"\n✅ Successfully loaded {len(dataset)} samples")
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
        test_path = "static/test_generation.png"
        image.save(test_path)
        print(f"✅ Test image saved: {test_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Public Dataset Training Test")
    print("=" * 50)
    
    # Test public dataset
    dataset_ok = test_public_dataset()
    
    # Test model training
    model_ok = test_model_training()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Public Dataset: {'✅' if dataset_ok else '❌'}")
    print(f"   Model Training: {'✅' if model_ok else '❌'}")
    
    if dataset_ok and model_ok:
        print("\n🎉 Ready for training!")
        print("\nNext steps:")
        print("1. Use web interface: http://localhost:3000")
        print("2. Generate images with current model")
        print("3. For LAION training: huggingface-cli login")
    else:
        print("\n⚠️  Some tests failed.")

if __name__ == "__main__":
    main() 