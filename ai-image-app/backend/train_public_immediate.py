#!/usr/bin/env python3
"""
Training script using completely public datasets
"""

import os
import torch
from datasets import load_dataset
from diffusers import StableDiffusionPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_public_dataset():
    """Test with a completely public dataset"""
    print("🧪 Testing Public Dataset")
    print("=" * 40)
    
    try:
        # Use a completely public dataset
        print("📥 Loading public dataset...")
        dataset = load_dataset(
            "Gustavosta/Stable-Diffusion-Prompts", 
            split="train"
        ).select(range(10))  # Just 10 samples
        
        print("✅ Dataset loaded successfully!")
        
        # Show some samples
        for i, item in enumerate(dataset):
            if i >= 3:  # Show first 3
                break
            prompt = item.get('Prompt', 'No prompt')
            print(f"📝 Sample {i+1}: {prompt[:100]}...")
        
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
        test_path = "static/test_public.png"
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
        print("3. For LAION datasets: Request access at:")
        print("   - https://huggingface.co/datasets/laion/laion2B-en")
        print("   - https://huggingface.co/datasets/laion/laion400m")
    else:
        print("\n⚠️  Some tests failed.")

if __name__ == "__main__":
    main() 