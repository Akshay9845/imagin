#!/usr/bin/env python3
"""
Script to generate images using our custom trained LoRA model
"""

from diffusers import StableDiffusionPipeline, DDIMScheduler
from peft import PeftModel
import torch
import os

def load_model_with_lora():
    """Load the base model with our custom LoRA weights"""
    
    print("🔄 Loading base model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load our custom LoRA weights
    lora_path = "lora_weights/lora_step_50_20250707_231755"
    if os.path.exists(lora_path):
        print(f"🔄 Loading custom LoRA weights from {lora_path}")
        try:
            # Load LoRA weights into the UNet
            pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
            print("✅ Custom LoRA weights loaded successfully!")
            return pipe, True
        except Exception as e:
            print(f"❌ Failed to load LoRA weights: {e}")
            print("⚠️  Falling back to base model")
            return pipe, False
    else:
        print("⚠️  No custom LoRA weights found")
        return pipe, False

def generate_image_with_lora(prompt, pipe, use_lora=True):
    """Generate image using the model (with or without LoRA)"""
    
    try:
        print(f"🎨 Generating image with {'custom LoRA' if use_lora else 'base model'}...")
        print(f"📝 Prompt: {prompt}")
        
        # Use fewer inference steps for speed
        image = pipe(
            prompt,
            num_inference_steps=20,
            guidance_scale=7.5
        ).images[0]
        
        # Create filename with model indicator
        model_suffix = "_lora" if use_lora else "_base"
        filename = f"{prompt.replace(' ', '_')[:25]}{model_suffix}.png"
        output_path = f"static/{filename}"
        
        # Ensure static directory exists
        os.makedirs("static", exist_ok=True)
        
        # Save the image
        image.save(output_path)
        print(f"✅ Image saved: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error generating image: {e}")
        raise e

def main():
    """Main function to test LoRA generation"""
    
    # Load model
    pipe, lora_loaded = load_model_with_lora()
    
    # Test prompts
    test_prompts = [
        "a beautiful sunset over mountains",
        "a magical unicorn in a rainbow forest",
        "a futuristic city skyline at night"
    ]
    
    print(f"\n🧪 Testing image generation with {'LoRA' if lora_loaded else 'base'} model...")
    
    for prompt in test_prompts:
        try:
            image_path = generate_image_with_lora(prompt, pipe, lora_loaded)
            print(f"✅ Generated: {image_path}")
        except Exception as e:
            print(f"❌ Failed to generate for '{prompt}': {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    main() 