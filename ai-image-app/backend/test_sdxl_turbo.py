#!/usr/bin/env python3
"""
Test SDXL Turbo for ultra-realistic image generation
Best quality for realistic images
"""

import torch
from diffusers import AutoPipelineForText2Image
import os

def test_sdxl_turbo():
    print("🚀 Testing SDXL Turbo for ultra-realistic images...")
    print("This model provides the best quality for realistic image generation")
    
    # Load SDXL Turbo pipeline
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    
    print(f"✅ SDXL Turbo loaded on {device}")
    
    # Test prompts for realistic images
    test_prompts = [
        "A photorealistic portrait of a professional woman in a modern office, natural lighting, high resolution, detailed",
        "A hyperrealistic landscape of a mountain lake at sunset, golden hour, crystal clear water, 8K quality",
        "A cinematic shot of a futuristic city at night, neon lights, rain, cyberpunk aesthetic, ultra detailed"
    ]
    
    # Create output directory
    output_dir = "static/sdxl_turbo_images"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🎨 Generating {len(test_prompts)} ultra-realistic images...")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"  Generating image {i}/{len(test_prompts)}: {prompt[:50]}...")
        
        # Generate image with SDXL Turbo (very fast, 1-4 steps)
        image = pipe(
            prompt=prompt,
            num_inference_steps=4,  # SDXL Turbo is very fast
            guidance_scale=0.0,     # No guidance for turbo
            height=1024,
            width=1024
        ).images[0]
        
        # Save image
        filename = f"{output_dir}/sdxl_turbo_{i:02d}.png"
        image.save(filename)
        
        print(f"    ✅ Saved: {filename}")
    
    print(f"\n🎉 SDXL Turbo testing complete!")
    print(f"📁 Images saved in: {output_dir}/")
    print("📊 SDXL Turbo provides the best quality for realistic image generation")

if __name__ == "__main__":
    test_sdxl_turbo() 