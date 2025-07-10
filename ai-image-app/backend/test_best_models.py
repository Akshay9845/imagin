#!/usr/bin/env python3
"""
Comprehensive test of the best pre-trained models for realistic images and videos
Compares SDXL Turbo, Stable Video Diffusion, and other top models
"""

import os
import torch
from diffusers import AutoPipelineForText2Image, StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image
import time

def test_best_models():
    print("🚀 Testing Best Pre-trained Models for Realistic Generation")
    print("=" * 60)
    
    # Test 1: SDXL Turbo (Best for Realistic Images)
    print("\n1️⃣ Testing SDXL Turbo (Ultra-Realistic Images)")
    print("-" * 40)
    
    try:
        pipe_turbo = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe_turbo = pipe_turbo.to(device)
        
        print("✅ SDXL Turbo loaded successfully")
        
        # Generate ultra-realistic image
        prompt = "A photorealistic portrait of a professional woman in a modern office, natural lighting, high resolution, detailed, 8K quality"
        
        start_time = time.time()
        image = pipe_turbo(
            prompt=prompt,
            num_inference_steps=4,
            guidance_scale=0.0,
            height=1024,
            width=1024
        ).images[0]
        
        generation_time = time.time() - start_time
        
        # Save image
        os.makedirs("static/best_models", exist_ok=True)
        image.save("static/best_models/sdxl_turbo_realistic.png")
        
        print(f"✅ SDXL Turbo image generated in {generation_time:.1f}s")
        print("📁 Saved: static/best_models/sdxl_turbo_realistic.png")
        
    except Exception as e:
        print(f"❌ SDXL Turbo failed: {e}")
    
    # Test 2: Stable Video Diffusion (Best for Videos)
    print("\n2️⃣ Testing Stable Video Diffusion (Veo/Sora-like Videos)")
    print("-" * 40)
    
    try:
        pipe_svd = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        
        pipe_svd = pipe_svd.to(device)
        print("✅ Stable Video Diffusion loaded successfully")
        
        # Use the generated image as input for video
        if os.path.exists("static/best_models/sdxl_turbo_realistic.png"):
            input_image = load_image("static/best_models/sdxl_turbo_realistic.png")
        else:
            # Create a simple test image
            img_array = torch.zeros((512, 512, 3), dtype=torch.uint8)
            for i in range(512):
                for j in range(512):
                    img_array[i, j] = torch.tensor([i//2, j//2, 128])
            
            input_image = Image.fromarray(img_array.numpy())
        
        print("🎥 Generating video from image...")
        start_time = time.time()
        
        video_frames = pipe_svd(
            input_image,
            num_frames=25,
            fps=25,
            motion_bucket_id=127,
            noise_aug_strength=0.4
        ).frames[0]
        
        generation_time = time.time() - start_time
        
        # Save video
        export_to_video(video_frames, "static/best_models/svd_video.mp4", fps=25)
        
        print(f"✅ SVD video generated in {generation_time:.1f}s")
        print("📁 Saved: static/best_models/svd_video.mp4")
        
    except Exception as e:
        print(f"❌ Stable Video Diffusion failed: {e}")
    
    # Test 3: SDXL 1.0 (High Quality Alternative)
    print("\n3️⃣ Testing SDXL 1.0 (High Quality Images)")
    print("-" * 40)
    
    try:
        pipe_sdxl = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        
        pipe_sdxl = pipe_sdxl.to(device)
        print("✅ SDXL 1.0 loaded successfully")
        
        # Generate high-quality image
        prompt = "A hyperrealistic landscape of a mountain lake at sunset, golden hour, crystal clear water, 8K quality, photorealistic"
        
        start_time = time.time()
        image = pipe_sdxl(
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            height=1024,
            width=1024
        ).images[0]
        
        generation_time = time.time() - start_time
        
        # Save image
        image.save("static/best_models/sdxl_landscape.png")
        
        print(f"✅ SDXL 1.0 image generated in {generation_time:.1f}s")
        print("📁 Saved: static/best_models/sdxl_landscape.png")
        
    except Exception as e:
        print(f"❌ SDXL 1.0 failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Best Models Testing Complete!")
    print("\n📊 Model Comparison:")
    print("• SDXL Turbo: Fastest, good quality (4 steps)")
    print("• SDXL 1.0: Best quality, slower (50 steps)")
    print("• Stable Video Diffusion: Veo/Sora-like videos")
    print("\n📁 All outputs saved in: static/best_models/")

if __name__ == "__main__":
    test_best_models() 