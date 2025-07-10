#!/usr/bin/env python3
"""
Test Stable Video Diffusion (SVD) for realistic video generation
Similar to Veo and GPT Sora quality
"""

import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
import os

def test_svd_video():
    print("🎬 Testing Stable Video Diffusion (SVD)...")
    print("This model generates realistic videos similar to Veo/Sora")
    
    # Load the SVD pipeline
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    
    print(f"✅ SVD model loaded on {device}")
    
    # Load a sample image (or create one)
    # You can replace this with any image path
    image_path = "static/test_generation.png"
    
    if os.path.exists(image_path):
        image = load_image(image_path)
    else:
        # Create a simple test image if none exists
        from PIL import Image
        import numpy as np
        
        # Create a simple gradient image
        img_array = np.zeros((512, 512, 3), dtype=np.uint8)
        for i in range(512):
            for j in range(512):
                img_array[i, j] = [i//2, j//2, 128]
        
        image = Image.fromarray(img_array)
        image.save("static/test_input.png")
        image_path = "static/test_input.png"
    
    print(f"📸 Using input image: {image_path}")
    
    # Generate video
    print("🎥 Generating video...")
    video_frames = pipe(
        image,
        num_frames=25,  # 25 frames = 1 second at 25fps
        fps=25,
        motion_bucket_id=127,
        noise_aug_strength=0.4
    ).frames[0]
    
    # Save video
    output_path = "static/svd_video.mp4"
    export_to_video(video_frames, output_path, fps=25)
    
    print(f"✅ Video saved: {output_path}")
    print("🎬 SVD video generation complete!")
    print("📊 This model provides Veo/Sora-like quality for video generation")

if __name__ == "__main__":
    test_svd_video() 