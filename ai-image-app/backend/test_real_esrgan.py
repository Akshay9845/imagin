#!/usr/bin/env python3
"""
Test Real-ESRGAN for ultra-realistic image upscaling
Best for enhancing image quality to photorealistic levels
"""

import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from PIL import Image
import os
import numpy as np

def test_real_esrgan():
    print("🔍 Testing Real-ESRGAN for ultra-realistic upscaling...")
    print("This model enhances images to photorealistic quality")
    
    # Load Real-ESRGAN model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    # Initialize upscaler
    upsampler = RealESRGANer(
        scale=4,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True
    )
    
    print("✅ Real-ESRGAN model loaded")
    
    # Test with existing images or create a test image
    test_images = []
    
    # Check for existing images
    if os.path.exists("static/test_generation.png"):
        test_images.append("static/test_generation.png")
    
    if os.path.exists("quick_test_images"):
        for file in os.listdir("quick_test_images"):
            if file.endswith(".png"):
                test_images.append(f"quick_test_images/{file}")
                if len(test_images) >= 2:  # Limit to 2 images
                    break
    
    # If no images found, create a test image
    if not test_images:
        print("📸 Creating test image for upscaling...")
        # Create a simple test image
        img_array = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            for j in range(256):
                img_array[i, j] = [i, j, 128]
        
        test_image = Image.fromarray(img_array)
        test_image.save("static/test_for_upscaling.png")
        test_images.append("static/test_for_upscaling.png")
    
    # Create output directory
    output_dir = "static/real_esrgan_upscaled"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔍 Upscaling {len(test_images)} images...")
    
    for i, image_path in enumerate(test_images, 1):
        print(f"  Upscaling image {i}/{len(test_images)}: {image_path}")
        
        # Load image
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        # Upscale with Real-ESRGAN
        output, _ = upsampler.enhance(img_array, outscale=4)
        
        # Save upscaled image
        upscaled_img = Image.fromarray(output)
        filename = f"{output_dir}/upscaled_{i:02d}.png"
        upscaled_img.save(filename)
        
        print(f"    ✅ Saved: {filename}")
        print(f"    📏 Original: {img.size} → Upscaled: {upscaled_img.size}")
    
    print(f"\n🎉 Real-ESRGAN testing complete!")
    print(f"📁 Upscaled images saved in: {output_dir}/")
    print("📊 Real-ESRGAN provides photorealistic image enhancement")

if __name__ == "__main__":
    test_real_esrgan() 