#!/usr/bin/env python3
"""
Generate sample images for the AI Image Generator
"""

import requests
import json
import time
import os

def generate_sample_image(prompt, filename=None):
    """Generate a single image"""
    if filename is None:
        filename = prompt.replace(' ', '_').replace(',', '').replace('.', '')[:30] + '.png'
    
    print(f"🎨 Generating: {prompt}")
    
    try:
        response = requests.post(
            "http://localhost:5001/generate",
            json={"prompt": prompt},
            timeout=120  # 2 minutes timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Generated: {data['image_url']}")
            return data['image_url']
        else:
            print(f"❌ Failed: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"⏰ Timeout generating: {prompt}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Generate sample images"""
    print("🚀 Generating Sample Images")
    print("=" * 40)
    
    # Sample prompts
    sample_prompts = [
        "a beautiful sunset over mountains",
        "a cute cat sitting on a windowsill",
        "a futuristic city skyline at night",
        "a serene lake with mountains in the background",
        "a magical forest with glowing mushrooms"
    ]
    
    generated_images = []
    
    for i, prompt in enumerate(sample_prompts, 1):
        print(f"\n[{i}/{len(sample_prompts)}] ", end="")
        image_url = generate_sample_image(prompt)
        if image_url:
            generated_images.append((prompt, image_url))
        
        # Wait between generations
        if i < len(sample_prompts):
            print("⏳ Waiting 10 seconds before next generation...")
            time.sleep(10)
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Generation Summary:")
    print(f"✅ Successfully generated: {len(generated_images)}/{len(sample_prompts)} images")
    
    if generated_images:
        print("\n📁 Generated Images:")
        for prompt, image_url in generated_images:
            print(f"   • {prompt}")
            print(f"     → {image_url}")
            print(f"     → http://localhost:5001/{image_url}")
    
    # Check static directory
    static_dir = "backend/static"
    if os.path.exists(static_dir):
        files = os.listdir(static_dir)
        if files:
            print(f"\n📂 Files in {static_dir}/:")
            for file in files:
                print(f"   • {file}")
        else:
            print(f"\n📂 {static_dir}/ is empty")

if __name__ == "__main__":
    main() 