#!/usr/bin/env python3
"""
Generate test images using the working image generation system
"""

import requests
import time
import json
from datetime import datetime

def generate_image(prompt, use_fast=True):
    """Generate an image using the API"""
    url = "http://localhost:5001"
    
    if use_fast:
        endpoint = f"{url}/generate_fast"
        print(f"🚀 Generating fast image: '{prompt}'")
    else:
        endpoint = f"{url}/generate"
        print(f"🎨 Generating high-quality image: '{prompt}'")
    
    try:
        response = requests.post(endpoint, json={"prompt": prompt}, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            if "image_path" in result:
                print(f"✅ Image generated successfully!")
                print(f"📁 Saved to: {result['image_path']}")
                return result['image_path']
            else:
                print(f"❌ No image path in response: {result}")
                return None
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out - model might still be loading")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Generate multiple test images"""
    print("🎨 Image Generation Test")
    print("=" * 40)
    
    # Test prompts
    test_prompts = [
        "A beautiful sunset over mountains, digital art",
        "A cute robot playing with a cat, cartoon style",
        "A futuristic city skyline at night, neon lights",
        "A magical forest with glowing mushrooms, fantasy art",
        "A portrait of a wise old wizard, detailed artwork"
    ]
    
    print("🌐 Starting image generation tests...")
    print("📝 Make sure the backend server is running on port 5001")
    print()
    
    generated_images = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}/{len(test_prompts)} ---")
        
        # Try fast generation first
        image_path = generate_image(prompt, use_fast=True)
        
        if image_path:
            generated_images.append({
                "prompt": prompt,
                "path": image_path,
                "type": "fast"
            })
        else:
            print("⚠️  Fast generation failed, trying regular generation...")
            image_path = generate_image(prompt, use_fast=False)
            
            if image_path:
                generated_images.append({
                    "prompt": prompt,
                    "path": image_path,
                    "type": "regular"
                })
        
        # Wait a bit between generations
        if i < len(test_prompts):
            print("⏳ Waiting 3 seconds before next generation...")
            time.sleep(3)
    
    # Summary
    print(f"\n{'='*40}")
    print("📊 Generation Summary:")
    print(f"✅ Successfully generated: {len(generated_images)}/{len(test_prompts)} images")
    
    if generated_images:
        print("\n📁 Generated images:")
        for img in generated_images:
            print(f"  • {img['type'].title()}: {img['path']}")
            print(f"    Prompt: '{img['prompt']}'")
    
    # Save summary to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"generation_summary_{timestamp}.json"
    
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "total_tests": len(test_prompts),
            "successful": len(generated_images),
            "images": generated_images
        }, f, indent=2)
    
    print(f"\n📄 Summary saved to: {summary_file}")

if __name__ == "__main__":
    main() 