#!/usr/bin/env python3
"""
Test fast image generation
"""

import requests
import time

def test_fast_generation():
    """Test the fast generation endpoint"""
    print("🚀 Testing Fast Image Generation")
    print("=" * 40)
    
    prompt = "a simple red circle"
    
    print(f"🎨 Generating: {prompt}")
    start_time = time.time()
    
    try:
        response = requests.post(
            "http://localhost:5001/generate-fast",
            json={"prompt": prompt},
            timeout=60  # 1 minute timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            end_time = time.time()
            print(f"✅ Generated in {end_time - start_time:.1f} seconds!")
            print(f"📁 Image: {data['image_url']}")
            print(f"🌐 View at: http://localhost:5001/{data['image_url']}")
            return True
        else:
            print(f"❌ Failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Timeout - model is still loading or too slow")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_fast_generation() 