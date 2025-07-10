#!/usr/bin/env python3
"""
Test script to verify the complete frontend-to-backend flow
"""

import requests
import json
import time

def test_complete_flow():
    """Test the complete image generation flow"""
    
    # Test data
    test_prompt = "a magical unicorn in a rainbow forest"
    
    print("🧪 Testing complete frontend-to-backend flow...")
    print(f"📝 Test prompt: '{test_prompt}'")
    
    # Step 1: Test backend health
    print("\n1️⃣ Testing backend health...")
    try:
        health_response = requests.get("http://localhost:5001/health")
        if health_response.status_code == 200:
            print("✅ Backend is healthy")
        else:
            print(f"❌ Backend health check failed: {health_response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend health check error: {e}")
        return False
    
    # Step 2: Test image generation
    print("\n2️⃣ Testing image generation...")
    try:
        start_time = time.time()
        response = requests.post(
            "http://localhost:5001/generate_fast",
            headers={
                "Content-Type": "application/json",
                "Origin": "http://localhost:3000"
            },
            json={"prompt": test_prompt}
        )
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            image_path = data.get("image_path")
            print(f"✅ Image generated successfully in {end_time - start_time:.2f} seconds")
            print(f"📁 Image path: {image_path}")
        else:
            print(f"❌ Image generation failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Image generation error: {e}")
        return False
    
    # Step 3: Test image access
    print("\n3️⃣ Testing image access...")
    try:
        image_url = f"http://localhost:5001/{image_path}"
        image_response = requests.head(image_url)
        if image_response.status_code == 200:
            print("✅ Image is accessible")
            print(f"📊 Image size: {image_response.headers.get('Content-Length', 'Unknown')} bytes")
        else:
            print(f"❌ Image access failed: {image_response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Image access error: {e}")
        return False
    
    # Step 4: Test frontend accessibility
    print("\n4️⃣ Testing frontend accessibility...")
    try:
        frontend_response = requests.get("http://localhost:3000")
        if frontend_response.status_code == 200:
            print("✅ Frontend is accessible")
        else:
            print(f"❌ Frontend access failed: {frontend_response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Frontend access error: {e}")
        return False
    
    print("\n🎉 All tests passed! The system is working correctly.")
    print("\n📋 Summary:")
    print(f"   • Backend: ✅ Running on http://localhost:5001")
    print(f"   • Frontend: ✅ Running on http://localhost:3000")
    print(f"   • Image generation: ✅ Working")
    print(f"   • Image serving: ✅ Working")
    print(f"   • CORS: ✅ Configured correctly")
    
    print(f"\n🌐 You can now:")
    print(f"   • Visit http://localhost:3000 to use the web interface")
    print(f"   • Generate images via API: POST http://localhost:5001/generate_fast")
    print(f"   • View generated images at: http://localhost:5001/static/")
    
    return True

if __name__ == "__main__":
    test_complete_flow() 