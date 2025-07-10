#!/usr/bin/env python3
"""
Quick test script to verify the AI Image Generator system is working.
"""

import requests
import json
import time

def test_backend():
    """Test the Flask backend API"""
    print("🧪 Testing Backend API...")
    
    # Test if server is running
    try:
        response = requests.get("http://localhost:5001/", timeout=5)
        print(f"✅ Backend server is running (Status: {response.status_code})")
    except requests.exceptions.ConnectionError:
        print("❌ Backend server is not running. Please start it with:")
        print("   cd ai-image-app/backend")
        print("   source venv/bin/activate")
        print("   python run_server.py")
        return False
    
    # Test image generation endpoint
    test_prompt = "a simple red circle on white background"
    print(f"🎨 Testing image generation with prompt: '{test_prompt}'")
    
    try:
        response = requests.post(
            "http://localhost:5001/generate",
            json={"prompt": test_prompt},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Image generated successfully!")
            print(f"   Image path: {data.get('image_url')}")
            return True
        else:
            print(f"❌ Generation failed (Status: {response.status_code})")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def test_frontend():
    """Test if frontend is accessible"""
    print("\n🌐 Testing Frontend...")
    
    # Try both common ports
    for port in [3000, 3001]:
        try:
            response = requests.get(f"http://localhost:{port}/", timeout=5)
            if response.status_code == 200:
                print(f"✅ Frontend is running at http://localhost:{port}")
                return True
        except requests.exceptions.ConnectionError:
            continue
    
    print("❌ Frontend is not running. Please start it with:")
    print("   cd ai-image-app/frontend")
    print("   npm run dev")
    return False

def main():
    """Run all tests"""
    print("🚀 AI Image Generator System Test")
    print("=" * 40)
    
    backend_ok = test_backend()
    frontend_ok = test_frontend()
    
    print("\n" + "=" * 40)
    if backend_ok and frontend_ok:
        print("🎉 All systems are running!")
        print("\n📱 You can now:")
        print("   1. Open http://localhost:3000 in your browser")
        print("   2. Enter a prompt and generate images")
        print("   3. Start LoRA training with: python train_lora_stream.py")
    else:
        print("⚠️  Some systems need attention.")
        print("\n🔧 Troubleshooting:")
        if not backend_ok:
            print("   - Check if Flask server is running on port 5001")
        if not frontend_ok:
            print("   - Check if Next.js server is running on port 3000 or 3001")

if __name__ == "__main__":
    main() 