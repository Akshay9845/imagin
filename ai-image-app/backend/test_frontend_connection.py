#!/usr/bin/env python3
"""
Test frontend-backend connection
"""

import requests
import json

def test_connection():
    """Test the connection between frontend and backend"""
    print("🔍 Testing Frontend-Backend Connection")
    print("=" * 40)
    
    # Test 1: Backend health
    print("1. Testing backend health...")
    try:
        response = requests.get("http://localhost:5001/health")
        if response.status_code == 200:
            print("✅ Backend is healthy")
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        return False
    
    # Test 2: Frontend accessibility
    print("2. Testing frontend accessibility...")
    try:
        response = requests.get("http://localhost:3000")
        if response.status_code == 200:
            print("✅ Frontend is accessible")
        else:
            print(f"❌ Frontend accessibility failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to frontend: {e}")
        return False
    
    # Test 3: CORS test
    print("3. Testing CORS...")
    try:
        response = requests.post(
            "http://localhost:5001/generate_fast",
            headers={
                "Content-Type": "application/json",
                "Origin": "http://localhost:3000"
            },
            json={"prompt": "test connection"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if "image_path" in data:
                print("✅ CORS is working, image generation successful")
                print(f"   Generated: {data['image_path']}")
            else:
                print("❌ No image_path in response")
                return False
        else:
            print(f"❌ CORS test failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ CORS test error: {e}")
        return False
    
    print("\n🎉 All tests passed! Your system is ready to use.")
    print("\n📝 Next steps:")
    print("   1. Open http://localhost:3000 in your browser")
    print("   2. Enter a prompt like 'forest'")
    print("   3. Click 'Generate' to create images")
    
    return True

if __name__ == "__main__":
    test_connection() 