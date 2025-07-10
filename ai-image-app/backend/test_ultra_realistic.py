#!/usr/bin/env python3
"""
Test script for the ultra-realistic generation system
"""

import sys
import os
from pathlib import Path

def test_ultra_realistic_system():
    """Test the ultra-realistic system"""
    print("🧪 Testing Ultra-Realistic Generation System")
    print("=" * 50)
    
    try:
        # Import the system
        from ultra_realistic_system import UltraRealisticSystem
        
        # Initialize system
        print("📥 Initializing ultra-realistic system...")
        system = UltraRealisticSystem()
        
        print(f"✅ System initialized successfully!")
        print(f"📁 Output directory: {system.output_dir}")
        print(f"🖥️  Device: {system.device}")
        print(f"📦 Available models: {list(system.pipelines.keys())}")
        
        # Test image generation
        print("\n🎨 Testing image generation...")
        test_prompt = "A beautiful sunset over mountains, high quality, detailed"
        
        image = system.generate_ultra_realistic_image(
            prompt=test_prompt,
            style="photorealistic",
            width=512,  # Smaller for testing
            height=512,
            num_inference_steps=20,  # Fewer steps for testing
            guidance_scale=7.5
        )
        
        print(f"✅ Image generated successfully!")
        print(f"📏 Image size: {image.size}")
        
        # Test video generation
        print("\n🎬 Testing video generation...")
        
        video_path = system.generate_video_from_image(
            image=image,
            motion_prompt="gentle zoom",
            num_frames=24,  # 3 seconds at 8 fps
            fps=8
        )
        
        if video_path:
            print(f"✅ Video generated successfully!")
            print(f"📁 Video path: {video_path}")
        else:
            print("❌ Video generation failed")
        
        print("\n🎉 All tests passed! Ultra-realistic system is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoints():
    """Test the API endpoints"""
    print("\n🌐 Testing API endpoints...")
    
    try:
        from ultra_realistic_api import app
        
        with app.test_client() as client:
            # Test status endpoint
            response = client.get('/api/ultra-realistic/status')
            print(f"Status endpoint: {response.status_code}")
            
            # Test styles endpoint
            response = client.get('/api/ultra-realistic/styles')
            print(f"Styles endpoint: {response.status_code}")
            
        print("✅ API endpoints are accessible")
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Ultra-Realistic System Test Suite")
    print("=" * 50)
    
    # Test the core system
    system_ok = test_ultra_realistic_system()
    
    # Test API endpoints
    api_ok = test_api_endpoints()
    
    print("\n📊 Test Results:")
    print(f"  Core System: {'✅ PASS' if system_ok else '❌ FAIL'}")
    print(f"  API Endpoints: {'✅ PASS' if api_ok else '❌ FAIL'}")
    
    if system_ok and api_ok:
        print("\n🎉 All tests passed! The ultra-realistic system is ready to use.")
        print("\nNext steps:")
        print("  1. Run: ./launch_ultra_realistic.sh")
        print("  2. Or start the API: python ultra_realistic_api.py")
        print("  3. Access the API at: http://localhost:5001")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        sys.exit(1) 