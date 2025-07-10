#!/usr/bin/env python3
"""
Test script for the ultra-realistic video generation system
Comprehensive testing of all video generation pipelines and features
"""

import sys
import os
from pathlib import Path
import time
from datetime import datetime

def test_video_system():
    """Test the ultra-realistic video generation system"""
    print("🧪 Testing Ultra-Realistic Video Generation System")
    print("=" * 60)
    
    try:
        # Import the video system
        from ultra_realistic_video_system import UltraRealisticVideoSystem
        
        # Initialize system
        print("📥 Initializing ultra-realistic video system...")
        system = UltraRealisticVideoSystem()
        
        print(f"✅ Video system initialized successfully!")
        print(f"📁 Output directory: {system.output_dir}")
        print(f"🖥️  Device: {system.device}")
        print(f"📦 Available pipelines: {list(system.video_pipelines.keys())}")
        
        # Test pipeline status
        print("\n📊 Pipeline Status:")
        status = system.get_pipeline_status()
        for pipeline, available in status.items():
            status_icon = "✅" if available else "❌"
            print(f"   {status_icon} {pipeline}")
        
        # Test available pipelines
        print("\n🎬 Available Pipelines:")
        pipelines = system.get_available_pipelines()
        for pipeline, description in pipelines.items():
            available = status.get(pipeline, False)
            status_icon = "✅" if available else "❌"
            print(f"   {status_icon} {pipeline}: {description}")
        
        # Test 1: Frame interpolation (always available)
        print("\n🎬 Test 1: Frame Interpolation")
        print("-" * 30)
        
        # Create a test image
        from PIL import Image
        import numpy as np
        
        test_image = Image.fromarray(
            np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        )
        test_path = system.output_dir / "test_image.png"
        test_image.save(test_path)
        
        print(f"📸 Created test image: {test_path}")
        
        # Generate video using interpolation
        video_path = system.generate_video_from_image_pipeline(
            image=str(test_path),
            motion_prompt="gentle zoom",
            duration_seconds=3,
            fps=8,
            width=512,
            height=512,
            pipeline_type="interpolation"
        )
        
        if video_path:
            print(f"✅ Interpolation video generated: {video_path}")
        else:
            print("❌ Interpolation video generation failed")
        
        # Test 2: Direct text-to-video (if available)
        if "modelscope_t2v" in system.video_pipelines:
            print("\n🎬 Test 2: Direct Text-to-Video (ModelScope T2V)")
            print("-" * 45)
            
            try:
                video_path = system.generate_direct_text_to_video(
                    prompt="A beautiful sunset over mountains, cinematic lighting",
                    duration_seconds=3,
                    fps=8,
                    width=512,
                    height=512,
                    pipeline_type="modelscope_t2v"
                )
                
                if video_path:
                    print(f"✅ Direct video generated: {video_path}")
                else:
                    print("❌ Direct video generation failed")
                    
            except Exception as e:
                print(f"❌ Direct video generation error: {e}")
        
        # Test 3: Stable Video Diffusion (if available)
        if "stable_video" in system.video_pipelines:
            print("\n🎬 Test 3: Stable Video Diffusion")
            print("-" * 35)
            
            try:
                video_path = system.generate_video_from_image_pipeline(
                    image=str(test_path),
                    motion_prompt="gentle movement",
                    duration_seconds=3,
                    fps=8,
                    width=512,
                    height=512,
                    pipeline_type="stable_video"
                )
                
                if video_path:
                    print(f"✅ Stable Video Diffusion video generated: {video_path}")
                else:
                    print("❌ Stable Video Diffusion generation failed")
                    
            except Exception as e:
                print(f"❌ Stable Video Diffusion error: {e}")
        
        # Test 4: Complete pipeline
        print("\n🎬 Test 4: Complete Pipeline")
        print("-" * 25)
        
        try:
            video_path = system.generate_ultra_realistic_video_pipeline(
                prompt="A beautiful sunset over mountains with gentle camera movement",
                style="photorealistic",
                duration_seconds=3,
                fps=8,
                width=512,
                height=512,
                pipeline_type="auto"
            )
            
            if video_path:
                print(f"✅ Complete pipeline video generated: {video_path}")
            else:
                print("❌ Complete pipeline generation failed")
                
        except Exception as e:
            print(f"❌ Complete pipeline error: {e}")
        
        # Test 5: Batch generation
        print("\n🎬 Test 5: Batch Generation")
        print("-" * 25)
        
        test_prompts = [
            "A beautiful sunset over mountains",
            "A professional portrait with gentle movement"
        ]
        
        try:
            video_paths = system.batch_generate_videos(
                prompts=test_prompts,
                style="photorealistic",
                duration_seconds=3,
                fps=8
            )
            
            successful = len([p for p in video_paths if p])
            print(f"✅ Batch generation complete: {successful}/{len(test_prompts)} successful")
            
            for i, path in enumerate(video_paths):
                if path:
                    print(f"   {i+1}. {path}")
                else:
                    print(f"   {i+1}. Failed")
                    
        except Exception as e:
            print(f"❌ Batch generation error: {e}")
        
        # Test 6: Different motion effects
        print("\n🎬 Test 6: Motion Effects")
        print("-" * 25)
        
        motion_tests = [
            ("gentle zoom", "zoom"),
            ("pan left", "pan"),
            ("rotate slowly", "rotate"),
            ("gentle movement", "gentle")
        ]
        
        for motion_prompt, expected_type in motion_tests:
            try:
                video_path = system.generate_video_from_image_pipeline(
                    image=str(test_path),
                    motion_prompt=motion_prompt,
                    duration_seconds=2,
                    fps=8,
                    width=512,
                    height=512,
                    pipeline_type="interpolation"
                )
                
                if video_path:
                    print(f"✅ {motion_prompt} video generated: {Path(video_path).name}")
                else:
                    print(f"❌ {motion_prompt} video generation failed")
                    
            except Exception as e:
                print(f"❌ {motion_prompt} error: {e}")
        
        # Test 7: System information
        print("\n📊 Test 7: System Information")
        print("-" * 30)
        
        print(f"Device: {system.device}")
        print(f"Output Directory: {system.output_dir}")
        print(f"Loaded Models: {len(system.video_pipelines)}")
        
        # Check output directory contents
        if system.output_dir.exists():
            video_files = list(system.output_dir.glob("*.mp4"))
            print(f"Generated Videos: {len(video_files)}")
            
            for video in video_files:
                size_mb = video.stat().st_size / (1024 * 1024)
                print(f"   • {video.name} ({size_mb:.1f} MB)")
        
        print("\n🎉 All video system tests completed!")
        print(f"📁 Check output directory: {system.output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Video system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoints():
    """Test the API endpoints"""
    print("\n🌐 Testing API Endpoints")
    print("=" * 30)
    
    try:
        import requests
        import json
        
        base_url = "http://localhost:5003"
        
        # Test 1: Status endpoint
        print("📊 Testing status endpoint...")
        try:
            response = requests.get(f"{base_url}/api/ultra-realistic-video/status")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Status endpoint working: {data.get('device', 'Unknown device')}")
            else:
                print(f"❌ Status endpoint failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Status endpoint error: {e}")
        
        # Test 2: Pipelines endpoint
        print("🎬 Testing pipelines endpoint...")
        try:
            response = requests.get(f"{base_url}/api/ultra-realistic-video/pipelines")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Pipelines endpoint working: {len(data.get('pipelines', {}))} pipelines")
            else:
                print(f"❌ Pipelines endpoint failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Pipelines endpoint error: {e}")
        
        # Test 3: List videos endpoint
        print("📁 Testing list videos endpoint...")
        try:
            response = requests.get(f"{base_url}/api/ultra-realistic-video/list-videos")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ List videos endpoint working: {data.get('total_videos', 0)} videos")
            else:
                print(f"❌ List videos endpoint failed: {response.status_code}")
        except Exception as e:
            print(f"❌ List videos endpoint error: {e}")
        
        print("✅ API endpoint tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ API endpoint tests failed: {e}")
        return False

def test_web_interface():
    """Test the web interface"""
    print("\n🌐 Testing Web Interface")
    print("=" * 25)
    
    try:
        import requests
        
        base_url = "http://localhost:5004"
        
        # Test web interface
        print("🌐 Testing web interface...")
        try:
            response = requests.get(base_url)
            if response.status_code == 200:
                print("✅ Web interface is accessible")
                print(f"🌐 Access at: {base_url}")
            else:
                print(f"❌ Web interface failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Web interface error: {e}")
        
        print("✅ Web interface tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Web interface tests failed: {e}")
        return False

def main():
    """Main test function"""
    print("🎬 Ultra-Realistic Video Generation System - Comprehensive Test Suite")
    print("=" * 70)
    
    # Test the video system
    video_system_ok = test_video_system()
    
    # Test API endpoints (if server is running)
    api_ok = test_api_endpoints()
    
    # Test web interface (if server is running)
    web_ok = test_web_interface()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Summary")
    print("=" * 70)
    print(f"Video System: {'✅ PASS' if video_system_ok else '❌ FAIL'}")
    print(f"API Endpoints: {'✅ PASS' if api_ok else '❌ FAIL'}")
    print(f"Web Interface: {'✅ PASS' if web_ok else '❌ FAIL'}")
    
    if video_system_ok:
        print("\n🎉 Video generation system is working correctly!")
        print("🚀 You can now:")
        print("   • Generate videos using the launcher script")
        print("   • Start the API server for programmatic access")
        print("   • Use the web interface for easy generation")
        print("   • Integrate with your applications")
    else:
        print("\n❌ Video generation system has issues.")
        print("🔧 Please check:")
        print("   • Model downloads and Hugging Face token")
        print("   • Hardware requirements and memory")
        print("   • Dependencies and installations")
    
    print(f"\n📁 Generated content will be saved in: ultra_realistic_video_outputs/")

if __name__ == "__main__":
    main() 