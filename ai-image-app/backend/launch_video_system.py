#!/usr/bin/env python3
"""
Ultra-Realistic Video Generation System Launcher
Comprehensive launcher for the video generation system with all top 5 models
"""

import sys
import argparse
import subprocess
import time
import os
from pathlib import Path
from ultra_realistic_video_system import UltraRealisticVideoSystem

def print_banner():
    """Print the system banner"""
    print("🎬 Ultra-Realistic Video Generation System")
    print("=" * 60)
    print("🚀 Combines the top 5 open-source video models:")
    print("   1️⃣ AnimateDiff (v2 + Motion LoRA)")
    print("   2️⃣ VideoCrafter2 (Tencent ARC)")
    print("   3️⃣ ModelScope T2V (DAMO Academy)")
    print("   4️⃣ Zeroscope v2 XL")
    print("   5️⃣ RIFE (Real-Time Frame Interpolation)")
    print("")
    print("🔁 Pipeline Options:")
    print("   • Option A: Image → Motion → Video (Best for control)")
    print("   • Option B: Text to Video Direct (Best for creativity)")
    print("")

def test_video_system():
    """Test the video generation system"""
    print("🧪 Testing Ultra-Realistic Video Generation System")
    print("-" * 50)
    
    try:
        # Initialize system
        print("📥 Initializing video system...")
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
        
        # Test simple video generation
        print("\n🎬 Testing video generation...")
        
        # Create a test image first
        from PIL import Image
        import numpy as np
        
        test_image = Image.fromarray(
            np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        )
        test_path = system.output_dir / "test_image.png"
        test_image.save(test_path)
        
        # Generate video from test image
        video_path = system.generate_video_from_image_pipeline(
            image=str(test_path),
            motion_prompt="gentle zoom",
            duration_seconds=3,
            fps=8,
            width=512,
            height=512,
            pipeline_type="interpolation"  # Use interpolation for testing
        )
        
        if video_path:
            print(f"✅ Test video generated: {video_path}")
        else:
            print("❌ Test video generation failed")
        
        print("\n🎉 Video system test complete!")
        return True
        
    except Exception as e:
        print(f"❌ Video system test failed: {e}")
        return False

def generate_single_video():
    """Generate a single video with user input"""
    print("🎬 Single Video Generation")
    print("-" * 30)
    
    try:
        system = UltraRealisticVideoSystem()
        
        # Get user input
        prompt = input("Enter video prompt: ")
        style = input("Enter style (photorealistic/artistic/cinematic) [photorealistic]: ") or "photorealistic"
        duration = input("Enter duration in seconds [8]: ") or "8"
        pipeline = input("Enter pipeline type (auto/stable_video/modelscope_t2v/interpolation) [auto]: ") or "auto"
        
        print(f"\n🎬 Generating video: {prompt[:50]}...")
        
        video_path = system.generate_ultra_realistic_video_pipeline(
            prompt=prompt,
            style=style,
            duration_seconds=int(duration),
            fps=24,
            pipeline_type=pipeline
        )
        
        if video_path:
            print(f"✅ Video generated successfully: {video_path}")
        else:
            print("❌ Video generation failed")
            
    except Exception as e:
        print(f"❌ Error generating video: {e}")

def generate_batch_videos():
    """Generate multiple videos in batch"""
    print("📦 Batch Video Generation")
    print("-" * 30)
    
    try:
        system = UltraRealisticVideoSystem()
        
        # Get prompts from user
        print("Enter video prompts (one per line, press Enter twice when done):")
        prompts = []
        while True:
            prompt = input("> ")
            if not prompt:
                break
            prompts.append(prompt)
        
        if not prompts:
            print("❌ No prompts provided")
            return
        
        style = input("Enter style (photorealistic/artistic/cinematic) [photorealistic]: ") or "photorealistic"
        duration = input("Enter duration in seconds [8]: ") or "8"
        
        print(f"\n📦 Generating {len(prompts)} videos...")
        
        video_paths = system.batch_generate_videos(
            prompts=prompts,
            style=style,
            duration_seconds=int(duration),
            fps=24
        )
        
        successful = len([p for p in video_paths if p])
        print(f"✅ Batch generation complete: {successful}/{len(prompts)} successful")
        
        for i, path in enumerate(video_paths):
            if path:
                print(f"   {i+1}. {path}")
            else:
                print(f"   {i+1}. Failed")
                
    except Exception as e:
        print(f"❌ Error in batch generation: {e}")

def start_api_server():
    """Start the video generation API server"""
    print("🚀 Starting Video Generation API Server")
    print("-" * 40)
    
    try:
        # Check if API file exists
        api_file = Path("ultra_realistic_video_api.py")
        if not api_file.exists():
            print("❌ API file not found: ultra_realistic_video_api.py")
            return
        
        print("🌐 API server will be available at: http://localhost:5003")
        print("📁 Videos will be saved in: ultra_realistic_video_outputs/")
        print("")
        print("🎯 Available endpoints:")
        print("   POST /api/ultra-realistic-video/generate-pipeline")
        print("   POST /api/ultra-realistic-video/generate-from-image")
        print("   POST /api/ultra-realistic-video/generate-direct")
        print("   GET  /api/ultra-realistic-video/status")
        print("   GET  /api/ultra-realistic-video/pipelines")
        print("")
        print("Press Ctrl+C to stop the server")
        print("")
        
        # Start the API server
        subprocess.run([sys.executable, "ultra_realistic_video_api.py"])
        
    except KeyboardInterrupt:
        print("\n🛑 API server stopped")
    except Exception as e:
        print(f"❌ Error starting API server: {e}")

def start_web_interface():
    """Start the video generation web interface"""
    print("🌐 Starting Video Generation Web Interface")
    print("-" * 45)
    
    try:
        # Check if web interface file exists
        web_file = Path("video_web_interface.py")
        if not web_file.exists():
            print("❌ Web interface file not found: video_web_interface.py")
            return
        
        print("🌐 Web interface will be available at: http://localhost:5004")
        print("📁 Videos will be saved in: ultra_realistic_video_outputs/")
        print("")
        print("🎯 Features:")
        print("   • Complete video generation pipeline")
        print("   • Image-to-video conversion")
        print("   • Direct text-to-video generation")
        print("   • System status monitoring")
        print("")
        print("Press Ctrl+C to stop the web interface")
        print("")
        
        # Start the web interface
        subprocess.run([sys.executable, "video_web_interface.py"])
        
    except KeyboardInterrupt:
        print("\n🛑 Web interface stopped")
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")

def show_system_info():
    """Show detailed system information"""
    print("📊 Video System Information")
    print("-" * 30)
    
    try:
        system = UltraRealisticVideoSystem()
        
        print(f"🖥️  Device: {system.device}")
        print(f"📁 Output Directory: {system.output_dir}")
        print(f"📦 Loaded Models: {len(system.video_pipelines)}")
        
        print("\n🎬 Available Pipelines:")
        pipelines = system.get_available_pipelines()
        status = system.get_pipeline_status()
        
        for pipeline, description in pipelines.items():
            available = status.get(pipeline, False)
            status_icon = "✅" if available else "❌"
            print(f"   {status_icon} {pipeline}: {description}")
        
        print("\n🔧 Model Details:")
        for model_name, model_path in system.video_models.items():
            loaded = model_name in system.video_pipelines
            status_icon = "✅" if loaded else "❌"
            print(f"   {status_icon} {model_name}: {model_path}")
        
        # Check output directory
        print(f"\n📁 Output Directory Contents:")
        if system.output_dir.exists():
            video_files = list(system.output_dir.glob("*.mp4"))
            print(f"   Videos: {len(video_files)}")
            for video in video_files[:5]:  # Show first 5
                size_mb = video.stat().st_size / (1024 * 1024)
                print(f"   • {video.name} ({size_mb:.1f} MB)")
            if len(video_files) > 5:
                print(f"   ... and {len(video_files) - 5} more")
        else:
            print("   Directory does not exist")
            
    except Exception as e:
        print(f"❌ Error getting system info: {e}")

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description="Ultra-Realistic Video Generation System")
    parser.add_argument("--test", action="store_true", help="Test the video system")
    parser.add_argument("--generate", action="store_true", help="Generate a single video")
    parser.add_argument("--batch", action="store_true", help="Generate videos in batch")
    parser.add_argument("--api", action="store_true", help="Start API server")
    parser.add_argument("--web", action="store_true", help="Start web interface")
    parser.add_argument("--info", action="store_true", help="Show system information")
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Handle command line arguments
    if args.test:
        test_video_system()
    elif args.generate:
        generate_single_video()
    elif args.batch:
        generate_batch_videos()
    elif args.api:
        start_api_server()
    elif args.web:
        start_web_interface()
    elif args.info:
        show_system_info()
    else:
        # Interactive mode
        while True:
            print("🎯 Choose an option:")
            print("1. 🧪 Test Video System")
            print("2. 🎬 Generate Single Video")
            print("3. 📦 Generate Batch Videos")
            print("4. 🚀 Start API Server")
            print("5. 🌐 Start Web Interface")
            print("6. 🔄 Run Dual Pipeline Example")
            print("7. 📊 Show System Info")
            print("8. 🚪 Exit")
            print("")
            
            choice = input("Enter your choice (1-8): ").strip()
            
            if choice == "1":
                test_video_system()
            elif choice == "2":
                generate_single_video()
            elif choice == "3":
                generate_batch_videos()
            elif choice == "4":
                start_api_server()
            elif choice == "5":
                start_web_interface()
            elif choice == "6":
                print("\n🔄 Running Dual Pipeline Example...")
                print("   This demonstrates both Option A and Option B pipelines")
                print("   Option A: Image → Motion → Video (Best for control)")
                print("   Option B: Text to Video Direct (Best for creativity)")
                print("")
                subprocess.run([sys.executable, "dual_pipeline_example.py"])
            elif choice == "7":
                show_system_info()
            elif choice == "8":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please try again.")
            
            print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main() 