#!/usr/bin/env python3
"""
Use Both Pipelines - Simple Example
Shows how to use both Option A and Option B pipelines for different scenarios
"""

from ultra_realistic_video_system import UltraRealisticVideoSystem
from ultra_realistic_system import UltraRealisticSystem
from pathlib import Path

def main():
    """Demonstrate using both pipelines"""
    
    print("🎬 Using Both Video Generation Pipelines")
    print("=" * 50)
    
    # Initialize systems
    video_system = UltraRealisticVideoSystem()
    image_system = UltraRealisticSystem()
    
    # Example 1: Professional Portrait (Use Option A for control)
    print("\n📸 Example 1: Professional Portrait")
    print("   Use Case: Controlled, realistic motion")
    print("   Recommended: Option A (Image → Motion → Video)")
    print("   Reason: Better control over motion and realism")
    
    try:
        # Option A: Generate image first, then animate
        print("   🎨 Step 1: Generating base image with RealVisXL...")
        base_image = image_system.generate_ultra_realistic_image(
            prompt="A professional portrait of a confident business person in a modern office",
            style="photorealistic",
            width=512,
            height=512,
            num_inference_steps=20,
            guidance_scale=7.5
        )
        
        print("   🎬 Step 2: Generating video with controlled motion...")
        video_path = video_system.generate_video_from_image_pipeline(
            image=base_image,
            motion_prompt="gentle zoom",  # Controlled motion
            duration_seconds=5,
            fps=24,
            width=512,
            height=512,
            pipeline_type="stable_video"
        )
        
        print(f"   ✅ Professional portrait video: {Path(video_path).name}")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Example 2: Creative Animation (Use Option B for creativity)
    print("\n🎨 Example 2: Creative Animation")
    print("   Use Case: Dynamic, artistic content")
    print("   Recommended: Option B (Text to Video Direct)")
    print("   Reason: More creative freedom and faster generation")
    
    try:
        # Option B: Direct text-to-video generation
        print("   🎬 Generating creative animation directly...")
        video_path = video_system.generate_direct_text_to_video(
            prompt="A magical forest with glowing butterflies and floating particles",
            duration_seconds=5,
            fps=24,
            width=512,
            height=512,
            pipeline_type="modelscope_t2v"
        )
        
        print(f"   ✅ Creative animation video: {Path(video_path).name}")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Example 3: Hybrid Approach (Use both for maximum quality)
    print("\n🔄 Example 3: Hybrid Approach")
    print("   Use Case: Maximum quality and flexibility")
    print("   Strategy: Generate with both pipelines, choose the best")
    
    prompt = "A majestic dragon flying over a medieval castle at dawn"
    
    try:
        # Generate with Option A
        print("   🎨 Option A: Generating base image...")
        base_image = image_system.generate_ultra_realistic_image(
            prompt=prompt,
            style="photorealistic",
            width=512,
            height=512,
            num_inference_steps=20,
            guidance_scale=7.5
        )
        
        print("   🎬 Option A: Creating video with controlled motion...")
        video_a = video_system.generate_video_from_image_pipeline(
            image=base_image,
            motion_prompt="gentle movement",
            duration_seconds=5,
            fps=24,
            width=512,
            height=512,
            pipeline_type="stable_video"
        )
        
        # Generate with Option B
        print("   🎥 Option B: Generating direct video...")
        video_b = video_system.generate_direct_text_to_video(
            prompt=prompt,
            duration_seconds=5,
            fps=24,
            width=512,
            height=512,
            pipeline_type="modelscope_t2v"
        )
        
        print(f"   ✅ Option A video: {Path(video_a).name}")
        print(f"   ✅ Option B video: {Path(video_b).name}")
        print("   💡 Both videos generated successfully!")
        print("   🎯 Choose based on your preference:")
        print("      • Option A: Better for realistic, controlled motion")
        print("      • Option B: Better for creative, dynamic content")
        
    except Exception as e:
        print(f"   ❌ Hybrid approach failed: {e}")
    
    print("\n🎉 Both pipeline examples complete!")
    print("\n💡 Key Takeaways:")
    print("   • Option A: Best for professional, controlled content")
    print("   • Option B: Best for creative, artistic content")
    print("   • Use both together for maximum flexibility")
    print("   • Choose based on your specific use case and requirements")

if __name__ == "__main__":
    main() 