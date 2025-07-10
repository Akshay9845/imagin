#!/usr/bin/env python3
"""
Dual Pipeline Example - Using Both Option A and Option B
Demonstrates how to use both video generation approaches for maximum flexibility
"""

from ultra_realistic_video_system import UltraRealisticVideoSystem
from ultra_realistic_system import UltraRealisticSystem
import time
from pathlib import Path

def demonstrate_dual_pipelines():
    """Demonstrate both pipeline options"""
    
    print("🎬 Dual Pipeline Video Generation Example")
    print("=" * 50)
    print("This example shows how to use both pipeline approaches:")
    print("• Option A: Image → Motion → Video (Best for control)")
    print("• Option B: Text to Video Direct (Best for creativity)")
    print("")
    
    # Initialize both systems
    video_system = UltraRealisticVideoSystem()
    image_system = UltraRealisticSystem()
    
    # Example prompts
    prompts = [
        "A beautiful sunset over mountains with golden hour lighting",
        "A professional portrait of a confident business person",
        "A futuristic city skyline with neon lights and flying cars"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n🎬 Example {i}: {prompt}")
        print("-" * 40)
        
        # Option A: Image → Motion → Video (Best for control)
        print("🔁 Option A: Image → Motion → Video Pipeline")
        print("   Step 1: Generate base image with RealVisXL")
        print("   Step 2: Animate with AnimateDiff + Motion LoRA")
        print("   Step 3: Upscale with Zeroscope")
        print("   Step 4: Interpolate with RIFE for smooth motion")
        
        try:
            # Step 1: Generate base image with RealVisXL
            print("   📸 Step 1: Generating base image...")
            base_image = image_system.generate_ultra_realistic_image(
                prompt=prompt,
                style="photorealistic",
                width=512,  # Smaller for faster processing
                height=512,
                num_inference_steps=20,  # Faster for demo
                guidance_scale=7.5
            )
            
            # Step 2-4: Generate video from image
            print("   🎬 Steps 2-4: Generating video from image...")
            video_path_a = video_system.generate_video_from_image_pipeline(
                image=base_image,
                motion_prompt="gentle zoom",
                duration_seconds=5,
                fps=24,
                width=512,
                height=512,
                pipeline_type="stable_video"  # Use Stable Video Diffusion
            )
            
            print(f"   ✅ Option A complete: {Path(video_path_a).name}")
            
        except Exception as e:
            print(f"   ❌ Option A failed: {e}")
            video_path_a = None
        
        # Option B: Text to Video Direct (Best for creativity)
        print("\n🎥 Option B: Text to Video Direct Pipeline")
        print("   Step 1: Use VideoCrafter2 for direct generation")
        print("   Step 2: Use ModelScope T2V as lightweight fallback")
        
        try:
            # Try VideoCrafter2 first (if available)
            print("   🎬 Step 1: Trying VideoCrafter2...")
            video_path_b = video_system.generate_direct_text_to_video(
                prompt=prompt,
                duration_seconds=5,
                fps=24,
                width=512,
                height=512,
                pipeline_type="modelscope_t2v"  # Use ModelScope T2V
            )
            
            print(f"   ✅ Option B complete: {Path(video_path_b).name}")
            
        except Exception as e:
            print(f"   ❌ Option B failed: {e}")
            video_path_b = None
        
        # Compare results
        print(f"\n📊 Results for Example {i}:")
        if video_path_a:
            print(f"   Option A (Image→Video): ✅ {Path(video_path_a).name}")
        else:
            print(f"   Option A (Image→Video): ❌ Failed")
            
        if video_path_b:
            print(f"   Option B (Direct): ✅ {Path(video_path_b).name}")
        else:
            print(f"   Option B (Direct): ❌ Failed")
        
        print("")

def demonstrate_hybrid_approach():
    """Demonstrate hybrid approach using both pipelines together"""
    
    print("\n🔄 Hybrid Approach: Combining Both Pipelines")
    print("=" * 50)
    print("This approach uses both pipelines for maximum quality:")
    print("1. Generate base image with RealVisXL")
    print("2. Create video with Option A (Image→Video)")
    print("3. Generate alternative with Option B (Direct)")
    print("4. Choose the best result or combine them")
    print("")
    
    video_system = UltraRealisticVideoSystem()
    image_system = UltraRealisticSystem()
    
    prompt = "A majestic dragon flying over a medieval castle at dawn"
    
    print(f"🎬 Generating: {prompt}")
    print("-" * 40)
    
    results = {}
    
    # Step 1: Generate base image
    print("📸 Step 1: Generating base image with RealVisXL...")
    try:
        base_image = image_system.generate_ultra_realistic_image(
            prompt=prompt,
            style="photorealistic",
            width=512,
            height=512,
            num_inference_steps=20,
            guidance_scale=7.5
        )
        print("   ✅ Base image generated")
        results['base_image'] = base_image
    except Exception as e:
        print(f"   ❌ Base image failed: {e}")
        return
    
    # Step 2: Option A - Image to Video
    print("\n🎬 Step 2: Option A - Image to Video Pipeline...")
    try:
        video_a = video_system.generate_video_from_image_pipeline(
            image=base_image,
            motion_prompt="gentle movement",
            duration_seconds=5,
            fps=24,
            width=512,
            height=512,
            pipeline_type="stable_video"
        )
        print(f"   ✅ Option A video: {Path(video_a).name}")
        results['option_a'] = video_a
    except Exception as e:
        print(f"   ❌ Option A failed: {e}")
    
    # Step 3: Option B - Direct Video
    print("\n🎥 Step 3: Option B - Direct Video Pipeline...")
    try:
        video_b = video_system.generate_direct_text_to_video(
            prompt=prompt,
            duration_seconds=5,
            fps=24,
            width=512,
            height=512,
            pipeline_type="modelscope_t2v"
        )
        print(f"   ✅ Option B video: {Path(video_b).name}")
        results['option_b'] = video_b
    except Exception as e:
        print(f"   ❌ Option B failed: {e}")
    
    # Step 4: Results summary
    print(f"\n📊 Hybrid Generation Results:")
    print(f"   Base Image: ✅ Generated")
    if 'option_a' in results:
        print(f"   Option A (Image→Video): ✅ {Path(results['option_a']).name}")
    else:
        print(f"   Option A (Image→Video): ❌ Failed")
        
    if 'option_b' in results:
        print(f"   Option B (Direct): ✅ {Path(results['option_b']).name}")
    else:
        print(f"   Option B (Direct): ❌ Failed")
    
    print(f"\n🎯 Recommendations:")
    if 'option_a' in results and 'option_b' in results:
        print("   • Both pipelines succeeded!")
        print("   • Option A: Better for controlled, realistic motion")
        print("   • Option B: Better for creative, dynamic content")
        print("   • Use Option A for professional/realistic content")
        print("   • Use Option B for creative/artistic content")
    elif 'option_a' in results:
        print("   • Option A succeeded - use for controlled motion")
    elif 'option_b' in results:
        print("   • Option B succeeded - use for creative content")
    else:
        print("   • Both pipelines failed - check system setup")

def demonstrate_pipeline_selection():
    """Demonstrate how to choose the right pipeline for different use cases"""
    
    print("\n🎯 Pipeline Selection Guide")
    print("=" * 40)
    print("Choose the right pipeline based on your needs:")
    print("")
    
    use_cases = [
        {
            "name": "Professional Portrait",
            "description": "Controlled, realistic motion",
            "recommended": "Option A (Image→Video)",
            "reason": "Better control over motion and realism"
        },
        {
            "name": "Creative Animation",
            "description": "Dynamic, artistic content",
            "recommended": "Option B (Direct)",
            "reason": "More creative freedom and faster generation"
        },
        {
            "name": "Product Demo",
            "description": "Smooth, controlled movement",
            "recommended": "Option A (Image→Video)",
            "reason": "Precise motion control and professional quality"
        },
        {
            "name": "Artistic Scene",
            "description": "Imaginative, stylized content",
            "recommended": "Option B (Direct)",
            "reason": "Better for creative and artistic expressions"
        },
        {
            "name": "Realistic Landscape",
            "description": "Natural, photorealistic motion",
            "recommended": "Option A (Image→Video)",
            "reason": "Superior realism and natural motion"
        }
    ]
    
    for i, use_case in enumerate(use_cases, 1):
        print(f"{i}. {use_case['name']}")
        print(f"   Description: {use_case['description']}")
        print(f"   Recommended: {use_case['recommended']}")
        print(f"   Reason: {use_case['reason']}")
        print("")

def main():
    """Main function to demonstrate dual pipeline usage"""
    
    print("🎬 Ultra-Realistic Video Generation - Dual Pipeline Example")
    print("=" * 60)
    
    # Demonstrate both pipelines
    demonstrate_dual_pipelines()
    
    # Demonstrate hybrid approach
    demonstrate_hybrid_approach()
    
    # Show pipeline selection guide
    demonstrate_pipeline_selection()
    
    print("🎉 Dual pipeline demonstration complete!")
    print("")
    print("💡 Key Takeaways:")
    print("   • Option A: Best for controlled, realistic motion")
    print("   • Option B: Best for creative, dynamic content")
    print("   • Use both together for maximum flexibility")
    print("   • Choose based on your specific use case")
    print("   • Both pipelines can be used in the same project")

if __name__ == "__main__":
    main() 