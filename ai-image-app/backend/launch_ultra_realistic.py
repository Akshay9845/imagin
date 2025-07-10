#!/usr/bin/env python3
"""
Ultra-Realistic Generation Launcher
Easy interface for generating ultra-realistic images and videos
"""

import sys
import argparse
from pathlib import Path
from ultra_realistic_system import UltraRealisticSystem

def main():
    parser = argparse.ArgumentParser(description="Ultra-Realistic Image & Video Generation")
    parser.add_argument("--prompt", "-p", required=True, help="Text prompt for generation")
    parser.add_argument("--style", "-s", default="photorealistic", 
                       choices=["photorealistic", "artistic", "portrait", "landscape", "anime"],
                       help="Generation style")
    parser.add_argument("--output", "-o", default="image", choices=["image", "video"],
                       help="Output format")
    parser.add_argument("--width", "-w", type=int, default=1024, help="Image width")
    parser.add_argument("--height", "-h", type=int, default=1024, help="Image height")
    parser.add_argument("--duration", "-d", type=int, default=8, help="Video duration in seconds")
    parser.add_argument("--fps", type=int, default=8, help="Video FPS")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--batch", "-b", action="store_true", help="Batch mode (read prompts from file)")
    parser.add_argument("--file", "-f", help="File containing prompts (one per line)")
    
    args = parser.parse_args()
    
    # Initialize system
    print("🚀 Initializing Ultra-Realistic Generation System...")
    system = UltraRealisticSystem()
    
    if args.batch and args.file:
        # Batch mode
        print(f"📁 Reading prompts from {args.file}")
        with open(args.file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        print(f"🔄 Generating {len(prompts)} {args.output}s...")
        outputs = system.batch_generate(
            prompts=prompts,
            style=args.style,
            output_format=args.output
        )
        
        print(f"✅ Batch generation complete! Generated {len([o for o in outputs if o])} files")
        
    else:
        # Single generation
        print(f"🎨 Generating {args.output}: {args.prompt[:50]}...")
        
        if args.output == "image":
            image = system.generate_ultra_realistic_image(
                prompt=args.prompt,
                style=args.style,
                width=args.width,
                height=args.height,
                seed=args.seed
            )
            print(f"✅ Image generated successfully!")
            
        else:  # video
            video_path = system.generate_ultra_realistic_video(
                prompt=args.prompt,
                style=args.style,
                duration_seconds=args.duration,
                fps=args.fps,
                width=args.width,
                height=args.height
            )
            print(f"✅ Video generated: {video_path}")
    
    print(f"📁 All outputs saved in: {system.output_dir}")

if __name__ == "__main__":
    main() 