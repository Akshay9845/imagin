#!/usr/bin/env python3
"""
Ultra-Realistic Image & Video Generation System
Combines the best open-source models for maximum realism
"""

import os
import torch
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from PIL import Image
import cv2
from diffusers import (
    StableDiffusionXLPipeline, 
    StableDiffusionPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler
)
from diffusers.utils import export_to_video
import requests
from io import BytesIO
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UltraRealisticSystem:
    def __init__(self, output_dir="ultra_realistic_outputs"):
        """Initialize the ultra-realistic generation system"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Model configurations
        self.models = {
            "realvisxl": "SG161222/RealVisXL_V4.0",
            "dreamshaper": "Lykon/dreamshaper-xl-1-0",
            "juggernaut": "RunDiffusion/Juggernaut-XL-v9",
            "sdxl_base": "stabilityai/stable-diffusion-xl-base-1.0",
            "anything_v5": "andite/anything-v4.0"
        }
        
        # Initialize pipelines
        self.pipelines = {}
        self.load_models()
        
    def load_models(self):
        """Load all required models"""
        logger.info("Loading ultra-realistic models...")
        
        # Load RealVisXL (best for photorealism)
        try:
            logger.info("Loading RealVisXL V4.0...")
            self.pipelines["realvisxl"] = StableDiffusionXLPipeline.from_pretrained(
                self.models["realvisxl"],
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                variant="fp16" if self.device == "cuda" else None,
                use_safetensors=True
            ).to(self.device)
            logger.info("✅ RealVisXL loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load RealVisXL: {e}")
        
        # Load SDXL Base (foundation)
        try:
            logger.info("Loading SDXL Base...")
            self.pipelines["sdxl_base"] = StableDiffusionXLPipeline.from_pretrained(
                self.models["sdxl_base"],
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                variant="fp16" if self.device == "cuda" else None,
                use_safetensors=True
            ).to(self.device)
            logger.info("✅ SDXL Base loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load SDXL Base: {e}")
        
        # Load DreamShaper XL (artistic realism)
        try:
            logger.info("Loading DreamShaper XL...")
            self.pipelines["dreamshaper"] = StableDiffusionXLPipeline.from_pretrained(
                self.models["dreamshaper"],
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                variant="fp16" if self.device == "cuda" else None,
                use_safetensors=True
            ).to(self.device)
            logger.info("✅ DreamShaper XL loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load DreamShaper XL: {e}")
    
    def generate_ultra_realistic_image(
        self, 
        prompt: str, 
        style: str = "photorealistic",
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Image.Image:
        """Generate ultra-realistic image using the best model combination"""
        
        logger.info(f"Generating ultra-realistic image: {prompt[:50]}...")
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed) if self.device == "cuda" else None
        
        # Choose model based on style
        if style == "photorealistic" and "realvisxl" in self.pipelines:
            pipeline = self.pipelines["realvisxl"]
            model_name = "RealVisXL"
        elif style == "artistic" and "dreamshaper" in self.pipelines:
            pipeline = self.pipelines["dreamshaper"]
            model_name = "DreamShaper"
        else:
            pipeline = self.pipelines.get("sdxl_base", list(self.pipelines.values())[0])
            model_name = "SDXL Base"
        
        # Enhanced prompts for realism
        enhanced_prompt = self._enhance_prompt(prompt, style)
        enhanced_negative = self._get_negative_prompt(style) + " " + negative_prompt
        
        logger.info(f"Using {model_name} for generation")
        
        # Generate image
        image = pipeline(
            prompt=enhanced_prompt,
            negative_prompt=enhanced_negative,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device=self.device).manual_seed(seed) if seed else None
        ).images[0]
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ultra_realistic_{style}_{timestamp}.png"
        filepath = self.output_dir / filename
        image.save(filepath)
        
        logger.info(f"✅ Ultra-realistic image saved: {filepath}")
        return image
    
    def _enhance_prompt(self, prompt: str, style: str) -> str:
        """Enhance prompt with style-specific additions"""
        base_enhancements = {
            "photorealistic": "professional photography, high resolution, detailed, sharp focus, 8k uhd, dslr, high quality",
            "artistic": "cinematic lighting, artistic composition, rich detail, professional photography, high quality",
            "portrait": "professional portrait photography, sharp focus, natural lighting, high resolution, detailed skin texture",
            "landscape": "professional landscape photography, golden hour lighting, high resolution, detailed, sharp focus",
            "anime": "anime style, high quality, detailed, vibrant colors, sharp focus"
        }
        
        enhancement = base_enhancements.get(style, base_enhancements["photorealistic"])
        return f"{prompt}, {enhancement}"
    
    def _get_negative_prompt(self, style: str) -> str:
        """Get style-specific negative prompt"""
        base_negative = "blurry, low quality, distorted, deformed, ugly, bad anatomy, watermark, signature, text"
        
        style_negatives = {
            "photorealistic": "cartoon, anime, painting, drawing, illustration, 3d render",
            "artistic": "photographic, realistic, plain, boring, simple",
            "portrait": "cartoon, anime, painting, multiple people, group photo",
            "landscape": "portrait, close-up, people, animals, buildings",
            "anime": "photorealistic, realistic, 3d render, photograph"
        }
        
        style_negative = style_negatives.get(style, "")
        return f"{base_negative}, {style_negative}".strip(", ")
    
    def generate_video_from_image(
        self,
        image: Image.Image,
        motion_prompt: str = "gentle movement",
        num_frames: int = 16,
        fps: int = 8,
        output_filename: Optional[str] = None
    ) -> str:
        """Generate video from image using AnimateDiff-like approach"""
        
        logger.info(f"Generating video with motion: {motion_prompt}")
        
        # For now, we'll create a simple frame interpolation
        # In a full implementation, you'd use AnimateDiff here
        frames = self._create_motion_frames(image, motion_prompt, num_frames)
        
        # Save as video
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"ultra_realistic_video_{timestamp}.mp4"
        
        video_path = self.output_dir / output_filename
        self._save_frames_as_video(frames, video_path, fps)
        
        logger.info(f"✅ Video saved: {video_path}")
        return str(video_path)
    
    def _create_motion_frames(self, image: Image.Image, motion_prompt: str, num_frames: int) -> List[Image.Image]:
        """Create motion frames from base image"""
        frames = []
        
        # Convert to numpy for processing
        img_array = np.array(image)
        
        # Simple motion simulation (in real implementation, use AnimateDiff)
        for i in range(num_frames):
            # Create slight variations for motion effect
            frame = img_array.copy()
            
            # Add subtle motion blur effect
            if "gentle" in motion_prompt.lower():
                # Gentle swaying motion
                offset = int(2 * np.sin(i * 0.5))
                frame = np.roll(frame, offset, axis=1)
            elif "zoom" in motion_prompt.lower():
                # Zoom effect
                scale = 1.0 + 0.1 * np.sin(i * 0.3)
                h, w = frame.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                frame = cv2.resize(frame, (new_w, new_h))
                # Crop to original size
                y_start = (new_h - h) // 2
                x_start = (new_w - w) // 2
                frame = frame[y_start:y_start+h, x_start:x_start+w]
            
            # Convert back to PIL
            frame_img = Image.fromarray(frame)
            frames.append(frame_img)
        
        return frames
    
    def _save_frames_as_video(self, frames: List[Image.Image], output_path: Path, fps: int):
        """Save frames as MP4 video"""
        if not frames:
            return
        
        # Get dimensions from first frame
        width, height = frames[0].size
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Write frames
        for frame in frames:
            # Convert PIL to OpenCV format
            frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            video_writer.write(frame_cv)
        
        video_writer.release()
    
    def generate_ultra_realistic_video(
        self,
        prompt: str,
        style: str = "photorealistic",
        duration_seconds: int = 8,
        fps: int = 8,
        width: int = 1024,
        height: int = 576
    ) -> str:
        """Generate ultra-realistic video from text prompt"""
        
        logger.info(f"Generating ultra-realistic video: {prompt[:50]}...")
        
        # Generate base image first
        base_image = self.generate_ultra_realistic_image(
            prompt=prompt,
            style=style,
            width=width,
            height=height
        )
        
        # Generate video from the image
        num_frames = duration_seconds * fps
        video_path = self.generate_video_from_image(
            image=base_image,
            motion_prompt="gentle movement",
            num_frames=num_frames,
            fps=fps
        )
        
        return video_path
    
    def batch_generate(
        self,
        prompts: List[str],
        style: str = "photorealistic",
        output_format: str = "image"  # "image" or "video"
    ) -> List[str]:
        """Generate multiple ultra-realistic outputs"""
        
        logger.info(f"Batch generating {len(prompts)} {output_format}s...")
        
        outputs = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            try:
                if output_format == "image":
                    image = self.generate_ultra_realistic_image(prompt, style)
                    outputs.append(str(self.output_dir / f"batch_{i+1}.png"))
                else:  # video
                    video_path = self.generate_ultra_realistic_video(prompt, style)
                    outputs.append(video_path)
                    
            except Exception as e:
                logger.error(f"Failed to generate {output_format} for prompt {i+1}: {e}")
                outputs.append(None)
        
        logger.info(f"✅ Batch generation complete: {len([o for o in outputs if o])} successful")
        return outputs

def main():
    """Main function to demonstrate the ultra-realistic system"""
    
    # Initialize the system
    system = UltraRealisticSystem()
    
    # Test prompts for different styles
    test_prompts = [
        "A professional portrait of a confident business person in a modern office",
        "A breathtaking sunset over snow-capped mountains with golden hour lighting",
        "A futuristic city skyline at night with neon lights and flying cars",
        "A serene lake reflecting the sky with crystal clear water",
        "A majestic dragon flying over a medieval castle at dawn"
    ]
    
    # Generate ultra-realistic images
    logger.info("🎨 Generating ultra-realistic images...")
    for i, prompt in enumerate(test_prompts):
        style = "photorealistic" if i < 3 else "artistic"
        system.generate_ultra_realistic_image(
            prompt=prompt,
            style=style,
            seed=42 + i  # Different seed for each image
        )
    
    # Generate a video
    logger.info("🎬 Generating ultra-realistic video...")
    video_path = system.generate_ultra_realistic_video(
        prompt="A beautiful sunset over mountains with gentle camera movement",
        style="photorealistic",
        duration_seconds=6
    )
    
    logger.info("🎉 Ultra-realistic generation complete!")
    logger.info(f"📁 Outputs saved in: {system.output_dir}")

if __name__ == "__main__":
    main() 