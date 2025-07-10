#!/usr/bin/env python3
"""
Real Video Generation System
Generates actual video content from text prompts like Veo/Sora
Uses working text-to-video models for real video generation
"""

import os
import torch
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image
import cv2
import requests
from io import BytesIO
import json
from datetime import datetime
import time
from diffusers import (
    TextToVideoSDPipeline,
    DiffusionPipeline,
    AutoPipelineForText2Image
)
from diffusers.utils import export_to_video
import imageio
import warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealVideoSystem:
    def __init__(self, output_dir="real_video_outputs"):
        """Initialize the real video generation system"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Working text-to-video models
        self.video_models = {
            "damo_t2v": "damo-vilab/text-to-video-ms-1.7b",
            "zeroscope_v2": "cerspense/zeroscope_v2_XL",
            "zeroscope_v2_1": "cerspense/zeroscope_v2_1",
            "text2video_zero": "runwayml/stable-diffusion-v1-5"  # For image generation
        }
        
        # Initialize pipelines
        self.video_pipelines = {}
        self.image_pipeline = None
        self.load_models()
        
    def load_models(self):
        """Load working video generation models"""
        logger.info("Loading real video generation models...")
        
        # Load image generation pipeline for base images
        try:
            logger.info("Loading image generation pipeline...")
            self.image_pipeline = AutoPipelineForText2Image.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                variant="fp16" if self.device == "cuda" else None
            ).to(self.device)
            logger.info("✅ Image generation pipeline loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load image pipeline: {e}")
        
        # Load Damo T2V (most reliable)
        try:
            logger.info("Loading Damo T2V (text-to-video)...")
            self.video_pipelines["damo_t2v"] = TextToVideoSDPipeline.from_pretrained(
                self.video_models["damo_t2v"],
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                variant="fp16" if self.device == "cuda" else None
            ).to(self.device)
            logger.info("✅ Damo T2V loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load Damo T2V: {e}")
        
        # Load Zeroscope v2 (HD quality)
        try:
            logger.info("Loading Zeroscope v2 XL...")
            self.video_pipelines["zeroscope_v2"] = TextToVideoSDPipeline.from_pretrained(
                self.video_models["zeroscope_v2"],
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                variant="fp16" if self.device == "cuda" else None
            ).to(self.device)
            logger.info("✅ Zeroscope v2 XL loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load Zeroscope v2: {e}")
        
        # Load Zeroscope v2.1 (alternative)
        try:
            logger.info("Loading Zeroscope v2.1...")
            self.video_pipelines["zeroscope_v2_1"] = TextToVideoSDPipeline.from_pretrained(
                self.video_models["zeroscope_v2_1"],
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                variant="fp16" if self.device == "cuda" else None
            ).to(self.device)
            logger.info("✅ Zeroscope v2.1 loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load Zeroscope v2.1: {e}")
        
        logger.info(f"✅ Loaded {len(self.video_pipelines)} video generation models")
    
    def generate_real_video_from_text(
        self,
        prompt: str,
        duration_seconds: int = 8,
        fps: int = 24,
        width: int = 512,
        height: int = 512,
        pipeline_type: str = "auto"
    ) -> str:
        """
        Generate real video from text prompt (like Veo/Sora)
        Creates actual video content, not just motion effects
        """
        
        logger.info(f"🎬 Generating real video from text: {prompt[:50]}...")
        
        # Choose best available pipeline
        if pipeline_type == "auto":
            if "damo_t2v" in self.video_pipelines:
                pipeline_type = "damo_t2v"
            elif "zeroscope_v2" in self.video_pipelines:
                pipeline_type = "zeroscope_v2"
            elif "zeroscope_v2_1" in self.video_pipelines:
                pipeline_type = "zeroscope_v2_1"
            else:
                raise ValueError("No video generation models available")
        
        if pipeline_type not in self.video_pipelines:
            raise ValueError(f"Pipeline {pipeline_type} not available")
        
        # Enhance prompt for better video quality
        enhanced_prompt = self._enhance_video_prompt(prompt)
        
        # Calculate frames
        num_frames = duration_seconds * fps
        
        # Limit frames for memory constraints
        if num_frames > 50:
            num_frames = 50
            logger.info(f"Limited frames to {num_frames} for memory constraints")
        
        # Generate video
        logger.info(f"Generating {num_frames} frames with {pipeline_type}...")
        
        try:
            video_frames = self.video_pipelines[pipeline_type](
                prompt=enhanced_prompt,
                num_frames=num_frames,
                fps=fps,
                height=height,
                width=width,
                num_inference_steps=50,
                guidance_scale=7.5
            ).frames[0]
            
            # Save video
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"real_video_{pipeline_type}_{timestamp}.mp4"
            video_path = self.output_dir / filename
            
            export_to_video(video_frames, str(video_path), fps=fps)
            
            logger.info(f"✅ Real video saved: {video_path}")
            return str(video_path)
            
        except Exception as e:
            logger.error(f"Error generating video with {pipeline_type}: {e}")
            raise
    
    def generate_video_from_image(
        self,
        image: Union[Image.Image, str],
        motion_prompt: str = "gentle movement",
        duration_seconds: int = 8,
        fps: int = 24,
        width: int = 512,
        height: int = 512
    ) -> str:
        """
        Generate video from image using text-to-video models
        Creates motion based on the image content
        """
        
        logger.info(f"🎬 Generating video from image with motion: {motion_prompt}")
        
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Resize image to target dimensions
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        
        # Create enhanced prompt based on image content and motion
        enhanced_prompt = self._create_motion_prompt_from_image(image, motion_prompt)
        
        # Generate video using text-to-video pipeline
        return self.generate_real_video_from_text(
            prompt=enhanced_prompt,
            duration_seconds=duration_seconds,
            fps=fps,
            width=width,
            height=height,
            pipeline_type="auto"
        )
    
    def generate_person_dancing_video(
        self,
        dance_style: str = "modern dance",
        duration_seconds: int = 8,
        fps: int = 24,
        width: int = 512,
        height: int = 512
    ) -> str:
        """
        Generate a person dancing video (like Veo examples)
        """
        
        # Create dance-specific prompts
        dance_prompts = {
            "modern dance": "A person performing modern dance moves, fluid motion, dynamic poses, professional dancer, studio lighting",
            "hip hop": "A person doing hip hop dance moves, urban style, rhythmic motion, street dance, energetic movement",
            "ballet": "A ballet dancer performing graceful moves, elegant motion, classical dance, stage lighting, flowing movement",
            "breakdance": "A breakdancer performing acrobatic moves, street dance, dynamic motion, urban style, athletic movement",
            "salsa": "A couple dancing salsa, Latin dance, rhythmic motion, colorful clothing, dance floor, romantic atmosphere"
        }
        
        base_prompt = dance_prompts.get(dance_style.lower(), dance_prompts["modern dance"])
        enhanced_prompt = f"{base_prompt}, high quality video, smooth motion, professional cinematography, ultra-realistic"
        
        return self.generate_real_video_from_text(
            prompt=enhanced_prompt,
            duration_seconds=duration_seconds,
            fps=fps,
            width=width,
            height=height,
            pipeline_type="auto"
        )
    
    def generate_scene_video(
        self,
        scene_description: str,
        motion_type: str = "gentle camera movement",
        duration_seconds: int = 8,
        fps: int = 24,
        width: int = 512,
        height: int = 512
    ) -> str:
        """
        Generate scene-based video with specific motion
        """
        
        motion_descriptions = {
            "gentle camera movement": "gentle camera pan, smooth motion",
            "zoom": "camera zoom, gradual approach",
            "pan": "camera pan, horizontal movement",
            "tilt": "camera tilt, vertical movement",
            "orbit": "camera orbit, circular movement",
            "dolly": "camera dolly, forward/backward movement"
        }
        
        motion_desc = motion_descriptions.get(motion_type, motion_descriptions["gentle camera movement"])
        enhanced_prompt = f"{scene_description}, {motion_desc}, high quality video, cinematic lighting, professional cinematography"
        
        return self.generate_real_video_from_text(
            prompt=enhanced_prompt,
            duration_seconds=duration_seconds,
            fps=fps,
            width=width,
            height=height,
            pipeline_type="auto"
        )
    
    def _enhance_video_prompt(self, prompt: str) -> str:
        """Enhance video prompt with quality improvements"""
        base_enhancements = "high quality video, detailed, smooth motion, professional cinematography, ultra-realistic"
        return f"{prompt}, {base_enhancements}"
    
    def _create_motion_prompt_from_image(self, image: Image.Image, motion_prompt: str) -> str:
        """Create a motion prompt based on image content"""
        # Analyze image content (simplified)
        # In a full implementation, you'd use image analysis here
        
        # Create enhanced prompt
        enhanced_prompt = f"Video of {motion_prompt}, based on the image content, high quality, smooth motion"
        return enhanced_prompt
    
    def batch_generate_videos(
        self,
        prompts: List[str],
        duration_seconds: int = 8,
        fps: int = 24
    ) -> List[str]:
        """Generate multiple videos in batch"""
        
        logger.info(f"🎬 Batch generating {len(prompts)} videos...")
        
        outputs = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing video {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            try:
                video_path = self.generate_real_video_from_text(
                    prompt=prompt,
                    duration_seconds=duration_seconds,
                    fps=fps
                )
                outputs.append(video_path)
                
            except Exception as e:
                logger.error(f"Failed to generate video for prompt {i+1}: {e}")
                outputs.append(None)
        
        logger.info(f"✅ Batch video generation complete: {len([o for o in outputs if o])} successful")
        return outputs
    
    def get_available_pipelines(self) -> Dict[str, str]:
        """Get list of available video generation pipelines"""
        return {
            "damo_t2v": "Damo T2V (Text-to-Video)",
            "zeroscope_v2": "Zeroscope v2 XL (HD Text-to-Video)",
            "zeroscope_v2_1": "Zeroscope v2.1 (Alternative HD)",
            "auto": "Auto-select best available model"
        }
    
    def get_pipeline_status(self) -> Dict[str, bool]:
        """Get status of loaded pipelines"""
        return {
            "damo_t2v": "damo_t2v" in self.video_pipelines,
            "zeroscope_v2": "zeroscope_v2" in self.video_pipelines,
            "zeroscope_v2_1": "zeroscope_v2_1" in self.video_pipelines,
            "image_generation": self.image_pipeline is not None
        }
    
    def get_system_info(self) -> Dict:
        """Get comprehensive system information"""
        return {
            "device": self.device,
            "available_pipelines": self.get_pipeline_status(),
            "pipeline_descriptions": self.get_available_pipelines(),
            "output_directory": str(self.output_dir),
            "models_loaded": len(self.video_pipelines)
        }

def main():
    """Test the real video generation system"""
    print("🎬 Real Video Generation System Test")
    print("=" * 50)
    
    # Initialize system
    video_system = RealVideoSystem()
    
    # Show system info
    info = video_system.get_system_info()
    print(f"Device: {info['device']}")
    print(f"Models loaded: {info['models_loaded']}")
    print("Available pipelines:")
    for pipeline, status in info['available_pipelines'].items():
        print(f"  {pipeline}: {'✅' if status else '❌'}")
    
    # Test video generation
    if info['models_loaded'] > 0:
        print("\n🎬 Testing real video generation...")
        
        # Test 1: Person dancing
        try:
            print("Testing person dancing video...")
            video_path = video_system.generate_person_dancing_video(
                dance_style="modern dance",
                duration_seconds=5,
                fps=24
            )
            print(f"✅ Dancing video generated: {video_path}")
        except Exception as e:
            print(f"❌ Dancing video failed: {e}")
        
        # Test 2: Scene video
        try:
            print("Testing scene video...")
            video_path = video_system.generate_scene_video(
                scene_description="A beautiful sunset over mountains",
                motion_type="gentle camera movement",
                duration_seconds=5,
                fps=24
            )
            print(f"✅ Scene video generated: {video_path}")
        except Exception as e:
            print(f"❌ Scene video failed: {e}")
    
    else:
        print("❌ No video generation models loaded")

if __name__ == "__main__":
    main() 