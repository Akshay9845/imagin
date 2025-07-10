#!/usr/bin/env python3
"""
Updated Video Generation System with Real Video Capabilities
Uses updated libraries (accelerate 1.8.1, diffusers 0.34.0, transformers 4.53.1)
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
import warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UpdatedVideoSystem:
    def __init__(self, output_dir="updated_video_outputs"):
        """Initialize the updated video generation system with real video capabilities"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize pipelines
        self.video_pipelines = {}
        self.image_pipeline = None
        
        # Load models with updated libraries
        self.load_models()
    
    def load_models(self):
        """Load real video generation models with updated libraries"""
        logger.info("Loading updated video generation models...")
        
        try:
            # Import updated libraries with correct class names
            from diffusers import (
                DiffusionPipeline, 
                StableDiffusionPipeline,
                StableVideoDiffusionPipeline
            )
            from diffusers.utils import export_to_video
            logger.info("✅ Updated diffusers library imported successfully")
            
            # Load real text-to-video models
            self._load_text_to_video_models()
            
            # Load image-to-video models
            self._load_image_to_video_models()
            
            # Load image generation model
            self._load_image_generation_model()
            
            logger.info(f"✅ Loaded {len(self.video_pipelines)} video generation models")
            
        except Exception as e:
            logger.error(f"Error during model loading: {e}")
    
    def _load_text_to_video_models(self):
        """Load text-to-video models for real video generation"""
        text_to_video_models = [
            {
                "name": "damo_t2v",
                "model_id": "damo-vilab/text-to-video-ms-1.7b",
                "description": "Damo T2V - High quality text-to-video"
            },
            {
                "name": "zeroscope_xl",
                "model_id": "cerspense/zeroscope_v2_XL",
                "description": "Zeroscope v2 XL - HD text-to-video"
            },
            {
                "name": "modelscope_t2v",
                "model_id": "damo-vilab/text-to-video-synthesis",
                "description": "ModelScope T2V - Fast text-to-video"
            }
        ]
        
        for model_info in text_to_video_models:
            try:
                logger.info(f"Loading {model_info['name']}: {model_info['model_id']}")
                
                pipeline = DiffusionPipeline.from_pretrained(
                    model_info["model_id"],
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    local_files_only=False
                ).to(self.device)
                
                self.video_pipelines[model_info["name"]] = {
                    "pipeline": pipeline,
                    "description": model_info["description"],
                    "type": "text_to_video"
                }
                
                logger.info(f"✅ {model_info['name']} loaded successfully")
                
            except Exception as e:
                logger.warning(f"Could not load {model_info['name']}: {e}")
    
    def _load_image_to_video_models(self):
        """Load image-to-video models for animating static images"""
        image_to_video_models = [
            {
                "name": "stable_video",
                "model_id": "stabilityai/stable-video-diffusion-img2vid-xt",
                "description": "Stable Video Diffusion - Image to video"
            }
        ]
        
        for model_info in image_to_video_models:
            try:
                logger.info(f"Loading {model_info['name']}: {model_info['model_id']}")
                
                pipeline = StableVideoDiffusionPipeline.from_pretrained(
                    model_info["model_id"],
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    local_files_only=False
                ).to(self.device)
                
                self.video_pipelines[model_info["name"]] = {
                    "pipeline": pipeline,
                    "description": model_info["description"],
                    "type": "image_to_video"
                }
                
                logger.info(f"✅ {model_info['name']} loaded successfully")
                
            except Exception as e:
                logger.warning(f"Could not load {model_info['name']}: {e}")
    
    def _load_image_generation_model(self):
        """Load image generation model for creating base images"""
        try:
            logger.info("Loading image generation model...")
            
            self.image_pipeline = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                local_files_only=False
            ).to(self.device)
            
            logger.info("✅ Image generation model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load image generation model: {e}")
    
    def generate_real_video_from_text(
        self,
        prompt: str,
        duration_seconds: int = 8,
        fps: int = 24,
        width: int = 512,
        height: int = 512,
        pipeline_name: str = "auto"
    ) -> str:
        """
        Generate real video from text prompt (like Veo/Sora)
        Creates actual video content, not just motion effects
        """
        
        logger.info(f"🎬 Generating real video from text: {prompt[:50]}...")
        
        # Find available text-to-video pipelines
        t2v_pipelines = {k: v for k, v in self.video_pipelines.items() 
                        if v["type"] == "text_to_video"}
        
        if not t2v_pipelines:
            raise ValueError("No text-to-video models available")
        
        # Choose pipeline
        if pipeline_name == "auto":
            pipeline_name = list(t2v_pipelines.keys())[0]
        
        if pipeline_name not in t2v_pipelines:
            raise ValueError(f"Pipeline {pipeline_name} not available")
        
        pipeline_info = t2v_pipelines[pipeline_name]
        pipeline = pipeline_info["pipeline"]
        
        # Enhance prompt for better video quality
        enhanced_prompt = self._enhance_video_prompt(prompt)
        
        # Calculate frames
        num_frames = duration_seconds * fps
        
        # Limit frames for memory constraints
        if num_frames > 25:
            num_frames = 25
            logger.info(f"Limited frames to {num_frames} for memory constraints")
        
        # Generate video
        logger.info(f"Generating {num_frames} frames with {pipeline_name}...")
        
        try:
            from diffusers.utils import export_to_video
            
            video_frames = pipeline(
                prompt=enhanced_prompt,
                num_frames=num_frames,
                fps=fps,
                height=height,
                width=width,
                num_inference_steps=25,
                guidance_scale=7.5
            ).frames[0]
            
            # Save video
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"real_video_{pipeline_name}_{timestamp}.mp4"
            video_path = self.output_dir / filename
            
            export_to_video(video_frames, str(video_path), fps=fps)
            
            logger.info(f"✅ Real video saved: {video_path}")
            return str(video_path)
            
        except Exception as e:
            logger.error(f"Error generating real video: {e}")
            raise
    
    def generate_person_dancing_video(
        self,
        dance_style: str = "modern dance",
        duration_seconds: int = 8,
        fps: int = 24,
        width: int = 512,
        height: int = 512
    ) -> str:
        """
        Generate a video of a person dancing
        Uses real video generation, not just motion effects
        """
        
        prompt = f"A person dancing {dance_style}, high quality, smooth motion, professional lighting"
        
        return self.generate_real_video_from_text(
            prompt=prompt,
            duration_seconds=duration_seconds,
            fps=fps,
            width=width,
            height=height,
            pipeline_name="auto"
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
        Generate a video of a scene with motion
        Uses real video generation for dynamic scenes
        """
        
        prompt = f"{scene_description}, {motion_type}, high quality, smooth motion, cinematic"
        
        return self.generate_real_video_from_text(
            prompt=prompt,
            duration_seconds=duration_seconds,
            fps=fps,
            width=width,
            height=height,
            pipeline_name="auto"
        )
    
    def generate_video_from_image(
        self,
        image_path: str,
        motion_prompt: str = "gentle movement",
        duration_seconds: int = 8,
        fps: int = 24,
        pipeline_name: str = "auto"
    ) -> str:
        """
        Generate video from a static image using image-to-video models
        Creates real motion, not just frame interpolation
        """
        
        logger.info(f"🎬 Generating video from image: {image_path}")
        
        # Find available image-to-video pipelines
        i2v_pipelines = {k: v for k, v in self.video_pipelines.items() 
                        if v["type"] == "image_to_video"}
        
        if not i2v_pipelines:
            raise ValueError("No image-to-video models available")
        
        # Choose pipeline
        if pipeline_name == "auto":
            pipeline_name = list(i2v_pipelines.keys())[0]
        
        if pipeline_name not in i2v_pipelines:
            raise ValueError(f"Pipeline {pipeline_name} not available")
        
        pipeline_info = i2v_pipelines[pipeline_name]
        pipeline = pipeline_info["pipeline"]
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((512, 512))  # Standard size for most models
        
        # Calculate frames
        num_frames = duration_seconds * fps
        
        # Limit frames for memory constraints
        if num_frames > 25:
            num_frames = 25
            logger.info(f"Limited frames to {num_frames} for memory constraints")
        
        # Generate video
        logger.info(f"Generating {num_frames} frames with {pipeline_name}...")
        
        try:
            from diffusers.utils import export_to_video
            
            if pipeline_name == "stable_video":
                video_frames = pipeline(
                    image,
                    num_frames=num_frames,
                    fps=fps,
                    motion_bucket_id=127,
                    noise_aug_strength=0.1
                ).frames[0]
            else:
                video_frames = pipeline(
                    image,
                    num_frames=num_frames,
                    fps=fps,
                    num_inference_steps=25,
                    guidance_scale=7.5
                ).frames[0]
            
            # Save video
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"image_video_{pipeline_name}_{timestamp}.mp4"
            video_path = self.output_dir / filename
            
            export_to_video(video_frames, str(video_path), fps=fps)
            
            logger.info(f"✅ Image-to-video saved: {video_path}")
            return str(video_path)
            
        except Exception as e:
            logger.error(f"Error generating image-to-video: {e}")
            raise
    
    def generate_image_from_text(self, prompt: str, width: int = 512, height: int = 512) -> str:
        """Generate an image from text prompt"""
        
        if not self.image_pipeline:
            raise ValueError("No image generation model available")
        
        logger.info(f"🎨 Generating image from text: {prompt[:50]}...")
        
        try:
            image = self.image_pipeline(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=25,
                guidance_scale=7.5
            ).images[0]
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_image_{timestamp}.png"
            image_path = self.output_dir / filename
            
            image.save(image_path)
            
            logger.info(f"✅ Image saved: {image_path}")
            return str(image_path)
            
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            raise
    
    def _enhance_video_prompt(self, prompt: str) -> str:
        """Enhance prompt for better video quality"""
        enhancements = [
            "high quality", "smooth motion", "professional lighting",
            "cinematic", "detailed", "realistic"
        ]
        
        enhanced = prompt
        for enhancement in enhancements:
            if enhancement not in prompt.lower():
                enhanced += f", {enhancement}"
        
        return enhanced
    
    def get_available_pipelines(self) -> Dict[str, Dict]:
        """Get information about available pipelines"""
        return {
            name: {
                "description": info["description"],
                "type": info["type"],
                "available": True
            }
            for name, info in self.video_pipelines.items()
        }
    
    def get_pipeline_status(self) -> Dict[str, bool]:
        """Get status of all pipelines"""
        return {
            name: True for name in self.video_pipelines.keys()
        }
    
    def get_system_info(self) -> Dict:
        """Get system information"""
        return {
            "device": self.device,
            "output_directory": str(self.output_dir),
            "pipelines_loaded": len(self.video_pipelines),
            "image_pipeline_loaded": self.image_pipeline is not None,
            "available_pipelines": list(self.video_pipelines.keys())
        }

def main():
    """Test the updated video generation system"""
    system = UpdatedVideoSystem()
    
    print("🎬 Updated Video Generation System")
    print("=" * 50)
    
    # Show system info
    info = system.get_system_info()
    print(f"Device: {info['device']}")
    print(f"Pipelines loaded: {info['pipelines_loaded']}")
    print(f"Image pipeline: {'✅' if info['image_pipeline_loaded'] else '❌'}")
    
    # Show available pipelines
    pipelines = system.get_available_pipelines()
    print("\nAvailable pipelines:")
    for name, details in pipelines.items():
        print(f"  • {name}: {details['description']} ({details['type']})")
    
    # Test real video generation
    if pipelines:
        print("\n🎬 Testing real video generation...")
        try:
            video_path = system.generate_real_video_from_text(
                prompt="A beautiful sunset over mountains with gentle camera movement",
                duration_seconds=5,
                fps=24
            )
            print(f"✅ Video generated: {video_path}")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 