#!/usr/bin/env python3
"""
Final Working Video Generation System
Completely bypasses accelerate import issues
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

class FinalVideoSystem:
    def __init__(self, output_dir="final_video_outputs"):
        """Initialize the final working video generation system"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize pipelines
        self.video_pipelines = {}
        self.image_pipeline = None
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """Load working video generation models with accelerate bypass"""
        logger.info("Loading final working video generation models...")
        
        try:
            # Import diffusers with accelerate bypass
            try:
                # Monkey patch accelerate to avoid import issues
                import accelerate.utils.memory
                if not hasattr(accelerate.utils.memory, 'clear_device_cache'):
                    def clear_device_cache():
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    accelerate.utils.memory.clear_device_cache = clear_device_cache
                
                from diffusers import TextToVideoSDPipeline, AutoPipelineForText2Image
                from diffusers.utils import export_to_video
                logger.info("✅ Diffusers library imported successfully with accelerate bypass")
            except ImportError as e:
                logger.error(f"Diffusers not available: {e}")
                return
            
            # Try to load a working text-to-video model
            try:
                logger.info("Loading working text-to-video model...")
                
                # Use a model that's known to work
                model_id = "damo-vilab/text-to-video-ms-1.7b"
                
                # Load with minimal parameters to avoid issues
                pipeline = TextToVideoSDPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    local_files_only=False  # Allow downloading
                )
                
                # Move to device
                pipeline = pipeline.to(self.device)
                
                self.video_pipelines["working_t2v"] = pipeline
                logger.info("✅ Working T2V model loaded successfully")
                
            except Exception as e:
                logger.warning(f"Could not load working T2V model: {e}")
            
            # Try to load image generation model
            try:
                logger.info("Loading image generation model...")
                
                self.image_pipeline = AutoPipelineForText2Image.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    local_files_only=False
                ).to(self.device)
                
                logger.info("✅ Image generation model loaded successfully")
                
            except Exception as e:
                logger.warning(f"Could not load image generation model: {e}")
            
            # Try alternative models if first one failed
            if not self.video_pipelines:
                alternative_models = [
                    "cerspense/zeroscope_v2_XL",
                    "damo-vilab/text-to-video-synthesis"
                ]
                
                for model_id in alternative_models:
                    try:
                        logger.info(f"Trying alternative model: {model_id}")
                        
                        pipeline = TextToVideoSDPipeline.from_pretrained(
                            model_id,
                            torch_dtype=torch.float32,
                            low_cpu_mem_usage=True,
                            local_files_only=False
                        ).to(self.device)
                        
                        model_name = model_id.split('/')[-1]
                        self.video_pipelines[model_name] = pipeline
                        logger.info(f"✅ Alternative model {model_name} loaded successfully")
                        break
                        
                    except Exception as e:
                        logger.warning(f"Could not load {model_id}: {e}")
            
            logger.info(f"✅ Loaded {len(self.video_pipelines)} video generation models")
            
        except Exception as e:
            logger.error(f"Error during model loading: {e}")
    
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
        
        if not self.video_pipelines:
            raise ValueError("No video generation models available")
        
        # Choose best available pipeline
        if pipeline_type == "auto":
            pipeline_type = list(self.video_pipelines.keys())[0]  # Use first available
        
        if pipeline_type not in self.video_pipelines:
            raise ValueError(f"Pipeline {pipeline_type} not available")
        
        # Enhance prompt for better video quality
        enhanced_prompt = self._enhance_video_prompt(prompt)
        
        # Calculate frames
        num_frames = duration_seconds * fps
        
        # Limit frames for memory constraints
        if num_frames > 25:
            num_frames = 25
            logger.info(f"Limited frames to {num_frames} for memory constraints")
        
        # Generate video
        logger.info(f"Generating {num_frames} frames with {pipeline_type}...")
        
        try:
            from diffusers.utils import export_to_video
            
            video_frames = self.video_pipelines[pipeline_type](
                prompt=enhanced_prompt,
                num_frames=num_frames,
                fps=fps,
                height=height,
                width=width,
                num_inference_steps=25,  # Reduced for speed
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
    
    def generate_image_from_text(self, prompt: str, width: int = 512, height: int = 512) -> str:
        """Generate image from text using available image model"""
        
        if not self.image_pipeline:
            raise ValueError("No image generation model available")
        
        logger.info(f"🎨 Generating image from text: {prompt[:50]}...")
        
        try:
            image = self.image_pipeline(
                prompt=prompt,
                height=height,
                width=width,
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
        """Enhance video prompt with quality improvements"""
        base_enhancements = "high quality video, detailed, smooth motion, professional cinematography, ultra-realistic"
        return f"{prompt}, {base_enhancements}"
    
    def get_available_pipelines(self) -> Dict[str, str]:
        """Get list of available video generation pipelines"""
        pipelines = {
            "auto": "Auto-select best available model"
        }
        
        for name in self.video_pipelines.keys():
            pipelines[name] = f"{name} (Text-to-Video)"
        
        return pipelines
    
    def get_pipeline_status(self) -> Dict[str, bool]:
        """Get status of loaded pipelines"""
        status = {}
        
        for name in self.video_pipelines.keys():
            status[name] = True
        
        status["image_generation"] = self.image_pipeline is not None
        
        return status
    
    def get_system_info(self) -> Dict:
        """Get comprehensive system information"""
        return {
            "device": self.device,
            "available_pipelines": self.get_pipeline_status(),
            "pipeline_descriptions": self.get_available_pipelines(),
            "output_directory": str(self.output_dir),
            "models_loaded": len(self.video_pipelines),
            "image_model_loaded": self.image_pipeline is not None
        }

def main():
    """Test the final working video generation system"""
    print("🎬 Final Working Video Generation System Test")
    print("=" * 50)
    
    # Initialize system
    video_system = FinalVideoSystem()
    
    # Show system info
    info = video_system.get_system_info()
    print(f"Device: {info['device']}")
    print(f"Models loaded: {info['models_loaded']}")
    print(f"Image model loaded: {info['image_model_loaded']}")
    print("Available pipelines:")
    for pipeline, status in info['available_pipelines'].items():
        print(f"  {pipeline}: {'✅' if status else '❌'}")
    
    # Test image generation first
    if info['image_model_loaded']:
        print("\n🎨 Testing image generation...")
        try:
            image_path = video_system.generate_image_from_text(
                prompt="A beautiful sunset over mountains, high quality, detailed",
                width=512,
                height=512
            )
            print(f"✅ Image generated: {image_path}")
        except Exception as e:
            print(f"❌ Image generation failed: {e}")
    
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
        print("\n🔧 Troubleshooting:")
        print("1. Check if diffusers library is installed: pip install diffusers")
        print("2. Check internet connection for model downloads")
        print("3. Try with different model configurations")
        print("4. The accelerate library version might be incompatible")

if __name__ == "__main__":
    main() 