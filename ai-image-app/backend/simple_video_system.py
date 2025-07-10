#!/usr/bin/env python3
"""
Simple Video Generation System
A reliable fallback system using frame interpolation for both pipeline options
"""

import torch
import logging
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Union
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleVideoSystem:
    def __init__(self, output_dir="ultra_realistic_video_outputs"):
        """Initialize the simple video generation system"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Available motion types
        self.motion_types = ["gentle", "zoom", "pan", "rotate", "wave", "pulse"]
        
    def generate_video_from_image_pipeline(
        self,
        image: Union[Image.Image, str],
        motion_prompt: str = "gentle movement",
        duration_seconds: int = 8,
        fps: int = 24,
        width: int = 1024,
        height: int = 576,
        pipeline_type: str = "interpolation"
    ) -> str:
        """
        Option A: Image → Motion → Video (Best for control)
        Generate video from image using frame interpolation
        """
        
        logger.info(f"🎬 Generating video from image with {pipeline_type} pipeline")
        
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Resize image to target dimensions
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        
        # Generate video using frame interpolation
        return self._generate_with_frame_interpolation(image, motion_prompt, duration_seconds, fps)
    
    def generate_direct_text_to_video(
        self,
        prompt: str,
        duration_seconds: int = 8,
        fps: int = 24,
        width: int = 1024,
        height: int = 576,
        pipeline_type: str = "interpolation"
    ) -> str:
        """
        Option B: Text to Video Direct (Best for creativity)
        Generate video directly from text using frame interpolation
        """
        
        logger.info(f"🎬 Generating direct text-to-video: {prompt[:50]}...")
        
        # Create a base image from the prompt (simplified)
        # In a full implementation, you'd use an image generation model here
        base_image = self._create_base_image_from_prompt(prompt, width, height)
        
        # Generate video using frame interpolation
        return self._generate_with_frame_interpolation(base_image, "dynamic movement", duration_seconds, fps)
    
    def _create_base_image_from_prompt(self, prompt: str, width: int, height: int) -> Image.Image:
        """Create a base image from prompt (simplified version)"""
        
        # Create a gradient background based on prompt keywords
        if "sunset" in prompt.lower() or "golden" in prompt.lower():
            # Create sunset gradient
            img_array = np.zeros((height, width, 3), dtype=np.uint8)
            for y in range(height):
                # Orange to purple gradient
                ratio = y / height
                r = int(255 * (1 - ratio) + 128 * ratio)
                g = int(128 * (1 - ratio) + 64 * ratio)
                b = int(64 * (1 - ratio) + 128 * ratio)
                img_array[y, :] = [r, g, b]
        
        elif "forest" in prompt.lower() or "nature" in prompt.lower():
            # Create forest gradient
            img_array = np.zeros((height, width, 3), dtype=np.uint8)
            for y in range(height):
                # Green gradient
                ratio = y / height
                r = int(34 * (1 - ratio) + 20 * ratio)
                g = int(139 * (1 - ratio) + 80 * ratio)
                b = int(34 * (1 - ratio) + 20 * ratio)
                img_array[y, :] = [r, g, b]
        
        elif "city" in prompt.lower() or "urban" in prompt.lower():
            # Create city gradient
            img_array = np.zeros((height, width, 3), dtype=np.uint8)
            for y in range(height):
                # Blue-gray gradient
                ratio = y / height
                r = int(70 * (1 - ratio) + 40 * ratio)
                g = int(130 * (1 - ratio) + 80 * ratio)
                b = int(180 * (1 - ratio) + 120 * ratio)
                img_array[y, :] = [r, g, b]
        
        else:
            # Default colorful gradient
            img_array = np.zeros((height, width, 3), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    # Create a colorful pattern
                    r = int(128 + 127 * np.sin(x * 0.01) * np.cos(y * 0.01))
                    g = int(128 + 127 * np.sin(x * 0.015) * np.cos(y * 0.015))
                    b = int(128 + 127 * np.sin(x * 0.02) * np.cos(y * 0.02))
                    img_array[y, x] = [r, g, b]
        
        return Image.fromarray(img_array)
    
    def _generate_with_frame_interpolation(
        self,
        image: Image.Image,
        motion_prompt: str,
        duration_seconds: int,
        fps: int
    ) -> str:
        """Generate video using frame interpolation"""
        
        logger.info("Using frame interpolation for video generation")
        
        # Create motion frames
        frames = self._create_motion_frames(image, motion_prompt, duration_seconds * fps)
        
        # Interpolate frames for smoother motion
        interpolated_frames = self._interpolate_frames(frames, fps)
        
        # Save video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"interpolated_video_{timestamp}.mp4"
        video_path = self.output_dir / filename
        
        self._save_frames_as_video(interpolated_frames, video_path, fps)
        
        logger.info(f"✅ Interpolated video saved: {video_path}")
        return str(video_path)
    
    def _create_motion_frames(
        self,
        image: Image.Image,
        motion_prompt: str,
        num_frames: int
    ) -> List[Image.Image]:
        """Create motion frames from base image"""
        
        frames = []
        motion_type = self._parse_motion_prompt(motion_prompt)
        
        for i in range(num_frames):
            progress = i / (num_frames - 1) if num_frames > 1 else 0
            frame = self._apply_motion_effect(image, motion_type, progress)
            frames.append(frame)
        
        return frames
    
    def _parse_motion_prompt(self, motion_prompt: str) -> str:
        """Parse motion prompt to determine motion type"""
        
        motion_prompt = motion_prompt.lower()
        
        if "zoom" in motion_prompt:
            return "zoom"
        elif "pan" in motion_prompt or "move" in motion_prompt:
            return "pan"
        elif "rotate" in motion_prompt or "spin" in motion_prompt:
            return "rotate"
        elif "wave" in motion_prompt or "flow" in motion_prompt:
            return "wave"
        elif "pulse" in motion_prompt or "breathe" in motion_prompt:
            return "pulse"
        else:
            return "gentle"
    
    def _apply_motion_effect(
        self,
        image: Image.Image,
        motion_type: str,
        progress: float
    ) -> Image.Image:
        """Apply motion effect to image"""
        
        width, height = image.size
        
        if motion_type == "zoom":
            # Zoom effect
            scale = 1.0 + (progress * 0.3)  # Zoom from 1.0 to 1.3
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize image
            resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Crop to original size from center
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            right = left + width
            bottom = top + height
            
            return resized.crop((left, top, right, bottom))
        
        elif motion_type == "pan":
            # Pan effect (horizontal movement)
            offset_x = int(progress * width * 0.2)  # Move 20% of width
            offset_y = 0
            
            # Create new image with offset
            new_image = Image.new("RGB", (width, height))
            new_image.paste(image, (-offset_x, -offset_y))
            
            return new_image
        
        elif motion_type == "rotate":
            # Rotation effect
            angle = progress * 10  # Rotate up to 10 degrees
            return image.rotate(angle, expand=False, resample=Image.Resampling.BICUBIC)
        
        elif motion_type == "wave":
            # Wave effect (subtle distortion)
            img_array = np.array(image)
            for y in range(height):
                offset = int(5 * np.sin(y * 0.1 + progress * 2 * np.pi))
                if 0 <= y + offset < height:
                    img_array[y] = img_array[(y + offset) % height]
            return Image.fromarray(img_array)
        
        elif motion_type == "pulse":
            # Pulse effect (brightness variation)
            img_array = np.array(image).astype(np.float32)
            pulse = 1.0 + 0.2 * np.sin(progress * 2 * np.pi)
            img_array = img_array * pulse
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            return Image.fromarray(img_array)
        
        else:  # gentle
            # Gentle movement (subtle zoom + slight pan)
            scale = 1.0 + (progress * 0.1)  # Subtle zoom
            offset_x = int(progress * width * 0.05)  # Subtle pan
            offset_y = int(progress * height * 0.02)
            
            # Apply zoom
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Crop and apply pan
            left = (new_width - width) // 2 - offset_x
            top = (new_height - height) // 2 - offset_y
            right = left + width
            bottom = top + height
            
            return resized.crop((left, top, right, bottom))
    
    def _interpolate_frames(self, frames: List[Image.Image], target_fps: int) -> List[Image.Image]:
        """Apply frame interpolation for smoother motion"""
        
        if len(frames) < 2:
            return frames
        
        # Simple frame interpolation
        interpolated_frames = []
        
        for i in range(len(frames) - 1):
            frame1 = frames[i]
            frame2 = frames[i + 1]
            
            # Add original frame
            interpolated_frames.append(frame1)
            
            # Create interpolated frame (simple blend)
            interpolated = Image.blend(frame1, frame2, 0.5)
            interpolated_frames.append(interpolated)
        
        # Add last frame
        interpolated_frames.append(frames[-1])
        
        return interpolated_frames
    
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
    
    def get_pipeline_status(self) -> Dict[str, bool]:
        """Get status of available pipelines"""
        return {
            "interpolation": True,  # Always available
            "stable_video": False,
            "modelscope_t2v": False,
            "zeroscope": False,
            "animatediff": False
        }
    
    def get_available_pipelines(self) -> Dict[str, str]:
        """Get available pipeline descriptions"""
        return {
            "interpolation": "Frame Interpolation (Motion Effects)",
            "stable_video": "Stable Video Diffusion (Image-to-Video)",
            "modelscope_t2v": "ModelScope T2V (Text-to-Video)",
            "zeroscope": "Zeroscope v2 XL (HD Text-to-Video)",
            "animatediff": "AnimateDiff-like (Image Animation)"
        }

def main():
    """Test the simple video system"""
    
    print("🎬 Simple Video Generation System")
    print("=" * 40)
    
    system = SimpleVideoSystem()
    
    # Test Option A: Image to Video
    print("\n🔁 Testing Option A: Image → Motion → Video")
    try:
        # Create a test image
        test_image = Image.new("RGB", (512, 512), color=(100, 150, 200))
        
        video_path = system.generate_video_from_image_pipeline(
            image=test_image,
            motion_prompt="gentle zoom",
            duration_seconds=3,
            fps=24,
            width=512,
            height=512
        )
        print(f"✅ Option A successful: {video_path}")
    except Exception as e:
        print(f"❌ Option A failed: {e}")
    
    # Test Option B: Direct Text to Video
    print("\n🎥 Testing Option B: Text to Video Direct")
    try:
        video_path = system.generate_direct_text_to_video(
            prompt="A beautiful sunset over mountains",
            duration_seconds=3,
            fps=24,
            width=512,
            height=512
        )
        print(f"✅ Option B successful: {video_path}")
    except Exception as e:
        print(f"❌ Option B failed: {e}")
    
    print("\n🎉 Simple video system test complete!")

if __name__ == "__main__":
    main() 