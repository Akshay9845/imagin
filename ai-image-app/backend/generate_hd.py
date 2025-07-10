#!/usr/bin/env python3
"""
High-Resolution Image Generation Script
- Supports up to 1080p generation
- Uses trained custom model
- Includes upscaling options
"""

import torch
import os
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import numpy as np

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionImg2ImgPipeline,
    DPMSolverMultistepScheduler
)
from peft import PeftModel

class HDImageGenerator:
    def __init__(
        self,
        model_path: str,
        use_sdxl: bool = False,
        device: str = "auto",
        enable_upscaling: bool = True
    ):
        self.model_path = model_path
        self.use_sdxl = use_sdxl
        self.enable_upscaling = enable_upscaling
        
        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"🚀 Initializing HD Image Generator on {self.device}")
        
        # Load model
        self.load_model()
        
    def load_model(self):
        """Load the trained model"""
        print(f"📦 Loading model from {self.model_path}")
        
        if self.use_sdxl:
            # Load SDXL pipeline
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                variant="fp16"
            )
        else:
            # Load SD v1.5 pipeline
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16
            )
        
        # Move to device
        self.pipeline = self.pipeline.to(self.device)
        
        # Enable memory optimizations
        if hasattr(self.pipeline, "enable_attention_slicing"):
            self.pipeline.enable_attention_slicing()
        
        if hasattr(self.pipeline, "enable_vae_slicing"):
            self.pipeline.enable_vae_slicing()
        
        # Use better scheduler for quality
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config
        )
        
        print("✅ Model loaded successfully!")
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        output_path: Optional[str] = None
    ) -> Image.Image:
        """Generate high-resolution image"""
        
        print(f"🎨 Generating {width}x{height} image...")
        print(f"📝 Prompt: {prompt}")
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Generate image
        with torch.autocast(self.device):
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device=self.device).manual_seed(seed) if seed else None
            )
        
        image = result.images[0]
        
        # Upscale if enabled and resolution is below target
        if self.enable_upscaling and (width < 1920 or height < 1080):
            image = self.upscale_image(image, target_width=1920, target_height=1080)
        
        # Save if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            image.save(output_path)
            print(f"💾 Image saved: {output_path}")
        
        return image
    
    def upscale_image(
        self,
        image: Image.Image,
        target_width: int = 1920,
        target_height: int = 1080,
        method: str = "lanczos"
    ) -> Image.Image:
        """Upscale image to target resolution"""
        
        current_width, current_height = image.size
        
        if current_width >= target_width and current_height >= target_height:
            return image
        
        print(f"🔍 Upscaling from {current_width}x{current_height} to {target_width}x{target_height}")
        
        # Calculate scaling factors
        scale_x = target_width / current_width
        scale_y = target_height / current_height
        scale = max(scale_x, scale_y)
        
        # Calculate new dimensions maintaining aspect ratio
        new_width = int(current_width * scale)
        new_height = int(current_height * scale)
        
        # Upscale
        upscaled = image.resize((new_width, new_height), getattr(Image, method.upper()))
        
        # Crop to target dimensions if needed
        if new_width > target_width or new_height > target_height:
            left = (new_width - target_width) // 2
            top = (new_height - target_height) // 2
            upscaled = upscaled.crop((left, top, left + target_width, top + target_height))
        
        return upscaled
    
    def generate_batch(
        self,
        prompts: list,
        output_dir: str = "hd_outputs",
        **kwargs
    ) -> list:
        """Generate multiple images"""
        
        os.makedirs(output_dir, exist_ok=True)
        images = []
        
        for i, prompt in enumerate(prompts):
            print(f"\n🔄 Generating image {i+1}/{len(prompts)}")
            
            # Create filename
            filename = f"hd_image_{i+1}_{prompt[:30].replace(' ', '_')}.png"
            output_path = os.path.join(output_dir, filename)
            
            # Generate image
            image = self.generate_image(
                prompt=prompt,
                output_path=output_path,
                **kwargs
            )
            
            images.append(image)
        
        return images
    
    def img2img(
        self,
        image_path: str,
        prompt: str,
        strength: float = 0.8,
        **kwargs
    ) -> Image.Image:
        """Image-to-image generation"""
        
        print(f"🔄 Running img2img on {image_path}")
        
        # Load image
        init_image = Image.open(image_path).convert("RGB")
        
        # Create img2img pipeline
        if self.use_sdxl:
            img2img_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                variant="fp16"
            )
        else:
            img2img_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16
            )
        
        img2img_pipeline = img2img_pipeline.to(self.device)
        
        # Generate
        result = img2img_pipeline(
            prompt=prompt,
            image=init_image,
            strength=strength,
            **kwargs
        )
        
        return result.images[0]

def main():
    """Example usage"""
    
    # Configuration
    config = {
        "model_path": "custom_model_outputs/deployable_model",  # Path to your trained model
        "use_sdxl": False,  # Set to True if using SDXL
        "enable_upscaling": True
    }
    
    # Initialize generator
    generator = HDImageGenerator(**config)
    
    # Example prompts for different styles
    prompts = [
        "A cinematic portrait of a wise old wizard in a magical library, dramatic lighting, 8k resolution",
        "A futuristic city skyline at sunset, neon lights, cyberpunk style, ultra high quality",
        "A serene mountain landscape with a crystal clear lake, golden hour lighting, professional photography",
        "A detailed close-up of a dragon's eye, fantasy art, intricate details, masterpiece"
    ]
    
    # Generate images
    images = generator.generate_batch(
        prompts=prompts,
        output_dir="hd_outputs",
        width=1024,
        height=1024,
        num_inference_steps=30,
        guidance_scale=7.5,
        seed=42  # For reproducible results
    )
    
    print(f"\n🎉 Generated {len(images)} high-resolution images!")
    print("📁 Check the 'hd_outputs' directory for your images.")

if __name__ == "__main__":
    main() 