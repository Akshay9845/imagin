#!/usr/bin/env python3
"""
Deploy Custom Trained Model to Web Application
- Integrates custom model into Flask API
- Supports both SD v1.5 and SDXL
- Includes model switching capabilities
"""

import os
import json
import torch
from pathlib import Path
from typing import Optional, Dict, Any

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler
)
from peft import PeftModel

class CustomModelDeployer:
    def __init__(self, config_path: str = "training_config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.models = {}
        self.current_model = None
        
    def load_config(self) -> Dict[str, Any]:
        """Load training configuration"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            print(f"⚠️  Config file not found: {self.config_path}")
            return {}
    
    def load_custom_model(
        self,
        model_path: str,
        model_name: str = "custom",
        use_sdxl: bool = False
    ):
        """Load a custom trained model"""
        
        print(f"🔄 Loading custom model: {model_name}")
        print(f"📁 Model path: {model_path}")
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model path not found: {model_path}")
        
        try:
            if use_sdxl:
                # Load SDXL pipeline
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    variant="fp16"
                )
            else:
                # Load SD v1.5 pipeline
                pipeline = StableDiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16
                )
            
            # Move to device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipeline = pipeline.to(device)
            
            # Enable optimizations
            if hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing()
            
            if hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()
            
            # Use better scheduler
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                pipeline.scheduler.config
            )
            
            # Store model
            self.models[model_name] = {
                "pipeline": pipeline,
                "use_sdxl": use_sdxl,
                "path": model_path
            }
            
            print(f"✅ Custom model '{model_name}' loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load custom model: {e}")
            return False
    
    def switch_model(self, model_name: str):
        """Switch to a different loaded model"""
        if model_name in self.models:
            self.current_model = model_name
            print(f"🔄 Switched to model: {model_name}")
            return True
        else:
            print(f"❌ Model '{model_name}' not found. Available: {list(self.models.keys())}")
            return False
    
    def generate_with_custom_model(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        model_name: Optional[str] = None
    ):
        """Generate image using custom model"""
        
        # Use specified model or current model
        if model_name is None:
            model_name = self.current_model
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded")
        
        model_info = self.models[model_name]
        pipeline = model_info["pipeline"]
        use_sdxl = model_info["use_sdxl"]
        
        print(f"🎨 Generating with custom model: {model_name}")
        print(f"📝 Prompt: {prompt}")
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Generate image
        with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device=pipeline.device).manual_seed(seed) if seed else None
            )
        
        return result.images[0]
    
    def list_models(self):
        """List all loaded models"""
        print("📋 Loaded Models:")
        for name, info in self.models.items():
            status = "🟢" if name == self.current_model else "⚪"
            print(f"  {status} {name}: {info['path']} (SDXL: {info['use_sdxl']})")
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        if model_name in self.models:
            info = self.models[model_name].copy()
            # Remove pipeline object for JSON serialization
            info.pop("pipeline", None)
            return info
        return {}

def integrate_with_flask():
    """Integration function for Flask app"""
    
    # Initialize deployer
    deployer = CustomModelDeployer()
    
    # Load your custom model
    custom_model_path = "custom_model_outputs/deployable_model"
    if os.path.exists(custom_model_path):
        success = deployer.load_custom_model(
            model_path=custom_model_path,
            model_name="custom_trained",
            use_sdxl=False  # Set to True if using SDXL
        )
        
        if success:
            deployer.switch_model("custom_trained")
            print("✅ Custom model integrated with Flask app!")
        else:
            print("⚠️  Using base model for Flask app")
    else:
        print("⚠️  Custom model not found, using base model")
    
    return deployer

def create_flask_routes(deployer):
    """Create Flask routes for custom model"""
    
    from flask import Flask, request, jsonify, send_from_directory
    from flask_cors import CORS
    import io
    import base64
    
    app = Flask(__name__)
    CORS(app)
    
    @app.route("/generate_custom", methods=["POST"])
    def generate_custom():
        """Generate image with custom model"""
        try:
            data = request.json
            prompt = data.get("prompt", "")
            negative_prompt = data.get("negative_prompt", "")
            width = data.get("width", 512)
            height = data.get("height", 512)
            num_steps = data.get("num_inference_steps", 30)
            guidance_scale = data.get("guidance_scale", 7.5)
            seed = data.get("seed")
            model_name = data.get("model_name")
            
            # Generate image
            image = deployer.generate_with_custom_model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                model_name=model_name
            )
            
            # Save image
            filename = f"custom_{prompt[:30].replace(' ', '_')}.png"
            output_path = f"static/{filename}"
            image.save(output_path)
            
            return jsonify({
                "success": True,
                "image_path": output_path,
                "model_used": deployer.current_model
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @app.route("/models", methods=["GET"])
    def list_models():
        """List available models"""
        models = {}
        for name, info in deployer.models.items():
            models[name] = {
                "path": info["path"],
                "use_sdxl": info["use_sdxl"],
                "is_current": name == deployer.current_model
            }
        
        return jsonify({
            "models": models,
            "current_model": deployer.current_model
        })
    
    @app.route("/switch_model", methods=["POST"])
    def switch_model():
        """Switch to a different model"""
        try:
            data = request.json
            model_name = data.get("model_name")
            
            if deployer.switch_model(model_name):
                return jsonify({
                    "success": True,
                    "current_model": model_name
                })
            else:
                return jsonify({
                    "success": False,
                    "error": f"Model '{model_name}' not found"
                }), 400
                
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    return app

def main():
    """Main deployment function"""
    
    print("🚀 Custom Model Deployment")
    print("=" * 50)
    
    # Initialize deployer
    deployer = CustomModelDeployer()
    
    # Load custom model
    custom_model_path = "custom_model_outputs/deployable_model"
    if os.path.exists(custom_model_path):
        print(f"📦 Found custom model at: {custom_model_path}")
        
        # Try to load the model
        success = deployer.load_custom_model(
            model_path=custom_model_path,
            model_name="custom_trained",
            use_sdxl=False  # Set to True if using SDXL
        )
        
        if success:
            deployer.switch_model("custom_trained")
            
            # Test generation
            print("\n🧪 Testing custom model...")
            try:
                test_image = deployer.generate_with_custom_model(
                    prompt="A beautiful test image generated by our custom model",
                    width=512,
                    height=512,
                    num_inference_steps=20
                )
                
                # Save test image
                test_image.save("static/test_custom_model.png")
                print("✅ Custom model test successful!")
                
            except Exception as e:
                print(f"❌ Custom model test failed: {e}")
        else:
            print("⚠️  Custom model loading failed")
    else:
        print(f"⚠️  Custom model not found at: {custom_model_path}")
    
    # List all models
    deployer.list_models()
    
    print("\n🎉 Deployment setup completed!")
    print("📝 To use in your Flask app, call integrate_with_flask()")

if __name__ == "__main__":
    main() 