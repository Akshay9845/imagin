#!/usr/bin/env python3
"""
Production Model Deployment Script
Deploys the trained LAION-2B model for production use
"""

import os
import json
import torch
import shutil
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionModelDeployer:
    def __init__(self):
        self.device = self.setup_device()
        self.base_model = "runwayml/stable-diffusion-v1-5"
        
    def setup_device(self):
        """Setup device with MPS support"""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def find_latest_checkpoint(self):
        """Find the latest training checkpoint"""
        checkpoint_dir = Path("laion2b_training_outputs/checkpoints")
        if not checkpoint_dir.exists():
            logger.warning("No training checkpoints found!")
            return None
        
        checkpoint_dirs = list(checkpoint_dir.glob("step_*"))
        if not checkpoint_dirs:
            logger.warning("No checkpoint directories found!")
            return None
        
        # Find the latest checkpoint by step number
        latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.name.split("_")[1]))
        logger.info(f"Found latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    
    def validate_checkpoint(self, checkpoint_path):
        """Validate that the checkpoint is complete"""
        required_files = [
            "adapter_config.json",
            "adapter_model.safetensors"
        ]
        
        for file in required_files:
            if not (checkpoint_path / file).exists():
                logger.error(f"Missing required file: {file}")
                return False
        
        # Check training state
        training_state_file = checkpoint_path / "training_state.json"
        if training_state_file.exists():
            with open(training_state_file, 'r') as f:
                training_state = json.load(f)
            logger.info(f"Checkpoint info: Step {training_state.get('step', 'unknown')}, "
                       f"Loss: {training_state.get('loss', 'unknown'):.4f}")
        
        return True
    
    def create_production_config(self, checkpoint_path):
        """Create production configuration for the deployed model"""
        config = {
            "model_info": {
                "base_model": self.base_model,
                "lora_checkpoint": str(checkpoint_path),
                "deployment_date": datetime.now().isoformat(),
                "device": self.device,
                "resolution": 512,
                "batch_size": 1
            },
            "generation_settings": {
                "num_inference_steps": 20,
                "guidance_scale": 7.5,
                "negative_prompt": "blurry, low quality, distorted, ugly, bad anatomy",
                "max_prompt_length": 200
            },
            "api_settings": {
                "port": 5001,
                "host": "0.0.0.0",
                "max_concurrent_requests": 4,
                "timeout": 300
            }
        }
        
        config_path = "production_model_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Created production config: {config_path}")
        return config_path
    
    def update_flask_app(self, checkpoint_path):
        """Update the Flask app to use the production model"""
        app_code = f'''#!/usr/bin/env python3
"""
Production AI Image Generator API
Uses the trained LAION-2B model for high-quality image generation
"""

import os
import torch
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import time
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set MPS environment variables for Mac
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

app = Flask(__name__)
CORS(app)

class ProductionImageGenerator:
    def __init__(self):
        self.device = self.setup_device()
        self.pipeline = None
        self.load_production_model()
    
    def setup_device(self):
        """Setup device with MPS support"""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def load_production_model(self):
        """Load the production model with LoRA weights"""
        logger.info("🔗 Loading production model...")
        
        # Load base pipeline
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            "{self.base_model}",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Load LoRA weights
        lora_path = "{checkpoint_path}"
        if Path(lora_path).exists():
            logger.info(f"🔗 Loading LoRA weights from: {{lora_path}}")
            self.pipeline.unet = PeftModel.from_pretrained(
                self.pipeline.unet, 
                lora_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            logger.info("✅ Production LoRA loaded successfully!")
        else:
            logger.warning("⚠️ LoRA weights not found, using base model")
        
        # Move to device
        self.pipeline = self.pipeline.to(self.device)
        
        # Enable memory efficient attention if available
        if hasattr(self.pipeline.unet, "enable_xformers_memory_efficient_attention"):
            self.pipeline.unet.enable_xformers_memory_efficient_attention()
        
        logger.info(f"✅ Production model loaded on {{self.device}}")
    
    def generate_image(self, prompt, negative_prompt="", num_steps=20, guidance_scale=7.5):
        """Generate image with production model"""
        try:
            # Generate image
            image = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                width=512,
                height=512
            ).images[0]
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"production_{{timestamp}}.png"
            image_path = f"static/{{filename}}"
            image.save(image_path)
            
            return {{
                "success": True,
                "image_path": f"/static/{{filename}}",
                "prompt": prompt,
                "model": "production_laion2b"
            }}
            
        except Exception as e:
            logger.error(f"Generation failed: {{e}}")
            return {{
                "success": False,
                "error": str(e)
            }}

# Initialize generator
generator = ProductionImageGenerator()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({{
        "status": "healthy",
        "model": "production_laion2b",
        "device": generator.device,
        "timestamp": datetime.now().isoformat()
    }})

@app.route('/generate', methods=['POST'])
def generate_image():
    """Generate image endpoint"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        negative_prompt = data.get('negative_prompt', '')
        num_steps = data.get('num_steps', 20)
        guidance_scale = data.get('guidance_scale', 7.5)
        
        if not prompt:
            return jsonify({{"error": "Prompt is required"}}), 400
        
        result = generator.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_steps=num_steps,
            guidance_scale=guidance_scale
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({{"error": str(e)}}), 500

@app.route('/generate_fast', methods=['POST'])
def generate_fast():
    """Fast generation endpoint with optimized settings"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({{"error": "Prompt is required"}}), 400
        
        result = generator.generate_image(
            prompt=prompt,
            negative_prompt="blurry, low quality, distorted, ugly, bad anatomy",
            num_steps=20,
            guidance_scale=7.5
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({{"error": str(e)}}), 500

@app.route('/static/<filename>')
def serve_image(filename):
    """Serve generated images"""
    return send_from_directory('static', filename)

if __name__ == '__main__':
    logger.info("🚀 Starting Production AI Image Generator API on port 5001...")
    app.run(host='0.0.0.0', port=5001, debug=True)
'''
        
        with open("app_production.py", 'w') as f:
            f.write(app_code)
        
        logger.info("✅ Updated Flask app: app_production.py")
    
    def create_deployment_script(self):
        """Create a deployment launcher script"""
        deployment_script = '''#!/bin/bash
# Production Model Deployment Launcher

echo "🚀 Deploying Production LAION-2B Model"
echo "====================================="

# Check if we're in the right directory
if [ ! -f "deploy_production_model.py" ]; then
    echo "❌ Error: deploy_production_model.py not found!"
    echo "Please run this script from the backend directory"
    exit 1
fi

# Activate virtual environment
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
else
    echo "❌ Virtual environment not found!"
    exit 1
fi

# Deploy the model
echo "📦 Deploying production model..."
python deploy_production_model.py

# Start the production server
echo "🚀 Starting production server..."
python app_production.py
'''
        
        with open("deploy_and_run.sh", 'w') as f:
            f.write(deployment_script)
        
        os.chmod("deploy_and_run.sh", 0o755)
        logger.info("✅ Created deployment script: deploy_and_run.sh")
    
    def deploy(self):
        """Main deployment process"""
        logger.info("🚀 Starting Production Model Deployment")
        logger.info("=" * 50)
        
        # Find latest checkpoint
        checkpoint_path = self.find_latest_checkpoint()
        if not checkpoint_path:
            logger.error("❌ No training checkpoint found!")
            logger.info("Please run training first: python train_laion2b_fixed.py")
            return False
        
        # Validate checkpoint
        if not self.validate_checkpoint(checkpoint_path):
            logger.error("❌ Checkpoint validation failed!")
            return False
        
        # Create production config
        config_path = self.create_production_config(checkpoint_path)
        
        # Update Flask app
        self.update_flask_app(checkpoint_path)
        
        # Create deployment script
        self.create_deployment_script()
        
        logger.info("✅ Production deployment setup complete!")
        logger.info("\n📝 Next steps:")
        logger.info("   1. Start production server: ./deploy_and_run.sh")
        logger.info("   2. Or manually: python app_production.py")
        logger.info("   3. Test generation: curl -X POST http://localhost:5001/generate")
        logger.info("   4. Access web UI: http://localhost:3000")
        
        return True

def main():
    deployer = ProductionModelDeployer()
    success = deployer.deploy()
    
    if success:
        print("\n🎉 Production model deployed successfully!")
    else:
        print("\n❌ Deployment failed! Check logs for details.")

if __name__ == "__main__":
    main() 