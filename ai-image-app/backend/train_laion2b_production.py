#!/usr/bin/env python3
"""
Production LAION-2B Training Script
Full-scale training on the complete 2B LAION-2B dataset
"""

import json
import os
from pathlib import Path

def create_production_config():
    """Create production configuration for full 2B training"""
    production_config = {
        "training": {
            "base_model": "runwayml/stable-diffusion-v1-5",
            "resolution": 512,
            "batch_size": 2,  # Increased for production
            "gradient_accumulation_steps": 8,  # Increased for stability
            "learning_rate": 5e-5,  # Lower learning rate for stability
            "total_samples": 2000000000,  # Full 2B samples
            "save_steps": 10000,  # Save every 10K steps
            "eval_steps": 5000,  # Evaluate every 5K steps
            "logging_steps": 100,  # Log every 100 steps
            "mixed_precision": "fp16",
            "max_grad_norm": 1.0,
            "warmup_steps": 10000,  # Longer warmup
            "lr_scheduler": "cosine"
        },
        "lora": {
            "r": 32,  # Higher rank for better quality
            "lora_alpha": 64,  # Higher alpha
            "target_modules": [
                "to_q", "to_k", "to_v", "to_out.0",
                "ff.net.0.proj", "ff.net.2"
            ],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "FEATURE_EXTRACTION"
        },
        "datasets": [
            {
                "type": "huggingface",
                "name": "laion/laion2B-en",
                "split": "train",
                "streaming": True,
                "shuffle_buffer": 50000  # Larger buffer for production
            }
        ],
        "advanced": {
            "enable_wandb": True,  # Enable for production tracking
            "enable_tensorboard": True,
            "gradient_checkpointing": True,
            "use_8bit_adam": True,  # Use 8-bit Adam for memory efficiency
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_weight_decay": 1e-2,
            "adam_epsilon": 1e-08,
            "max_grad_norm": 1.0,
            "warmup_steps": 10000,
            "lr_scheduler": "cosine",
            "resume_from_checkpoint": None,
            "max_memory_usage": 0.9  # Use more memory for production
        }
    }
    
    with open("production_laion2b_config.json", 'w') as f:
        json.dump(production_config, f, indent=2)
    
    print("✅ Created production config: production_laion2b_config.json")
    return production_config

def create_training_launcher():
    """Create a launcher script for production training"""
    launcher_script = '''#!/bin/bash
# Production LAION-2B Training Launcher
# This script manages the full 2B training process

echo "🚀 Starting Production LAION-2B Training"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "train_laion2b_fixed.py" ]; then
    echo "❌ Error: train_laion2b_fixed.py not found!"
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

# Check GPU/MPS availability
echo "🔍 Checking hardware..."
python -c "
import torch
if torch.backends.mps.is_available():
    print('✅ MPS (Apple Silicon GPU) available')
elif torch.cuda.is_available():
    print('✅ CUDA GPU available')
else:
    print('⚠️ Using CPU - training will be very slow!')
"

# Create production config
echo "📝 Creating production configuration..."
python -c "
from train_laion2b_production import create_production_config
create_production_config()
"

# Start training
echo "🎯 Starting production training..."
echo "📊 Target: 2B samples"
echo "⏱️ Estimated time: 2-4 weeks (depending on hardware)"
echo "💾 Checkpoints will be saved every 10K steps"
echo ""

# Run training with nohup for background execution
nohup python train_laion2b_fixed.py production_laion2b_config.json > production_training.log 2>&1 &

# Get the process ID
TRAINING_PID=$!
echo "✅ Training started with PID: $TRAINING_PID"
echo "📝 Logs are being written to: production_training.log"
echo ""
echo "🔍 Monitor training progress:"
echo "   tail -f production_training.log"
echo ""
echo "🛑 Stop training:"
echo "   kill $TRAINING_PID"
echo ""
echo "📊 Check training status:"
echo "   ps aux | grep train_laion2b_fixed"
'''
    
    with open("launch_production_training.sh", 'w') as f:
        f.write(launcher_script)
    
    # Make executable
    os.chmod("launch_production_training.sh", 0o755)
    
    print("✅ Created launcher script: launch_production_training.sh")

def create_monitoring_script():
    """Create a script to monitor training progress"""
    monitoring_script = '''#!/usr/bin/env python3
"""
Production Training Monitor
Monitors the progress of LAION-2B training
"""

import json
import time
import os
from pathlib import Path
from datetime import datetime, timedelta

def get_training_status():
    """Get current training status"""
    # Check if training is running
    import subprocess
    result = subprocess.run(['pgrep', '-f', 'train_laion2b_fixed'], 
                          capture_output=True, text=True)
    is_running = result.returncode == 0
    
    # Check latest checkpoint
    checkpoint_dirs = list(Path("laion2b_training_outputs/checkpoints").glob("step_*"))
    latest_checkpoint = None
    if checkpoint_dirs:
        latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.name.split("_")[1]))
    
    # Check log file
    log_file = "production_training.log"
    last_log_lines = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
            last_log_lines = lines[-10:]  # Last 10 lines
    
    return {
        "is_running": is_running,
        "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint else None,
        "log_lines": last_log_lines
    }

def main():
    print("📊 Production Training Monitor")
    print("=" * 40)
    
    status = get_training_status()
    
    print(f"🔄 Training Running: {'✅ Yes' if status['is_running'] else '❌ No'}")
    
    if status['latest_checkpoint']:
        step = status['latest_checkpoint'].split("_")[1]
        print(f"📁 Latest Checkpoint: Step {step}")
    
    print("\n📝 Recent Logs:")
    for line in status['log_lines']:
        print(f"   {line.strip()}")
    
    print("\n🔍 Commands:")
    print("   Monitor logs: tail -f production_training.log")
    print("   Check status: python monitor_production.py")
    print("   Stop training: pkill -f train_laion2b_fixed")

if __name__ == "__main__":
    main()
'''
    
    with open("monitor_production.py", 'w') as f:
        f.write(monitoring_script)
    
    print("✅ Created monitoring script: monitor_production.py")

if __name__ == "__main__":
    print("🚀 Production LAION-2B Training Setup")
    print("=" * 50)
    
    # Create production config
    create_production_config()
    
    # Create launcher script
    create_training_launcher()
    
    # Create monitoring script
    create_monitoring_script()
    
    print("\n✅ Production setup complete!")
    print("\n📝 Next steps:")
    print("   1. Test with small run: python test_laion2b_fix.py")
    print("   2. Start production training: ./launch_production_training.sh")
    print("   3. Monitor progress: python monitor_production.py")
    print("   4. Deploy when complete: python deploy_production_model.py")
    
    print("\n⚠️ Important notes:")
    print("   - Full training will take 2-4 weeks")
    print("   - Ensure you have sufficient storage (100GB+)")
    print("   - Monitor system resources during training")
    print("   - Checkpoints are saved every 10K steps") 