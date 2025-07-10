#!/usr/bin/env python3
"""
Test LAION-2B Fix with Small Training Run
Tests the fixed training script with 100 steps to verify dataset handling
"""

import json
import os
from pathlib import Path

def create_test_config():
    """Create a test configuration with minimal steps"""
    test_config = {
        "training": {
            "base_model": "runwayml/stable-diffusion-v1-5",
            "resolution": 512,
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1e-4,
            "total_samples": 400,  # 100 steps * 4 grad accum steps
            "save_steps": 50,
            "eval_steps": 25,
            "logging_steps": 10,
            "mixed_precision": "fp16",
            "max_grad_norm": 1.0,
            "warmup_steps": 10,
            "lr_scheduler": "cosine"
        },
        "lora": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": [
                "to_q", "to_k", "to_v", "to_out.0",
                "ff.net.0.proj", "ff.net.2"
            ],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        },
        "datasets": [
            {
                "type": "huggingface",
                "name": "laion/laion2B-en",
                "split": "train",
                "streaming": True,
                "shuffle_buffer": 1000  # Smaller buffer for testing
            }
        ],
        "advanced": {
            "enable_wandb": False,
            "enable_tensorboard": False,  # Disable for testing
            "gradient_checkpointing": True,
            "use_8bit_adam": False,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_weight_decay": 1e-2,
            "adam_epsilon": 1e-08,
            "max_grad_norm": 1.0,
            "warmup_steps": 10,
            "lr_scheduler": "cosine",
            "resume_from_checkpoint": None,
            "max_memory_usage": 0.8
        }
    }
    
    with open("test_laion2b_config.json", 'w') as f:
        json.dump(test_config, f, indent=2)
    
    print("✅ Created test config: test_laion2b_config.json")
    return test_config

def run_test_training():
    """Run the test training with the fixed script"""
    print("🧪 Testing LAION-2B fix with 100 steps...")
    
    # Import the fixed trainer
    from train_laion2b_fixed import FixedLAION2BTrainer
    
    # Create test config
    create_test_config()
    
    # Initialize trainer with test config
    trainer = FixedLAION2BTrainer("test_laion2b_config.json")
    
    # Run training
    try:
        trainer.train()
        print("✅ Test training completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Test training failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 LAION-2B Fix Test")
    print("=" * 50)
    
    success = run_test_training()
    
    if success:
        print("\n✅ Test passed! Ready for full training.")
        print("📝 Next steps:")
        print("   1. Run full training with: python train_laion2b_fixed.py")
        print("   2. Deploy trained model with: python deploy_production_model.py")
    else:
        print("\n❌ Test failed! Check logs for details.") 