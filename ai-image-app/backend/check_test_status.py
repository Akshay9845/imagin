#!/usr/bin/env python3
"""
Quick Status Check for LAION-2B Test Training
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime

def check_test_status():
    """Check the status of the current test training run"""
    print("🧪 LAION-2B Test Training Status")
    print("=" * 40)
    
    # Check if training is running
    import subprocess
    result = subprocess.run(['pgrep', '-f', 'test_laion2b_fix'], 
                          capture_output=True, text=True)
    is_running = result.returncode == 0
    
    print(f"🔄 Training Running: {'✅ Yes' if is_running else '❌ No'}")
    
    # Check log file
    log_file = "laion2b_training.log"
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
            if lines:
                print(f"📝 Log file: {log_file} ({len(lines)} lines)")
                
                # Show last few lines
                print("\n📋 Recent Logs:")
                for line in lines[-5:]:
                    print(f"   {line.strip()}")
            else:
                print("📝 Log file is empty")
    else:
        print("📝 No log file found")
    
    # Check checkpoints
    checkpoint_dir = Path("laion2b_training_outputs/checkpoints")
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("step_*"))
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.name.split("_")[1]))
            print(f"\n💾 Latest Checkpoint: {latest.name}")
            
            # Check checkpoint info
            training_state_file = latest / "training_state.json"
            if training_state_file.exists():
                with open(training_state_file, 'r') as f:
                    state = json.load(f)
                print(f"   Step: {state.get('step', 'unknown')}")
                print(f"   Loss: {state.get('loss', 'unknown'):.4f}")
                print(f"   Samples: {state.get('total_samples', 'unknown'):,}")
        else:
            print("\n💾 No checkpoints found yet")
    else:
        print("\n💾 No checkpoint directory found")
    
    # Check test config
    config_file = "test_laion2b_config.json"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        target_samples = config["training"]["total_samples"]
        print(f"\n🎯 Target: {target_samples:,} samples")
    
    print("\n🔍 Commands:")
    print("   Monitor logs: tail -f laion2b_training.log")
    print("   Check status: python check_test_status.py")
    print("   Stop training: pkill -f test_laion2b_fix")

if __name__ == "__main__":
    check_test_status() 