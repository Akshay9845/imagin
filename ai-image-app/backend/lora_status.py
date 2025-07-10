#!/usr/bin/env python3
"""
LoRA Integration Status
Shows the current status of the LoRA model integration
"""

import json
import os
from pathlib import Path
from datetime import datetime

def get_lora_status():
    """Get current LoRA status"""
    lora_path = "lora_weights/lora_step_50_20250707_231755"
    
    if not Path(lora_path).exists():
        return {
            "status": "❌ Not Found",
            "message": "LoRA weights not found"
        }
    
    # Get file size
    size_mb = round(Path(lora_path).stat().st_size / (1024 * 1024), 2)
    
    # Check if LoRA is integrated
    with open("generate_fast.py", 'r') as f:
        content = f.read()
        is_integrated = "load_lora_weights" in content
    
    # Read LoRA config
    config_path = Path(lora_path) / "adapter_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return {
            "status": "✅ Integrated" if is_integrated else "⚠️ Available",
            "location": lora_path,
            "size_mb": size_mb,
            "task_type": config.get("task_type", "Unknown"),
            "r": config.get("r", "Unknown"),
            "lora_alpha": config.get("lora_alpha", "Unknown"),
            "target_modules": config.get("target_modules", []),
            "integration_date": datetime.now().isoformat(),
            "message": "LoRA is integrated and ready to use!" if is_integrated else "LoRA available but not integrated"
        }
    else:
        return {
            "status": "⚠️ Incomplete",
            "location": lora_path,
            "size_mb": size_mb,
            "message": "LoRA found but config missing"
        }

def main():
    """Display LoRA status"""
    print("🔍 LoRA Integration Status")
    print("=" * 50)
    
    status = get_lora_status()
    
    for key, value in status.items():
        if key == "target_modules":
            print(f"{key}: {', '.join(value)}")
        else:
            print(f"{key}: {value}")
    
    print("\n" + "=" * 50)
    
    if status["status"] == "✅ Integrated":
        print("🎉 Your custom LoRA model is now integrated!")
        print("🔄 Restart your Flask app to use it:")
        print("   cd ai-image-app/backend")
        print("   source venv/bin/activate")
        print("   python app.py")
        print("\n🌐 Then visit: http://localhost:3000")
    else:
        print("⚠️ LoRA needs to be integrated first")

if __name__ == "__main__":
    main() 