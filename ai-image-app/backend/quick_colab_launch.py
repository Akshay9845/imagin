#!/usr/bin/env python3
"""
Quick Colab Launch for LAION-2B Training
Direct integration from Cursor to Colab
"""

import webbrowser
import json
from pathlib import Path
import subprocess
import os

def create_colab_notebook():
    """Create a Colab notebook with our training script"""
    print("📝 Creating Colab notebook...")
    
    # Read our training script
    script_path = Path("train_laion2b_colab_progressive.py")
    if script_path.exists():
        with open(script_path, 'r') as f:
            training_script = f.read()
    else:
        training_script = "# Training script will be downloaded"
    
    # Create notebook content
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🚀 LAION-2B Progressive Training\n",
                    "**Launched from Cursor**\n",
                    "\n",
                    "This notebook will train your custom LoRA model using the progressive approach:\n",
                    "- Phase 1: 1,000 samples (5 minutes)\n",
                    "- Phase 2: 5,000 samples (15 minutes)\n",
                    "- Phase 3: 10,000 samples (30 minutes)\n",
                    "- Phase 4: 50,000 samples (2-3 hours)\n",
                    "\n",
                    "**Total Time: 2.5-3.5 hours on Colab Free**"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Check GPU\n",
                    "import torch\n",
                    "print(f\"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}\")\n",
                    "print(f\"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\" if torch.cuda.is_available() else 'No GPU')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Install required packages\n",
                    "!pip install -q diffusers transformers accelerate peft datasets xformers webdataset torchvision pillow requests\n",
                    "print(\"✅ Packages installed\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Download training script\n",
                    "import requests\n",
                    "\n",
                    "# Get the full training script\n",
                    "script_content = '''" + training_script.replace("'", "\\'").replace("\n", "\\n") + "'''\n",
                    "\n",
                    "with open('train_laion2b_colab_progressive.py', 'w') as f:\n",
                    "    f.write(script_content)\n",
                    "\n",
                    "print(\"✅ Training script downloaded\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Run progressive training\n",
                    "!python train_laion2b_colab_progressive.py"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Download results\n",
                    "from google.colab import files\n",
                    "import zipfile\n",
                    "import os\n",
                    "\n",
                    "# Create zip file\n",
                    "!zip -r progressive_trained_model.zip progressive_training_outputs/\n",
                    "\n",
                    "# Download to local machine\n",
                    "files.download('progressive_trained_model.zip')\n",
                    "\n",
                    "print(\"✅ Model downloaded successfully!\")\n",
                    "print(\"📁 Check your Downloads folder for 'progressive_trained_model.zip'\")"
                ]
            }
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    # Save notebook
    notebook_path = Path("cursor_laion2b_training.ipynb")
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✅ Notebook created: {notebook_path}")
    return notebook_path

def launch_colab():
    """Launch Colab in browser"""
    print("🌐 Launching Google Colab...")
    
    # Open Colab
    colab_url = "https://colab.research.google.com"
    webbrowser.open(colab_url)
    
    print("✅ Colab opened in browser")
    return colab_url

def show_instructions():
    """Show step-by-step instructions"""
    print("\n" + "="*60)
    print("🚀 COLAB TRAINING INSTRUCTIONS")
    print("="*60)
    print("\n📋 Step-by-Step Guide:")
    print("\n1️⃣ Upload Notebook:")
    print("   • Go to the opened Colab tab")
    print("   • Click 'File' → 'Upload notebook'")
    print("   • Select 'cursor_laion2b_training.ipynb'")
    print("   • Click 'Upload'")
    
    print("\n2️⃣ Enable GPU:")
    print("   • Click 'Runtime' → 'Change runtime type'")
    print("   • Set 'Hardware accelerator' to 'GPU'")
    print("   • Choose 'T4' (free) or 'A100' (Pro)")
    print("   • Click 'Save'")
    
    print("\n3️⃣ Run Training:")
    print("   • Click 'Runtime' → 'Run all'")
    print("   • Or run cells one by one")
    print("   • Monitor progress in real-time")
    
    print("\n4️⃣ Download Results:")
    print("   • Training will automatically download results")
    print("   • Check your Downloads folder")
    print("   • Extract 'progressive_trained_model.zip'")
    
    print("\n⏱️ Expected Timeline:")
    print("   • Phase 1: 5 minutes")
    print("   • Phase 2: 15 minutes")
    print("   • Phase 3: 30 minutes")
    print("   • Phase 4: 2-3 hours")
    print("   • Total: 2.5-3.5 hours")
    
    print("\n💡 Tips:")
    print("   • Keep Colab tab open during training")
    print("   • Don't close browser or computer")
    print("   • Monitor GPU usage in Colab")
    print("   • Download checkpoints after each phase")

def main():
    """Main function"""
    print("🚀 Quick Colab Launch for LAION-2B Training")
    print("="*60)
    
    # Create notebook
    notebook_path = create_colab_notebook()
    
    # Launch Colab
    colab_url = launch_colab()
    
    # Show instructions
    show_instructions()
    
    print(f"\n🎉 Setup Complete!")
    print(f"📁 Notebook: {notebook_path}")
    print(f"🌐 Colab URL: {colab_url}")
    print(f"\n🚀 Ready to train your custom AI model!")

if __name__ == "__main__":
    main() 