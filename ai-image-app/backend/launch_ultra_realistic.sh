#!/bin/bash

# Ultra-Realistic Generation System Launcher
# This script sets up and launches the ultra-realistic image and video generation system

echo "🚀 Ultra-Realistic Generation System"
echo "=================================="
echo ""

# Check if we're in the right directory
if [ ! -f "ultra_realistic_system.py" ]; then
    echo "❌ Error: ultra_realistic_system.py not found!"
    echo "Please run this script from the backend directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install additional dependencies for ultra-realistic system
echo "Installing ultra-realistic dependencies..."
pip install opencv-python
pip install diffusers[torch]
pip install transformers
pip install accelerate
pip install safetensors

# Check if Hugging Face token is set
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "⚠️  Warning: No Hugging Face token found in environment."
    echo "Some models may not download properly."
    echo "Set HUGGINGFACE_TOKEN environment variable for full access."
else
    echo "✅ Hugging Face token found."
fi

# Create necessary directories
echo "Creating output directories..."
mkdir -p ultra_realistic_outputs
mkdir -p static

# Test the system
echo ""
echo "🧪 Testing ultra-realistic system..."
python -c "
from ultra_realistic_system import UltraRealisticSystem
try:
    system = UltraRealisticSystem()
    print('✅ Ultra-realistic system initialized successfully!')
    print(f'📁 Output directory: {system.output_dir}')
    print(f'🖥️  Device: {system.device}')
    print(f'📦 Available models: {list(system.pipelines.keys())}')
except Exception as e:
    print(f'❌ Error initializing system: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 System ready! Choose an option:"
    echo ""
    echo "1. 🖼️  Run Ultra-Realistic API (Flask server)"
    echo "2. 🎨 Generate test images"
    echo "3. 🎬 Generate test video"
    echo "4. 📊 System status"
    echo "5. 🚪 Exit"
    echo ""
    read -p "Enter your choice (1-5): " choice
    
    case $choice in
        1)
            echo "🚀 Starting Ultra-Realistic API..."
            echo "API will be available at: http://localhost:5001"
            echo "Press Ctrl+C to stop"
            python ultra_realistic_api.py
            ;;
        2)
            echo "🎨 Generating test images..."
            python -c "
from ultra_realistic_system import UltraRealisticSystem
system = UltraRealisticSystem()
prompts = [
    'A professional portrait of a confident business person',
    'A majestic lion in the African savanna at golden hour',
    'A futuristic cityscape with flying cars and neon lights'
]
for prompt in prompts:
    print(f'Generating: {prompt}')
    system.generate_ultra_realistic_image(prompt, 'photorealistic')
print('✅ Test images generated!')
"
            ;;
        3)
            echo "🎬 Generating test video..."
            python -c "
from ultra_realistic_system import UltraRealisticSystem
from PIL import Image
import numpy as np
system = UltraRealisticSystem()
# Create a test image first
test_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
test_path = 'test_image.png'
test_image.save(test_path)
# Generate video
video_path = system.generate_video_from_image(test_path, 'gentle zoom', 5, 24)
if video_path:
    print(f'✅ Test video generated: {video_path}')
else:
    print('❌ Video generation failed')
"
            ;;
        4)
            echo "📊 System Status:"
            python -c "
from ultra_realistic_system import UltraRealisticSystem
import torch
system = UltraRealisticSystem()
print(f'Device: {system.device}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'Available Models: {list(system.pipelines.keys())}')
print(f'Output Directory: {system.output_dir}')
"
            ;;
        5)
            echo "👋 Goodbye!"
            exit 0
            ;;
        *)
            echo "❌ Invalid choice. Exiting."
            exit 1
            ;;
    esac
else
    echo "❌ System test failed. Please check the error messages above."
    exit 1
fi 