#!/bin/bash

# MPS Training Launcher for Mac M3
# This script sets up the proper environment and launches LoRA training

echo "🚀 Launching MPS LoRA Training on Mac M3..."

# Set MPS environment variables
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_MPS_LOW_MEMORY_FALLBACK=1

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if we're in the right directory
if [ ! -f "training_config.json" ]; then
    echo "❌ Error: training_config.json not found. Please run this script from the backend directory."
    exit 1
fi

# Check MPS availability
echo "🔍 Checking MPS availability..."
python -c "
import torch
if torch.backends.mps.is_available():
    print('✅ MPS (Apple Silicon GPU) is available')
    print(f'PyTorch version: {torch.__version__}')
else:
    print('❌ MPS not available')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ MPS not available. Please check your PyTorch installation."
    exit 1
fi

# Check Hugging Face authentication
echo "🔐 Checking Hugging Face authentication..."
python -c "
from huggingface_hub import whoami
try:
    user = whoami()
    print(f'✅ Authenticated as: {user}')
except Exception as e:
    print(f'❌ Authentication failed: {e}')
    print('Please run: huggingface-cli login')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Hugging Face authentication failed."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p lora_weights
mkdir -p training_logs
mkdir -p custom_model_outputs

# Start training
echo "🎯 Starting LoRA training with LAION-2B..."
echo "📊 Training will use your Mac M3 GPU for acceleration"
echo "⏱️  Expected duration: 2-4 hours"
echo "💾 Checkpoints will be saved every 1000 steps"
echo ""

python train_custom_model_mps.py

# Check training result
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
    echo "🎉 Your custom LoRA model is ready!"
    echo ""
    echo "📁 Output files:"
    echo "   - custom_model_outputs/lora_model/ (final model)"
    echo "   - lora_weights/ (checkpoints)"
    echo "   - training_logs/ (training history)"
    echo ""
    echo "🚀 Next steps:"
    echo "   1. Test your model: python test_custom_model.py"
    echo "   2. Deploy to API: python deploy_custom_model.py"
    echo "   3. Generate HD images: python generate_hd.py"
else
    echo ""
    echo "❌ Training failed. Check the logs above for details."
    exit 1
fi 