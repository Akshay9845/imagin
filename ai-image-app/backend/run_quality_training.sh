#!/bin/bash

# Quality Training Runner Script
# This script runs high-quality training for better image generation

echo "=== Starting Quality Training ==="
echo "This will train your custom model with high-quality data and parameters"
echo "Training will take longer but produce much better results"
echo ""

# Check if we're in the right directory
if [ ! -f "quality_training.py" ]; then
    echo "Error: quality_training.py not found!"
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

# Install dependencies if needed
echo "Installing dependencies..."
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate peft datasets
pip install pillow numpy

# Check if Hugging Face token is set
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "Warning: No Hugging Face token found in environment."
    echo "Training will use synthetic data only."
    echo "For better results, set HUGGINGFACE_TOKEN environment variable."
else
    echo "Hugging Face token found. Will attempt to download real datasets."
fi

# Run the quality training
echo "Starting quality training..."
echo "This may take 1-2 hours depending on your hardware..."
echo ""

# Run with timeout and progress monitoring
timeout 7200 python quality_training.py  # 2 hour timeout

if [ $? -eq 124 ]; then
    echo "Training timed out after 2 hours."
    echo "Check the quality_training_outputs directory for partial results."
else
    echo "Quality training completed!"
    echo "Check the 'quality_training_outputs' directory for results"
fi 