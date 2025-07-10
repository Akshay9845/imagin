#!/bin/bash

# Minimal Training Runner Script
# This script runs the minimal training in Cursor

echo "=== Starting Minimal Training ==="
echo "This will train your custom model with a small dataset"
echo ""

# Check if we're in the right directory
if [ ! -f "minimal_training_cursor.py" ]; then
    echo "Error: minimal_training_cursor.py not found!"
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

# Run the training
echo "Starting training..."
python minimal_training_cursor.py

echo "Training completed!"
echo "Check the 'minimal_training_outputs' directory for results" 