#!/bin/bash

# 🚀 One-Click AI Image Generator Launcher
# Complete system for Mac M3 with custom LoRA training and web interface

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_header() {
    echo -e "${PURPLE}🎯 $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    print_error "Please run this script from the backend directory"
    exit 1
fi

print_header "🚀 AI Image Generator - One-Click Launcher"
echo ""

# Function to check system requirements
check_system() {
    print_info "Checking system requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found"
        exit 1
    fi
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        print_error "Virtual environment not found. Please run setup first."
        exit 1
    fi
    
    print_status "System requirements met"
}

# Function to activate virtual environment
activate_venv() {
    print_info "Activating virtual environment..."
    source venv/bin/activate
    print_status "Virtual environment activated"
}

# Function to check MPS availability
check_mps() {
    print_info "Checking MPS (Apple Silicon GPU) availability..."
    
    python3 -c "
import torch
if torch.backends.mps.is_available():
    print('✅ MPS (Apple Silicon GPU) is available')
    print(f'PyTorch version: {torch.__version__}')
else:
    print('❌ MPS not available')
    exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_status "MPS GPU acceleration ready"
    else
        print_warning "MPS not available - will use CPU (slower)"
    fi
}

# Function to check Hugging Face authentication
check_hf_auth() {
    print_info "Checking Hugging Face authentication..."
    
    python3 -c "
from huggingface_hub import whoami
try:
    user = whoami()
    print(f'✅ Authenticated as: {user}')
except Exception as e:
    print(f'❌ Authentication failed: {e}')
    print('Please run: huggingface-cli login')
    exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_status "Hugging Face authentication OK"
    else
        print_warning "Hugging Face authentication failed - some features may not work"
    fi
}

# Function to start backend server
start_backend() {
    print_header "Starting Backend Server"
    
    # Set MPS environment variables
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    
    print_info "Starting Flask API server on port 5001..."
    print_info "API endpoints:"
    print_info "  - POST /generate_fast (base model)"
    print_info "  - POST /generate_custom (custom LoRA)"
    print_info "  - GET /health (health check)"
    
    # Start backend in background
    python3 app.py &
    BACKEND_PID=$!
    
    # Wait a moment for server to start
    sleep 3
    
    # Check if server is running
    if curl -s http://localhost:5001/health > /dev/null; then
        print_status "Backend server started successfully (PID: $BACKEND_PID)"
    else
        print_error "Backend server failed to start"
        exit 1
    fi
}

# Function to start frontend
start_frontend() {
    print_header "Starting Frontend"
    
    # Check if frontend directory exists
    if [ ! -d "../frontend" ]; then
        print_warning "Frontend directory not found"
        return
    fi
    
    print_info "Starting Next.js frontend on port 3000..."
    
    # Start frontend in background
    cd ../frontend
    npm run dev &
    FRONTEND_PID=$!
    cd ../backend
    
    # Wait a moment for server to start
    sleep 5
    
    # Check if frontend is running
    if curl -s http://localhost:3000 > /dev/null; then
        print_status "Frontend started successfully (PID: $FRONTEND_PID)"
    else
        print_warning "Frontend may still be starting up"
    fi
}

# Function to test generation
test_generation() {
    print_header "Testing Image Generation"
    
    print_info "Generating test images..."
    
    if python3 generate_working.py; then
        print_status "Image generation test successful"
        print_info "Check 'generated_images/' folder for results"
    else
        print_error "Image generation test failed"
    fi
}

# Function to show system status
show_status() {
    print_header "System Status"
    
    echo ""
    print_info "Services:"
    
    # Check backend
    if curl -s http://localhost:5001/health > /dev/null; then
        print_status "Backend API: http://localhost:5001"
    else
        print_error "Backend API: Not running"
    fi
    
    # Check frontend
    if curl -s http://localhost:3000 > /dev/null; then
        print_status "Frontend UI: http://localhost:3000"
    else
        print_warning "Frontend UI: May not be running"
    fi
    
    echo ""
    print_info "Available Models:"
    
    # Check for LoRA weights
    if [ -d "lora_weights" ]; then
        LORA_COUNT=$(find lora_weights -name "*.safetensors" | wc -l)
        print_status "LoRA models: $LORA_COUNT found"
    else
        print_warning "LoRA models: None found"
    fi
    
    # Check for generated images
    if [ -d "generated_images" ]; then
        IMAGE_COUNT=$(find generated_images -name "*.png" | wc -l)
        print_status "Generated images: $IMAGE_COUNT found"
    else
        print_warning "Generated images: None found"
    fi
    
    echo ""
    print_info "Quick Commands:"
    echo "  Generate images: python3 generate_working.py"
    echo "  Train LoRA: ./launch_training.sh"
    echo "  Test custom model: python3 test_custom_model.py"
    echo "  Stop servers: pkill -f 'python3 app.py' && pkill -f 'npm run dev'"
}

# Function to show menu
show_menu() {
    echo ""
    print_header "Available Actions"
    echo ""
    echo "1. 🚀 Start Full System (Backend + Frontend)"
    echo "2. 🔧 Start Backend Only"
    echo "3. 🎨 Generate Test Images"
    echo "4. 🧠 Train Custom LoRA Model"
    echo "5. 📊 Show System Status"
    echo "6. 🛑 Stop All Services"
    echo "7. 🔍 Check System Requirements"
    echo "8. 📖 Show Help"
    echo "9. 🚪 Exit"
    echo ""
    read -p "Choose an option (1-9): " choice
    
    case $choice in
        1)
            check_system
            activate_venv
            check_mps
            check_hf_auth
            start_backend
            start_frontend
            test_generation
            show_status
            ;;
        2)
            check_system
            activate_venv
            start_backend
            show_status
            ;;
        3)
            activate_venv
            test_generation
            ;;
        4)
            activate_venv
            check_mps
            check_hf_auth
            ./launch_training.sh
            ;;
        5)
            show_status
            ;;
        6)
            print_info "Stopping all services..."
            pkill -f "python3 app.py" 2>/dev/null || true
            pkill -f "npm run dev" 2>/dev/null || true
            print_status "All services stopped"
            ;;
        7)
            check_system
            activate_venv
            check_mps
            check_hf_auth
            ;;
        8)
            show_help
            ;;
        9)
            print_info "Exiting..."
            exit 0
            ;;
        *)
            print_error "Invalid option"
            show_menu
            ;;
    esac
}

# Function to show help
show_help() {
    print_header "AI Image Generator Help"
    echo ""
    echo "This launcher manages your complete AI image generation system:"
    echo ""
    echo "📁 Directory Structure:"
    echo "  backend/           - Flask API server"
    echo "  frontend/          - Next.js web interface"
    echo "  lora_weights/      - Custom trained LoRA models"
    echo "  generated_images/  - Generated images"
    echo "  training_logs/     - Training history"
    echo ""
    echo "🔧 Key Features:"
    echo "  ✅ Mac M3 GPU acceleration (MPS)"
    echo "  ✅ Custom LoRA training on LAION-2B"
    echo "  ✅ Web interface for easy generation"
    echo "  ✅ API endpoints for integration"
    echo "  ✅ Real-time image generation"
    echo ""
    echo "🌐 Web Interface:"
    echo "  Frontend: http://localhost:3000"
    echo "  Backend API: http://localhost:5001"
    echo ""
    echo "📚 Documentation:"
    echo "  Training Guide: MPS_TRAINING_GUIDE.md"
    echo "  Quick Start: QUICK_START.md"
    echo ""
}

# Main execution
main() {
    # Check if running interactively
    if [ -t 0 ]; then
        show_menu
    else
        # Non-interactive mode - start full system
        check_system
        activate_venv
        check_mps
        check_hf_auth
        start_backend
        start_frontend
        test_generation
        show_status
        
        print_header "System Ready!"
        echo ""
        print_info "Your AI Image Generator is now running:"
        echo "  🌐 Web Interface: http://localhost:3000"
        echo "  🔌 API Endpoint: http://localhost:5001"
        echo ""
        print_info "Press Ctrl+C to stop all services"
        
        # Keep script running
        wait
    fi
}

# Run main function
main "$@" 