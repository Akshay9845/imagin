# 🚀 Setup Guide - AI Video Generation Platform

## 📋 Prerequisites

### System Requirements
- **OS**: macOS 10.15+ (with MPS support), Windows 10+ (with CUDA), or Linux
- **Python**: 3.9 or higher
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: At least 10GB free space for models
- **GPU**: Mac M1/M2 (MPS) or NVIDIA GPU (CUDA) recommended

### Software Requirements
- Python 3.9+
- pip (Python package manager)
- Git (for cloning the repository)

## 🔧 Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ai-video-generation-platform.git
cd ai-video-generation-platform
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
cd ai-image-app/backend
pip install -r requirements.txt
```

### 4. Set Up Hugging Face Token
```bash
# Install huggingface-hub if not already installed
pip install huggingface-hub

# Login to Hugging Face
huggingface-cli login
# Enter your token when prompted
```

**Get your Hugging Face token:**
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "read" permissions
3. Copy the token and use it when prompted

## 🎯 Running the System

### Option 1: Ultra-Realistic Video API (Recommended)
```bash
cd ai-image-app/backend
python3 ultra_realistic_video_api.py
```
- **Port**: 5007
- **Features**: Real AI video generation with multiple models
- **Best for**: Production use and high-quality videos

### Option 2: Simple Video API
```bash
cd ai-image-app/backend
python3 working_video_api.py
```
- **Port**: 5003
- **Features**: Motion effects and frame interpolation
- **Best for**: Quick motion videos from images

### Option 3: Web Interface
```bash
cd ai-image-app/backend
python3 working_video_web_interface.py
```
- **Port**: 5004
- **Features**: User-friendly web interface
- **Best for**: Easy-to-use GUI

## 🧪 Testing the Installation

### Test Memory Optimization
```bash
cd ai-image-app/backend
python3 test_memory_optimization.py
```

### Test API Endpoints
```bash
# Test health check
curl http://localhost:5007/api/ultra-realistic-video/health

# Test video generation
curl -X POST http://localhost:5007/api/ultra-realistic-video/generate-direct \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A beautiful sunset", "duration": 3, "fps": 8, "width": 256, "height": 256}'
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Memory Allocation Error
**Error**: `Invalid buffer size: 12.50 GB`
**Solution**: The system is already optimized for memory. Try reducing parameters:
```json
{
  "duration": 3,
  "fps": 8,
  "width": 256,
  "height": 256
}
```

#### 2. Model Loading Issues
**Error**: `Repository Not Found`
**Solution**: 
- Ensure you're logged into Hugging Face: `huggingface-cli login`
- Check your internet connection
- Try using a VPN if models are geo-restricted

#### 3. PyTorch Version Conflicts
**Error**: `torch.load` vulnerability warnings
**Solution**: Update PyTorch to version 2.6.0+:
```bash
pip install --upgrade torch torchvision
```

#### 4. Port Already in Use
**Error**: `Address already in use`
**Solution**: 
- Kill existing processes: `pkill -f python3`
- Or use different ports by modifying the code

### Performance Optimization

#### For Mac Users
- Enable MPS acceleration (automatic)
- Close other applications to free memory
- Use SSD storage for faster model loading

#### For Windows/Linux Users
- Install CUDA drivers for GPU acceleration
- Ensure PyTorch is installed with CUDA support
- Monitor GPU memory usage

## 📁 Project Structure

```
ai-video-generation-platform/
├── README.md                           # Main documentation
├── SETUP.md                           # This setup guide
├── requirements.txt                   # Python dependencies
├── LICENSE                           # MIT License
├── .gitignore                        # Git ignore rules
└── ai-image-app/
    ├── backend/
    │   ├── ultra_realistic_video_system.py    # Core AI system
    │   ├── ultra_realistic_video_api.py       # Main API server
    │   ├── working_video_api.py               # Simple video API
    │   ├── working_video_web_interface.py     # Web interface
    │   ├── test_memory_optimization.py        # Memory tests
    │   └── ultra_realistic_video_outputs/     # Generated videos
    └── frontend/                              # React UI (if applicable)
```

## 🎬 Usage Examples

### Generate a Nature Video
```bash
curl -X POST http://localhost:5007/api/ultra-realistic-video/generate-direct \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunrise over mountains with golden light, photorealistic",
    "duration": 5,
    "fps": 8,
    "width": 512,
    "height": 512,
    "pipeline_type": "modelscope_t2v"
  }'
```

### Generate a Sci-Fi Video
```bash
curl -X POST http://localhost:5007/api/ultra-realistic-video/generate-direct \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A futuristic robot dancing in a neon-lit cyberpunk city, glowing blue energy",
    "duration": 3,
    "fps": 8,
    "width": 256,
    "height": 256,
    "pipeline_type": "modelscope_t2v"
  }'
```

## 📊 System Monitoring

### Check System Status
```bash
# Check API health
curl http://localhost:5007/api/ultra-realistic-video/health

# List available pipelines
curl http://localhost:5007/api/ultra-realistic-video/pipelines

# Check system status
curl http://localhost:5007/api/ultra-realistic-video/status
```

### Monitor Generated Videos
```bash
# List generated videos
curl http://localhost:5007/api/ultra-realistic-video/list-videos

# Download a specific video
curl http://localhost:5007/api/ultra-realistic-video/download/filename.mp4
```

## 🚀 Next Steps

1. **Explore the Web Interface**: Visit http://localhost:5004
2. **Try Different Prompts**: Experiment with various video styles
3. **Adjust Parameters**: Modify duration, FPS, and resolution
4. **Check Generated Videos**: Look in `ultra_realistic_video_outputs/`
5. **Read the API Documentation**: Explore all available endpoints

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the logs in the terminal
3. Create an issue on GitHub
4. Check the project documentation

---

🎉 **Congratulations! Your AI Video Generation Platform is ready to use!** 