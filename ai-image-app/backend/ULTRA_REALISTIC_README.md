# 🚀 Ultra-Realistic Image & Video Generation System

A comprehensive system that combines the best open-source AI models to generate ultra-realistic images and videos, similar to Sora quality.

## ✨ Features

### 🎨 Image Generation
- **RealVisXL V4.0** - Best photorealism with accurate lighting
- **DreamShaper XL** - Artistic realism with cinematic feel
- **Juggernaut XL v9** - High-contrast realism with dynamic lighting
- **SDXL Base 1.0** - Foundation model for all SDXL operations
- **Anything V5** - Anime + realism blend

### 🎬 Video Generation
- **AnimateDiff-like** frame interpolation
- **Motion control** with customizable prompts
- **HD upscaling** capabilities
- **Smooth frame interpolation** for fluid motion

### 🎯 Generation Styles
- **Photorealistic** - Ultra-realistic photos with professional quality
- **Artistic** - Realistic images with artistic flair
- **Cinematic** - Movie-like quality with dramatic lighting
- **Anime Realistic** - Anime style with realistic elements

## 🚀 Quick Start

### 1. Launch the System
```bash
cd ai-image-app/backend
./launch_ultra_realistic.sh
```

### 2. Test the System
```bash
python test_ultra_realistic.py
```

### 3. Start the API Server
```bash
python ultra_realistic_api.py
```

## 📡 API Endpoints

### Generate Image
```bash
POST /api/ultra-realistic/generate-image
{
  "prompt": "A professional portrait of a confident business person",
  "style": "photorealistic",
  "width": 1024,
  "height": 1024,
  "steps": 50,
  "guidance": 7.5
}
```

### Generate Video
```bash
POST /api/ultra-realistic/generate-video
{
  "image_path": "path/to/image.png",
  "motion_prompt": "gentle zoom",
  "duration": 8,
  "fps": 24
}
```

### Batch Generation
```bash
POST /api/ultra-realistic/batch-generate
{
  "prompts": [
    "A majestic lion in the savanna",
    "A futuristic cityscape at night",
    "A serene lake at sunset"
  ],
  "style": "photorealistic"
}
```

### Get Available Styles
```bash
GET /api/ultra-realistic/styles
```

### System Status
```bash
GET /api/ultra-realistic/status
```

## 🎨 Usage Examples

### Python Script Usage
```python
from ultra_realistic_system import UltraRealisticSystem

# Initialize system
system = UltraRealisticSystem()

# Generate photorealistic image
image = system.generate_ultra_realistic_image(
    prompt="A beautiful sunset over mountains",
    style="photorealistic",
    width=1024,
    height=1024
)

# Generate video from image
video_path = system.generate_video_from_image(
    image_path="image.png",
    motion_prompt="gentle zoom",
    duration=8,
    fps=24
)

# Batch generation
prompts = [
    "A professional portrait",
    "A majestic landscape",
    "A futuristic city"
]
images = system.batch_generate(prompts, style="photorealistic")
```

### cURL Examples
```bash
# Generate image
curl -X POST http://localhost:5001/api/ultra-realistic/generate-image \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A professional portrait of a confident business person",
    "style": "photorealistic",
    "width": 1024,
    "height": 1024
  }'

# Get system status
curl http://localhost:5001/api/ultra-realistic/status
```

## 🔧 Configuration

### Environment Variables
```bash
export HUGGINGFACE_TOKEN="your_token_here"
```

### Model Configuration
The system automatically loads the best models:
- **RealVisXL** for photorealistic images
- **DreamShaper** for artistic realism
- **Juggernaut** for cinematic quality
- **SDXL Base** as foundation
- **Anything V5** for anime-realistic blend

### Output Settings
- **Image Resolution**: Up to 1024x1024 (SDXL) or 512x512 (SD 1.5)
- **Video Duration**: 1-30 seconds
- **Frame Rate**: 8-60 FPS
- **Quality**: Professional grade with optimized parameters

## 📊 Performance

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU
- **Recommended**: 16GB+ RAM, CUDA GPU
- **Optimal**: 32GB+ RAM, RTX 4090 or better

### Generation Times
- **Images**: 30-120 seconds (depending on resolution and steps)
- **Videos**: 2-10 minutes (depending on duration and quality)
- **Batch**: Linear scaling with number of images

## 🎯 Best Practices

### For Photorealistic Images
- Use detailed, descriptive prompts
- Include lighting and camera specifications
- Add quality enhancers like "8k uhd", "professional photography"
- Use negative prompts to avoid artifacts

### For Video Generation
- Start with high-quality base images
- Use specific motion prompts (zoom, pan, gentle movement)
- Higher FPS for smoother motion
- Consider post-processing for final polish

### Prompt Engineering
```
Good: "A professional portrait of a confident business person in a modern office, natural lighting, sharp focus, 8k uhd, dslr"
Bad: "person in office"
```

## 🔍 Troubleshooting

### Common Issues

**Model Loading Errors**
```bash
# Check Hugging Face token
echo $HUGGINGFACE_TOKEN

# Reinstall dependencies
pip install --upgrade diffusers transformers accelerate
```

**Memory Issues**
```bash
# Reduce batch size or resolution
# Use CPU if GPU memory is insufficient
```

**Quality Issues**
```bash
# Increase inference steps (50-100)
# Adjust guidance scale (7-9)
# Use better prompts with quality enhancers
```

### Error Messages
- **"Model not found"**: Check internet connection and Hugging Face token
- **"CUDA out of memory"**: Reduce resolution or use CPU
- **"Invalid prompt"**: Check prompt format and length

## 📁 File Structure
```
ai-image-app/backend/
├── ultra_realistic_system.py      # Core generation system
├── ultra_realistic_api.py         # Flask API server
├── launch_ultra_realistic.sh      # Launcher script
├── test_ultra_realistic.py        # Test suite
├── ultra_realistic_outputs/       # Generated content
├── static/                        # Web-accessible files
└── ULTRA_REALISTIC_README.md      # This file
```

## 🤝 Contributing

To improve the system:
1. Test with different model combinations
2. Optimize generation parameters
3. Add new video generation techniques
4. Enhance prompt engineering

## 📄 License

This system uses open-source models and is provided as-is for educational and research purposes.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review error logs in the console
3. Test with the provided test script
4. Verify system requirements

---

**🎉 Enjoy generating ultra-realistic content!** 