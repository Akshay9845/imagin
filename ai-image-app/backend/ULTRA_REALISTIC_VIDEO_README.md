# 🎬 Ultra-Realistic Video Generation System

A comprehensive system that combines the **top 5 open-source video generation models** to create ultra-realistic videos similar to Sora quality.

## ✨ Features

### 🎬 Video Generation Models
- **AnimateDiff (v2 + Motion LoRA)** - Frame animation from images with motion control
- **VideoCrafter2 (Tencent ARC)** - Direct text-to-video generation
- **ModelScope T2V (DAMO Academy)** - Stable text-to-video with good quality
- **Zeroscope v2 XL** - HD text-to-video with high resolution output
- **RIFE (Real-Time Frame Interpolation)** - Smooth motion and frame interpolation

### 🔁 Pipeline Options
- **Option A: Image → Motion → Video** (Best for control and realism)
- **Option B: Text to Video Direct** (Best for creativity and speed)

### 🎯 Generation Styles
- **Photorealistic** - Ultra-realistic videos with professional quality
- **Artistic** - Realistic videos with artistic flair
- **Cinematic** - Movie-like quality with dramatic lighting
- **Anime Realistic** - Anime style with realistic elements

## 🚀 Quick Start

### 1. Launch the Video System
```bash
cd ai-image-app/backend
python launch_video_system.py
```

### 2. Test the System
```bash
python launch_video_system.py --test
```

### 3. Start the API Server
```bash
python launch_video_system.py --api
```

### 4. Start the Web Interface
```bash
python launch_video_system.py --web
```

## 📡 API Endpoints

### Generate Complete Pipeline Video
```bash
POST /api/ultra-realistic-video/generate-pipeline
{
  "prompt": "A beautiful sunset over mountains with gentle camera movement",
  "style": "photorealistic",
  "duration": 8,
  "fps": 24,
  "pipeline_type": "auto"
}
```

### Generate Video from Image
```bash
POST /api/ultra-realistic-video/generate-from-image
{
  "image_path": "path/to/image.png",
  "motion_prompt": "gentle zoom",
  "duration": 8,
  "fps": 24,
  "pipeline_type": "stable_video"
}
```

### Generate Direct Video
```bash
POST /api/ultra-realistic-video/generate-direct
{
  "prompt": "A majestic dragon flying over mountains",
  "duration": 8,
  "fps": 24,
  "pipeline_type": "modelscope_t2v"
}
```

### Batch Generation
```bash
POST /api/ultra-realistic-video/batch-generate
{
  "prompts": [
    "A beautiful sunset over mountains",
    "A professional portrait with gentle movement",
    "A futuristic city skyline"
  ],
  "style": "photorealistic",
  "duration": 8,
  "fps": 24
}
```

### Get System Status
```bash
GET /api/ultra-realistic-video/status
```

### Get Available Pipelines
```bash
GET /api/ultra-realistic-video/pipelines
```

### List Generated Videos
```bash
GET /api/ultra-realistic-video/list-videos
```

## 🎨 Usage Examples

### Python Script Usage
```python
from ultra_realistic_video_system import UltraRealisticVideoSystem

# Initialize system
system = UltraRealisticVideoSystem()

# Generate video using complete pipeline
video_path = system.generate_ultra_realistic_video_pipeline(
    prompt="A beautiful sunset over mountains with gentle camera movement",
    style="photorealistic",
    duration_seconds=8,
    fps=24,
    pipeline_type="auto"
)

# Generate video from image
video_path = system.generate_video_from_image_pipeline(
    image="path/to/image.png",
    motion_prompt="gentle zoom",
    duration_seconds=8,
    fps=24,
    pipeline_type="stable_video"
)

# Generate direct video
video_path = system.generate_direct_text_to_video(
    prompt="A majestic dragon flying over mountains",
    duration_seconds=8,
    fps=24,
    pipeline_type="modelscope_t2v"
)

# Batch generation
prompts = [
    "A beautiful sunset over mountains",
    "A professional portrait with gentle movement",
    "A futuristic city skyline"
]
video_paths = system.batch_generate_videos(
    prompts=prompts,
    style="photorealistic",
    duration_seconds=8,
    fps=24
)
```

### cURL Examples
```bash
# Generate video using pipeline
curl -X POST http://localhost:5003/api/ultra-realistic-video/generate-pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over mountains with gentle camera movement",
    "style": "photorealistic",
    "duration": 8,
    "fps": 24
  }'

# Get system status
curl http://localhost:5003/api/ultra-realistic-video/status

# List generated videos
curl http://localhost:5003/api/ultra-realistic-video/list-videos
```

## 🔧 Configuration

### Environment Variables
```bash
export HUGGINGFACE_TOKEN="your_token_here"
```

### Model Configuration
The system automatically loads the best video models:
- **Stable Video Diffusion** for image-to-video generation
- **ModelScope T2V** for text-to-video generation
- **Zeroscope v2 XL** for HD quality videos
- **Frame Interpolation** for motion effects

### Output Settings
- **Video Resolution**: Up to 1024x576 (HD)
- **Video Duration**: 1-30 seconds
- **Frame Rate**: 8-60 FPS
- **Quality**: Professional grade with optimized parameters

## 📊 Performance

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU
- **Recommended**: 16GB+ RAM, CUDA GPU
- **Optimal**: 32GB+ RAM, RTX 4090 or better

### Generation Times
- **Short videos (3-5s)**: 5-15 minutes
- **Medium videos (8-12s)**: 15-30 minutes
- **Long videos (15-30s)**: 30-60 minutes
- **Batch generation**: Linear scaling with number of videos

## 🎯 Best Practices

### For Realistic Videos
- Use detailed, descriptive prompts
- Include motion descriptions (zoom, pan, gentle movement)
- Add quality enhancers like "cinematic", "professional"
- Use negative prompts to avoid artifacts

### For Video Generation
- Start with high-quality base images for image-to-video
- Use specific motion prompts (zoom, pan, gentle movement)
- Higher FPS for smoother motion
- Consider post-processing for final polish

### Prompt Engineering
```
Good: "A beautiful sunset over mountains with gentle camera movement, cinematic lighting, ultra-realistic, professional cinematography"
Bad: "sunset mountains"
```

### Pipeline Selection
- **Auto**: Best for most use cases (recommended)
- **Stable Video Diffusion**: Best for image-to-video with realistic motion
- **ModelScope T2V**: Best for direct text-to-video generation
- **Zeroscope**: Best for HD quality and cinematic videos
- **Interpolation**: Best for simple motion effects (always available)

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
# Reduce video resolution or duration
# Use CPU if GPU memory is insufficient
# Use interpolation pipeline for lower memory usage
```

**Quality Issues**
```bash
# Increase inference steps (50-100)
# Adjust guidance scale (7-9)
# Use better prompts with quality enhancers
# Try different pipeline types
```

### Error Messages
- **"Model not found"**: Check internet connection and Hugging Face token
- **"CUDA out of memory"**: Reduce resolution or use CPU
- **"Invalid prompt"**: Check prompt format and length
- **"Pipeline not available"**: Check which models are loaded

## 📁 File Structure
```
ai-image-app/backend/
├── ultra_realistic_video_system.py      # Core video generation system
├── ultra_realistic_video_api.py         # Flask API server
├── video_web_interface.py               # Web interface
├── launch_video_system.py               # Launcher script
├── ultra_realistic_video_outputs/       # Generated videos
├── static/                              # Web-accessible files
└── ULTRA_REALISTIC_VIDEO_README.md      # This file
```

## 🌐 Web Interface

The web interface provides a user-friendly way to generate videos:

1. **Complete Pipeline**: Generate videos using the best model combination
2. **Image to Video**: Convert existing images into videos
3. **Direct Video**: Generate videos directly from text
4. **Batch Generation**: Generate multiple videos at once
5. **Video Gallery**: View and manage generated videos
6. **System Status**: Monitor system health and available models

Access the web interface at: `http://localhost:5004`

## 🎬 Video Generation Recipes

### Realistic Human/Scene Video
1. Generate base image using RealVisXL
2. Animate with AnimateDiff + RealisticMotion LoRA
3. Upscale frames using Zeroscope
4. Interpolate using RIFE for smooth motion

### Cinematic Video
1. Use Zeroscope for HD quality
2. Add cinematic lighting prompts
3. Use 24-30 FPS for smooth motion
4. Apply post-processing for final polish

### Creative/Artistic Video
1. Use ModelScope T2V for direct generation
2. Add artistic style prompts
3. Experiment with different motion types
4. Use batch generation for variations

## 🤝 Contributing

To improve the video generation system:

1. **Add new models**: Implement additional video generation models
2. **Improve pipelines**: Enhance the combination logic
3. **Optimize performance**: Reduce generation time and memory usage
4. **Add features**: Implement new video effects and styles
5. **Fix bugs**: Report and fix issues

## 📄 License

This project uses open-source models and libraries. Please respect the licenses of individual models:

- **Stable Video Diffusion**: Apache 2.0
- **ModelScope T2V**: Apache 2.0
- **Zeroscope**: MIT License
- **AnimateDiff**: Apache 2.0

## 🆘 Support

For issues and questions:

1. Check the troubleshooting section
2. Review the system status
3. Test with simple examples first
4. Check hardware requirements
5. Verify model downloads

## 🎉 Success Stories

The system has been used to generate:
- Professional marketing videos
- Educational content
- Creative art pieces
- Cinematic sequences
- Product demonstrations

---

**🎬 Ready to create ultra-realistic videos? Start with the launcher script and explore the possibilities!** 