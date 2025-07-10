# 🚀 Complete AI Image Generator System

**Your Mac M3-powered custom AI image generation system with LoRA training, web interface, and production deployment.**

## 🎉 What You Have

✅ **Custom LoRA Model** - Trained on LAION-2B dataset  
✅ **Mac M3 GPU Acceleration** - MPS backend for fast training  
✅ **Web Interface** - Next.js frontend + Flask backend  
✅ **Production Ready** - Complete API with multiple endpoints  
✅ **Training Pipeline** - MPS-compatible LoRA training  

## 📁 System Structure

```
ai-image-app/
├── backend/                    # Flask API server
│   ├── app.py                 # Main Flask application
│   ├── generate_fast.py       # Fast image generation
│   ├── generate_working.py    # Working generation script
│   ├── train_custom_model_mps.py  # MPS LoRA training
│   ├── train_lora_fixed.py    # Fixed LoRA training
│   ├── test_custom_model.py   # Model testing
│   ├── one_click_launcher.sh  # Complete system launcher
│   ├── launch_training.sh     # Training launcher
│   ├── training_config.json   # Training configuration
│   ├── lora_weights/          # Trained LoRA models
│   ├── generated_images/      # Generated images
│   └── training_logs/         # Training history
└── frontend/                  # Next.js web interface
    ├── app/                   # React components
    ├── package.json           # Dependencies
    └── next.config.ts         # Next.js config
```

## 🚀 Quick Start

### 1. One-Click Launch (Recommended)

```bash
cd ai-image-app/backend
chmod +x one_click_launcher.sh
./one_click_launcher.sh
```

This will:
- ✅ Check system requirements
- ✅ Activate virtual environment
- ✅ Start backend server (port 5001)
- ✅ Start frontend server (port 3000)
- ✅ Test image generation
- ✅ Show system status

### 2. Manual Launch

#### Start Backend Only
```bash
cd ai-image-app/backend
source venv/bin/activate
python app.py
```

#### Start Frontend Only
```bash
cd ai-image-app/frontend
npm run dev
```

#### Generate Images
```bash
cd ai-image-app/backend
source venv/bin/activate
python generate_working.py
```

## 🧠 Training Your Custom Model

### Option 1: Fixed LoRA Training (Recommended)

```bash
cd ai-image-app/backend
source venv/bin/activate
python train_lora_fixed.py
```

**Features:**
- ✅ Correct task type for SD compatibility
- ✅ MPS GPU acceleration
- ✅ LAION-2B dataset streaming
- ✅ Automatic checkpointing

### Option 2: MPS LoRA Training

```bash
cd ai-image-app/backend
./launch_training.sh
```

**Features:**
- ✅ Full MPS optimization
- ✅ Environment variable setup
- ✅ Comprehensive error checking

### Training Configuration

Edit `training_config.json` to customize:

```json
{
  "training": {
    "base_model": "runwayml/stable-diffusion-v1-5",
    "resolution": 512,
    "batch_size": 1,
    "max_train_steps": 5000,
    "learning_rate": 1e-4
  },
  "lora": {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
    "task_type": "CAUSAL_LM"
  },
  "datasets": [
    {
      "name": "laion/laion2B-en",
      "streaming": true,
      "filter": "caption_length_10_200"
    }
  ]
}
```

## 🎨 Image Generation

### Web Interface
Visit: **http://localhost:3000**

### API Endpoints

#### Generate with Base Model
```bash
curl -X POST http://localhost:5001/generate_fast \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A beautiful sunset over mountains"}'
```

#### Generate with Custom LoRA
```bash
curl -X POST http://localhost:5001/generate_custom \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A beautiful sunset over mountains"}'
```

#### Health Check
```bash
curl http://localhost:5001/health
```

### Command Line Generation

#### Working Generation (Base Model)
```bash
python generate_working.py
```

#### Custom LoRA Generation
```bash
python generate_with_custom_lora.py
```

#### Test Custom Model
```bash
python test_custom_model.py
```

## 📊 System Monitoring

### Check System Status
```bash
./one_click_launcher.sh
# Choose option 5: Show System Status
```

### Monitor Training
```bash
# Watch training logs
tail -f training_logs/training_info_*.json

# Check GPU usage (if available)
sudo powermetrics --samplers gpu_power -n 1
```

### Check Generated Images
```bash
ls -la generated_images/
ls -la lora_weights/
```

## 🔧 Troubleshooting

### Common Issues

#### 1. MPS Not Available
```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"
# Should return: True
```

#### 2. Hugging Face Authentication
```bash
huggingface-cli login
# Enter your token from: https://huggingface.co/settings/tokens
```

#### 3. Port Already in Use
```bash
# Stop existing services
pkill -f "python3 app.py"
pkill -f "npm run dev"

# Or change ports in app.py and next.config.ts
```

#### 4. Out of Memory
```bash
# Reduce batch size in training_config.json
"batch_size": 1,
"gradient_accumulation_steps": 8
```

### Performance Optimization

#### For Mac M3
- ✅ Keep batch_size=1 for stability
- ✅ Use 512x512 resolution for training
- ✅ Close other apps during training
- ✅ Keep Mac cool for best performance

#### For Better Results
- ✅ Train for 5000-10000 steps on LAION-2B
- ✅ Use learning rate 1e-4 to 5e-5
- ✅ LoRA rank 16-32 for quality
- ✅ Mix LAION-2B with custom datasets

## 🚀 Deployment Options

### 1. Local Production
```bash
# Backend with production settings
export FLASK_ENV=production
python app.py

# Frontend build
cd frontend
npm run build
npm start
```

### 2. Cloud Deployment

#### Backend (Flask)
- **Render.com**: Easy deployment with GPU support
- **Railway**: Simple container deployment
- **Heroku**: Traditional deployment (no GPU)

#### Frontend (Next.js)
- **Vercel**: Optimized for Next.js
- **Netlify**: Static site hosting
- **Cloudflare Pages**: Fast global CDN

#### GPU Backend
- **RunPod**: Pay-per-use GPU instances
- **Lambda Labs**: Affordable GPU cloud
- **Google Colab Pro**: Jupyter-based training

### 3. Hugging Face Spaces
```bash
# Create a Hugging Face Space for your model
# Upload your LoRA weights and create a demo
```

## 📈 Advanced Features

### Custom Datasets

Create your dataset structure:
```
your_dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── captions.txt
# Format: image1.jpg|A fantasy ruin glowing in moonlight
```

Update `training_config.json`:
```json
{
  "custom_datasets": [
    {
      "type": "imagefolder",
      "data_dir": "path/to/your/dataset",
      "description": "Your custom style"
    }
  ]
}
```

### SDXL Training

For SDXL models, update configuration:
```json
{
  "training": {
    "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
    "resolution": 1024,
    "max_train_steps": 3000
  },
  "lora": {
    "target_modules": [
      "down_blocks.*.attentions.*.transformer_blocks.*.attn1.*",
      "mid_block.attentions.*.transformer_blocks.*.attn1.*",
      "up_blocks.*.attentions.*.transformer_blocks.*.attn1.*"
    ]
  }
}
```

### HD Generation

Use the HD generation script for high-resolution images:
```bash
python generate_hd.py
```

## 🎯 Success Indicators

### Training Success
- ✅ `"device": "mps"` in training logs
- ✅ `"lora_success": true`
- ✅ Loss decreasing over time
- ✅ LoRA weights saved in `lora_weights/`

### Generation Success
- ✅ Images generated in `generated_images/`
- ✅ Web interface accessible at `http://localhost:3000`
- ✅ API endpoints responding
- ✅ No errors in console

### System Health
- ✅ Backend: `http://localhost:5001/health` returns 200
- ✅ Frontend: `http://localhost:3000` loads
- ✅ GPU: MPS available and working
- ✅ Memory: Sufficient RAM for generation

## 📚 Documentation

- **Training Guide**: `MPS_TRAINING_GUIDE.md`
- **Quick Start**: `QUICK_START.md`
- **API Docs**: Check Flask routes in `app.py`
- **Frontend Docs**: Next.js documentation

## 🆘 Support

### Quick Commands Reference
```bash
# Start everything
./one_click_launcher.sh

# Generate images
python generate_working.py

# Train LoRA
python train_lora_fixed.py

# Test system
python test_custom_model.py

# Stop services
pkill -f "python3 app.py" && pkill -f "npm run dev"

# Check status
curl http://localhost:5001/health
```

### Getting Help
1. Check logs: `tail -f training_logs/*.json`
2. Verify MPS: `python -c "import torch; print(torch.backends.mps.is_available())"`
3. Test generation: `python generate_working.py`
4. Check memory: Activity Monitor → Memory tab

## 🎉 Congratulations!

You now have a **complete, production-ready AI image generation system** running on your Mac M3 with:

- ✅ **Custom LoRA training** on LAION-2B
- ✅ **GPU acceleration** via MPS
- ✅ **Web interface** for easy use
- ✅ **API endpoints** for integration
- ✅ **Production deployment** options

**Your system is ready to create amazing AI-generated images!** 🚀 