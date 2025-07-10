# 🚀 Quick Start Guide

Your AI Image Generator with LAION-2B-en streaming training is ready!

## ⚡ Start Everything (Easiest Way)

```bash
./start_servers.sh
```

This starts both backend and frontend automatically.

## 🔧 Manual Start

### Backend (Flask API)
```bash
cd backend
source venv/bin/activate
python run_server.py
```
→ Runs on http://localhost:5000

### Frontend (Next.js UI)
```bash
cd frontend
npm run dev
```
→ Runs on http://localhost:3000

## 🎯 Use the System

1. **Open your browser** to http://localhost:3000
2. **Enter a prompt** like "a beautiful sunset over mountains"
3. **Click Generate** and watch your image appear!

## 🧠 Train Custom LoRA

```bash
cd backend
source venv/bin/activate
python train_lora_stream.py
```

This will:
- Stream 5,000 images from LAION-2B-en
- Train a LoRA adapter in memory
- Save weights to `lora_weights/` directory
- Discard images after training (memory efficient)

## 🧪 Test Everything

```bash
python test_system.py
```

This verifies both servers are working correctly.

## 📁 What You Have

- ✅ **Flask Backend** with image generation API
- ✅ **Next.js Frontend** with modern UI
- ✅ **LAION-2B-en Streaming Trainer** (use-and-discard)
- ✅ **LoRA Training** with progressive rounds
- ✅ **CORS Support** for frontend-backend communication
- ✅ **Error Handling** and logging

## 🎨 Next Steps

1. **Generate some images** to test the system
2. **Start LoRA training** to create custom models
3. **Deploy to Vercel/Render** when ready
4. **Add video generation** with SVD/AnimateDiff

---

**You're all set! Your custom AI image generator is ready to use! 🎉** 