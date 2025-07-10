# AI Image Generator with LAION-2B-en Streaming Training

A lightweight custom image generation system using Flask + Next.js, with streaming LoRA training on LAION-2B-en dataset.

## 🚀 Features

- **Streaming Training**: Train LoRA adapters on LAION-2B-en without downloading the full dataset
- **Use-and-Discard**: Process images in memory, save only LoRA weights (~100MB each)
- **Modern UI**: Next.js frontend with real-time image generation
- **Production Ready**: Flask backend with CORS support

## 📁 Project Structure

```
ai-image-app/
├── backend/
│   ├── app.py              # Flask API endpoints
│   ├── generate.py         # Image generation logic
│   ├── train_lora_stream.py # LAION streaming LoRA trainer
│   ├── run_server.py       # Server runner with CORS
│   └── venv/               # Python virtual environment
├── frontend/
│   ├── app/
│   │   └── page.tsx        # Main UI component
│   └── package.json        # Next.js dependencies
└── README.md
```

## 🛠️ Setup

### Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd ai-image-app/backend
   ```

2. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Run Flask server:**
   ```bash
   python run_server.py
   ```
   Server will start at `http://localhost:5000`

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd ai-image-app/frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Run Next.js development server:**
   ```bash
   npm run dev
   ```
   Frontend will start at `http://localhost:3000`

## 🎯 Usage

### Image Generation
1. Open `http://localhost:3000` in your browser
2. Enter a prompt in the text field
3. Click "Generate" to create an image
4. View the generated image below the form

### LoRA Training
1. **Start training on LAION-2B-en:**
   ```bash
   cd backend
   source venv/bin/activate
   python train_lora_stream.py
   ```

2. **Training process:**
   - Streams 5,000 images at a time from LAION-2B-en
   - Trains LoRA adapter in memory
   - Saves weights to `lora_weights/` directory
   - Discards images after training
   - Repeats with different seeds for progressive training

3. **Training outputs:**
   - LoRA weights: `lora_weights/lora_step_X_timestamp.safetensors`
   - Training logs: `training_logs/training_info_timestamp.json`

## 🔧 Configuration

### Training Parameters
Edit `train_lora_stream.py` to adjust:
- `batch_size`: Number of images per training batch (default: 1)
- `learning_rate`: Training learning rate (default: 1e-4)
- `max_train_steps`: Maximum training steps per round (default: 1000)
- `save_steps`: How often to save weights (default: 500)

### Model Selection
Change the base model in `generate.py` and `train_lora_stream.py`:
```python
base_model="SG161222/RealVisXL_V3.0"  # Current
# Alternatives:
# "stabilityai/stable-diffusion-xl-base-1.0"
# "runwayml/stable-diffusion-v1-5"
```

## 🚀 Deployment

### Backend (Render)
1. Push code to GitHub
2. Connect repository to Render
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `python run_server.py`

### Frontend (Vercel)
1. Push code to GitHub
2. Import repository in Vercel
3. Deploy automatically

## 📊 LAION-2B-en Streaming

The system uses LAION-2B-en dataset with these features:

- **Streaming**: No need to download 2.3B images
- **Chunked Processing**: 5K-10K images per training round
- **Quality Filtering**: Basic text quality checks
- **Memory Efficient**: Process and discard approach
- **Progressive Training**: Multiple rounds with different seeds

## 🔍 API Endpoints

- `POST /generate`: Generate image from prompt
  ```json
  {
    "prompt": "a beautiful sunset over mountains"
  }
  ```
- `GET /static/<filename>`: Serve generated images

## 🛡️ Error Handling

- Frontend shows error messages for failed requests
- Backend logs training progress and errors
- Graceful handling of missing images or invalid prompts

## 🔄 Next Steps

1. **Enhanced Training**: Implement full diffusion training loop
2. **Image Download**: Add URL image downloading for LAION
3. **Model Loading**: Load trained LoRA weights in generation
4. **Video Generation**: Add SVD/AnimateDiff support
5. **Deployment**: Deploy to Render/Vercel

## 📝 Notes

- First run will download the base model (~6GB)
- Training requires significant GPU memory
- Generated images are saved in `backend/static/`
- LoRA weights are saved in `backend/lora_weights/`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

**Built with ❤️ using Flask, Next.js, and LAION-2B-en** 