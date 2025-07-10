# 🎨 Custom AI Image Model Training & Deployment Guide

This guide will help you train your own custom AI image generation model using large datasets and deploy it to your web application.

## 🚀 Quick Start

### 1. Train Your Custom Model

```bash
cd ai-image-app/backend
python train_custom_model.py
```

### 2. Generate High-Resolution Images

```bash
python generate_hd.py
```

### 3. Deploy to Web App

```bash
python deploy_custom_model.py
```

## 📋 Complete Training Pipeline

### What You Get:

✅ **Streaming Dataset Support** - Train on LAION-2B-en without downloading  
✅ **LoRA Training** - Efficient fine-tuning with minimal memory usage  
✅ **Multiple Resolutions** - Support for 512x512 (SD v1.5) and 1024x1024 (SDXL)  
✅ **1080p Generation** - Upscale to full HD resolution  
✅ **Web Deployment** - Integrate with your Flask API  
✅ **Model Switching** - Use different models in your app  

---

## 🧠 Training Your Custom Model

### Configuration

Edit `training_config.json` to customize your training:

```json
{
  "training": {
    "base_model": "runwayml/stable-diffusion-v1-5",
    "use_sdxl": false,
    "resolution": 512,
    "batch_size": 1,
    "learning_rate": 1e-4,
    "num_epochs": 1,
    "max_train_steps": 1000
  },
  "datasets": [
    {
      "type": "huggingface",
      "name": "laion/laion2B-en",
      "streaming": true
    }
  ]
}
```

### Add Your Custom Dataset

1. **Create dataset folder:**
```
your_custom_dataset/
├── images/
│   ├── image1.jpg
│   └── image2.jpg
└── captions.txt
```

2. **Format captions.txt:**
```
image1.jpg|A beautiful sunset over mountains
image2.jpg|A futuristic city skyline at night
```

3. **Add to config:**
```json
{
  "type": "imagefolder",
  "data_dir": "path/to/your_custom_dataset"
}
```

### Start Training

```bash
# Basic training
python train_custom_model.py

# With custom config
python train_custom_model.py --config my_config.json
```

**Training Output:**
- Checkpoints saved every 500 steps
- Final deployable model in `custom_model_outputs/deployable_model/`
- Training logs in `custom_model_outputs/logs/`

---

## 🖼️ High-Resolution Generation

### Generate 1080p Images

```python
from generate_hd import HDImageGenerator

# Initialize generator
generator = HDImageGenerator(
    model_path="custom_model_outputs/deployable_model",
    use_sdxl=False,
    enable_upscaling=True
)

# Generate HD image
image = generator.generate_image(
    prompt="A cinematic portrait, 8k resolution, masterpiece",
    width=1024,
    height=1024,
    num_inference_steps=30,
    guidance_scale=7.5
)
```

### Batch Generation

```python
prompts = [
    "A magical forest with glowing mushrooms",
    "A cyberpunk city street at night",
    "A serene mountain landscape"
]

images = generator.generate_batch(
    prompts=prompts,
    output_dir="hd_outputs",
    width=1024,
    height=1024
)
```

---

## 🌐 Web Deployment

### Integrate with Flask App

1. **Load custom model:**
```python
from deploy_custom_model import integrate_with_flask

# Initialize deployer
deployer = integrate_with_flask()
```

2. **Add new API endpoints:**
```python
@app.route("/generate_custom", methods=["POST"])
def generate_custom():
    data = request.json
    image = deployer.generate_with_custom_model(
        prompt=data["prompt"],
        width=data.get("width", 512),
        height=data.get("height", 512)
    )
    # Save and return image
```

3. **Model switching:**
```python
@app.route("/switch_model", methods=["POST"])
def switch_model():
    model_name = request.json["model_name"]
    deployer.switch_model(model_name)
    return jsonify({"success": True})
```

### API Usage

**Generate with custom model:**
```bash
curl -X POST http://localhost:5001/generate_custom \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful landscape",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 30
  }'
```

**List available models:**
```bash
curl http://localhost:5001/models
```

**Switch models:**
```bash
curl -X POST http://localhost:5001/switch_model \
  -H "Content-Type: application/json" \
  -d '{"model_name": "custom_trained"}'
```

---

## 🔧 Advanced Configuration

### SDXL Training (1080p Native)

To train for native 1080p generation:

```json
{
  "training": {
    "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
    "use_sdxl": true,
    "resolution": 1024
  }
}
```

### Multiple Datasets

Combine multiple datasets:

```json
{
  "datasets": [
    {
      "type": "huggingface",
      "name": "laion/laion2B-en",
      "streaming": true
    },
    {
      "type": "imagefolder",
      "data_dir": "my_artwork_dataset"
    },
    {
      "type": "imagefolder", 
      "data_dir": "my_photography_dataset"
    }
  ]
}
```

### LoRA Configuration

Customize LoRA parameters:

```json
{
  "lora": {
    "r": 32,
    "lora_alpha": 64,
    "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
    "lora_dropout": 0.1
  }
}
```

---

## 📊 Monitoring & Optimization

### Training Monitoring

- **TensorBoard logs:** `custom_model_outputs/logs/`
- **Checkpoints:** Saved every 500 steps
- **Training info:** JSON files with metadata

### Memory Optimization

For limited GPU memory:

```json
{
  "training": {
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "mixed_precision": "fp16"
  },
  "advanced": {
    "gradient_checkpointing": true,
    "use_8bit_adam": true
  }
}
```

### Quality vs Speed

**Fast generation:**
```python
generator.generate_image(
    prompt="your prompt",
    num_inference_steps=20,
    guidance_scale=7.0
)
```

**High quality:**
```python
generator.generate_image(
    prompt="your prompt", 
    num_inference_steps=50,
    guidance_scale=8.5
)
```

---

## 🎯 Best Practices

### Dataset Preparation

1. **Quality over quantity** - Use high-quality, well-captioned images
2. **Consistent style** - Group similar styles together
3. **Diverse prompts** - Include various descriptions and styles
4. **Clean captions** - Remove typos and irrelevant text

### Training Tips

1. **Start small** - Begin with 1000 steps to test
2. **Monitor loss** - Watch for overfitting
3. **Save checkpoints** - Don't lose progress
4. **Test regularly** - Generate samples during training

### Deployment

1. **Test thoroughly** - Verify model works before deployment
2. **Monitor performance** - Track generation times and quality
3. **Backup models** - Keep multiple versions
4. **Document changes** - Track what each model was trained on

---

## 🚨 Troubleshooting

### Common Issues

**Out of Memory:**
- Reduce batch size
- Enable gradient checkpointing
- Use mixed precision

**Poor Quality:**
- Increase training steps
- Adjust learning rate
- Check dataset quality

**Model Not Loading:**
- Verify model path
- Check file permissions
- Ensure all dependencies installed

### Getting Help

1. Check training logs in `custom_model_outputs/logs/`
2. Verify configuration in `training_config.json`
3. Test with smaller datasets first
4. Monitor GPU memory usage

---

## 🎉 Next Steps

### What You Can Do Now:

1. **Train on your own data** - Add your custom dataset
2. **Experiment with SDXL** - Try 1080p native training
3. **Fine-tune for specific styles** - Train specialized models
4. **Deploy multiple models** - Switch between different trained models
5. **Build a model marketplace** - Share your trained models

### Advanced Features:

- **Textual Inversion** - Learn new concepts
- **DreamBooth** - Personalize models
- **ControlNet** - Add control over generation
- **Upscaling** - Generate even higher resolution

---

## 📚 Resources

- [Diffusers Documentation](https://huggingface.co/docs/diffusers/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [LAION Dataset](https://laion.ai/)
- [Stable Diffusion Guide](https://github.com/CompVis/stable-diffusion)

---

**🎨 Happy Training!** Your custom AI image generation model is just a few commands away! 