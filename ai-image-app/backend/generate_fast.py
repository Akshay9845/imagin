from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
import os

# Use a smaller, faster model for testing
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    safety_checker=None,  # Disable safety checker for speed
    requires_safety_checker=False
).to("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Load custom LoRA weights
try:
    print("🔗 Loading custom LoRA weights...")
    pipe.load_lora_weights("lora_weights/lora_step_50_20250707_231755")
    print("✅ Custom LoRA loaded successfully!")
except Exception as e:
    print(f"⚠️ Failed to load LoRA: {e}")
    print("📝 Using base model only")

# Set a more stable scheduler
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Enable memory efficient attention if available
if hasattr(pipe, "enable_attention_slicing"):
    pipe.enable_attention_slicing()

os.makedirs("static", exist_ok=True)

def generate_image(prompt):
    """Generate image with faster settings"""
    try:
        # Use fewer inference steps for speed
        image = pipe(
            prompt,
            num_inference_steps=20,  # Reduced from default 50
            guidance_scale=7.5
        ).images[0]
        
        filename = f"{prompt.replace(' ', '_')[:30]}.png"
        output_path = f"static/{filename}"
        image.save(output_path)
        return f"static/{filename}"
    except Exception as e:
        print(f"Error generating image: {e}")
        raise e 