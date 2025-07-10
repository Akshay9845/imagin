from diffusers import StableDiffusionXLPipeline
import torch
import os

# Load LoRA weights you trained (placeholder, update path as needed)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V3.0",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("static", exist_ok=True)

def generate_image(prompt):
    image = pipe(prompt).images[0]
    filename = f"{prompt.replace(' ', '_')}.png"
    output_path = f"static/{filename}"
    image.save(output_path)
    return f"static/{filename}" 