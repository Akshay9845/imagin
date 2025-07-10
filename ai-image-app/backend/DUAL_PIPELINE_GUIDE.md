# Dual Pipeline Video Generation Guide

## Overview

This guide explains how to use both video generation pipeline options effectively:

- **Option A: Image → Motion → Video** (Best for control)
- **Option B: Text to Video Direct** (Best for creativity)

## Pipeline Options

### Option A: Image → Motion → Video (Best for Control)

**Workflow:**
1. Generate base image with RealVisXL
2. Animate with AnimateDiff + Motion LoRA
3. Upscale with Zeroscope
4. Interpolate with RIFE for smooth motion

**Best For:**
- Professional portraits
- Product demonstrations
- Realistic landscapes
- Controlled motion sequences
- High-quality, predictable results

**Advantages:**
- Superior control over motion
- Better realism and consistency
- Predictable results
- Professional quality output

**Example Use Cases:**
```python
# Professional portrait with controlled motion
base_image = image_system.generate_ultra_realistic_image(
    prompt="A professional portrait of a confident business person",
    style="photorealistic",
    width=512,
    height=512
)

video_path = video_system.generate_video_from_image_pipeline(
    image=base_image,
    motion_prompt="gentle zoom",  # Controlled motion
    duration_seconds=5,
    fps=24,
    pipeline_type="stable_video"
)
```

### Option B: Text to Video Direct (Best for Creativity)

**Workflow:**
1. Use VideoCrafter2 for direct generation
2. Use ModelScope T2V as lightweight fallback

**Best For:**
- Creative animations
- Artistic scenes
- Dynamic content
- Imaginative storytelling
- Fast prototyping

**Advantages:**
- More creative freedom
- Faster generation
- Better for artistic content
- Dynamic motion

**Example Use Cases:**
```python
# Creative animation with dynamic motion
video_path = video_system.generate_direct_text_to_video(
    prompt="A magical forest with glowing butterflies and floating particles",
    duration_seconds=5,
    fps=24,
    width=512,
    height=512,
    pipeline_type="modelscope_t2v"
)
```

## When to Use Each Pipeline

### Use Option A (Image → Video) When:
- ✅ You need controlled, realistic motion
- ✅ Working with professional content
- ✅ Quality and consistency are priority
- ✅ You have a specific image as reference
- ✅ Motion should be subtle and natural

### Use Option B (Direct) When:
- ✅ You want creative, dynamic content
- ✅ Speed is important
- ✅ Artistic or stylized results needed
- ✅ Complex motion sequences required
- ✅ Prototyping or experimentation

## Hybrid Approach: Using Both Together

For maximum flexibility and quality, you can use both pipelines together:

```python
def generate_hybrid_video(prompt, style="photorealistic"):
    """Generate video using both pipelines and choose the best result"""
    
    results = {}
    
    # Option A: Image → Video (for control)
    try:
        base_image = image_system.generate_ultra_realistic_image(
            prompt=prompt,
            style=style,
            width=512,
            height=512
        )
        
        video_a = video_system.generate_video_from_image_pipeline(
            image=base_image,
            motion_prompt="gentle movement",
            duration_seconds=5,
            fps=24,
            pipeline_type="stable_video"
        )
        results['option_a'] = video_a
    except Exception as e:
        print(f"Option A failed: {e}")
    
    # Option B: Direct (for creativity)
    try:
        video_b = video_system.generate_direct_text_to_video(
            prompt=prompt,
            duration_seconds=5,
            fps=24,
            width=512,
            height=512,
            pipeline_type="modelscope_t2v"
        )
        results['option_b'] = video_b
    except Exception as e:
        print(f"Option B failed: {e}")
    
    return results
```

## Practical Examples

### Example 1: Professional Content
```python
# Use Option A for professional portraits
prompt = "A confident business executive in a modern office"
video_path = generate_professional_video(prompt)

def generate_professional_video(prompt):
    # Generate high-quality base image
    base_image = image_system.generate_ultra_realistic_image(
        prompt=prompt,
        style="photorealistic",
        width=1024,
        height=1024,
        num_inference_steps=30,
        guidance_scale=7.5
    )
    
    # Create controlled motion video
    return video_system.generate_video_from_image_pipeline(
        image=base_image,
        motion_prompt="gentle zoom",
        duration_seconds=8,
        fps=30,
        width=1024,
        height=1024,
        pipeline_type="stable_video"
    )
```

### Example 2: Creative Content
```python
# Use Option B for creative animations
prompt = "A magical forest with glowing butterflies and floating particles"
video_path = generate_creative_video(prompt)

def generate_creative_video(prompt):
    return video_system.generate_direct_text_to_video(
        prompt=prompt,
        duration_seconds=10,
        fps=24,
        width=512,
        height=512,
        pipeline_type="modelscope_t2v"
    )
```

### Example 3: Batch Generation with Both Pipelines
```python
def batch_generate_with_both_pipelines(prompts):
    """Generate videos using both pipelines for each prompt"""
    
    results = []
    
    for prompt in prompts:
        prompt_results = {}
        
        # Try Option A first (for quality)
        try:
            base_image = image_system.generate_ultra_realistic_image(
                prompt=prompt,
                style="photorealistic",
                width=512,
                height=512
            )
            
            video_a = video_system.generate_video_from_image_pipeline(
                image=base_image,
                motion_prompt="gentle movement",
                duration_seconds=5,
                fps=24,
                pipeline_type="stable_video"
            )
            prompt_results['option_a'] = video_a
        except Exception as e:
            print(f"Option A failed for '{prompt}': {e}")
        
        # Try Option B as backup (for speed)
        try:
            video_b = video_system.generate_direct_text_to_video(
                prompt=prompt,
                duration_seconds=5,
                fps=24,
                width=512,
                height=512,
                pipeline_type="modelscope_t2v"
            )
            prompt_results['option_b'] = video_b
        except Exception as e:
            print(f"Option B failed for '{prompt}': {e}")
        
        results.append(prompt_results)
    
    return results
```

## Pipeline Selection Guide

| Use Case | Recommended Pipeline | Reason |
|----------|---------------------|---------|
| Professional Portrait | Option A | Better control and realism |
| Product Demo | Option A | Precise motion control |
| Creative Animation | Option B | More creative freedom |
| Artistic Scene | Option B | Better for stylized content |
| Realistic Landscape | Option A | Superior realism |
| Fast Prototyping | Option B | Faster generation |
| High-Quality Output | Option A | Better quality control |
| Dynamic Content | Option B | Better motion variety |

## Best Practices

### For Option A (Image → Video):
1. **Use high-quality base images** - The better the image, the better the video
2. **Choose appropriate motion** - Use "gentle" for professional content
3. **Optimize image generation** - Use more inference steps for better quality
4. **Consider aspect ratios** - Match video dimensions to image dimensions

### For Option B (Direct):
1. **Write detailed prompts** - More specific prompts yield better results
2. **Experiment with styles** - Try different artistic styles
3. **Use appropriate duration** - Longer videos may need more processing
4. **Consider motion keywords** - Include motion descriptions in prompts

### For Both Pipelines:
1. **Start with smaller sizes** - Test with 512x512 before going larger
2. **Use appropriate FPS** - 24fps for most content, 30fps for smooth motion
3. **Monitor system resources** - Video generation is resource-intensive
4. **Save intermediate results** - Keep base images for Option A

## Troubleshooting

### Common Issues and Solutions:

**Option A Fails:**
- Check if base image generation works
- Verify video models are loaded
- Try different motion prompts
- Reduce image size for testing

**Option B Fails:**
- Check if direct video models are available
- Try different pipeline types
- Simplify the prompt
- Check system memory

**Both Pipelines Fail:**
- Verify all dependencies are installed
- Check GPU memory availability
- Try with smaller dimensions
- Restart the system

## Running Examples

### Command Line:
```bash
# Run dual pipeline example
python dual_pipeline_example.py

# Run simple both pipelines example
python use_both_pipelines.py

# Use launcher with dual pipeline option
python launch_video_system.py
```

### API Usage:
```python
# Option A: Image to Video
response = requests.post("http://localhost:5003/api/ultra-realistic-video/generate-from-image", {
    "image": base_image_path,
    "motion_prompt": "gentle zoom",
    "duration_seconds": 5,
    "fps": 24
})

# Option B: Direct Video
response = requests.post("http://localhost:5003/api/ultra-realistic-video/generate-direct", {
    "prompt": "A magical forest scene",
    "duration_seconds": 5,
    "fps": 24
})
```

## Conclusion

Both pipeline options offer unique advantages:

- **Option A** excels at controlled, realistic motion with professional quality
- **Option B** provides creative freedom and faster generation for artistic content
- **Using both together** gives you maximum flexibility and quality

Choose the right pipeline based on your specific use case, or use both together for the best results. The system is designed to support both approaches seamlessly, allowing you to create the perfect video for any project. 