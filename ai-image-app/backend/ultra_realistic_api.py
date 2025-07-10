#!/usr/bin/env python3
"""
Ultra-Realistic Generation API
Flask API for the ultra-realistic image and video generation system
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import logging
from pathlib import Path
from datetime import datetime
import json
from ultra_realistic_system import UltraRealisticSystem
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize the ultra-realistic system
try:
    ultra_system = UltraRealisticSystem()
    logger.info("✅ Ultra-realistic system initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize ultra-realistic system: {e}")
    ultra_system = None

@app.route('/api/ultra-realistic/generate-image', methods=['POST'])
def generate_ultra_realistic_image():
    """Generate ultra-realistic image"""
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Prompt is required'}), 400
        
        prompt = data['prompt']
        style = data.get('style', 'photorealistic')
        width = data.get('width', 1024)
        height = data.get('height', 1024)
        steps = data.get('steps', 50)
        guidance = data.get('guidance', 7.5)
        
        if not ultra_system:
            return jsonify({'error': 'Ultra-realistic system not available'}), 500
        
        logger.info(f"🎨 Generating ultra-realistic image: {prompt[:50]}...")
        
        # Generate image
        image = ultra_system.generate_ultra_realistic_image(
            prompt=prompt,
            style=style,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance
        )
        
        # Save to static directory for web access
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ultra_realistic_{style}_{timestamp}.png"
        static_path = Path("static") / filename
        image.save(static_path)
        
        return jsonify({
            'success': True,
            'image_url': f'/static/{filename}',
            'filename': filename,
            'style': style,
            'prompt': prompt
        })
        
    except Exception as e:
        logger.error(f"❌ Error generating image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ultra-realistic/generate-video', methods=['POST'])
def generate_ultra_realistic_video():
    """Generate video from image"""
    try:
        data = request.get_json()
        
        if not data or 'image_path' not in data:
            return jsonify({'error': 'Image path is required'}), 400
        
        image_path = data['image_path']
        motion_prompt = data.get('motion_prompt', 'gentle movement')
        duration = data.get('duration', 8)
        fps = data.get('fps', 24)
        
        if not ultra_system:
            return jsonify({'error': 'Ultra-realistic system not available'}), 500
        
        logger.info(f"🎬 Generating video from image: {motion_prompt}")
        
        # Generate video
        video_path = ultra_system.generate_video_from_image(
            image_path=image_path,
            motion_prompt=motion_prompt,
            duration=duration,
            fps=fps
        )
        
        if video_path:
            # Copy to static directory for web access
            import shutil
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ultra_video_{timestamp}.mp4"
            static_path = Path("static") / filename
            shutil.copy2(video_path, static_path)
            
            return jsonify({
                'success': True,
                'video_url': f'/static/{filename}',
                'filename': filename,
                'motion_prompt': motion_prompt,
                'duration': duration,
                'fps': fps
            })
        else:
            return jsonify({'error': 'Failed to generate video'}), 500
        
    except Exception as e:
        logger.error(f"❌ Error generating video: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ultra-realistic/batch-generate', methods=['POST'])
def batch_generate_images():
    """Generate multiple images in batch"""
    try:
        data = request.get_json()
        
        if not data or 'prompts' not in data:
            return jsonify({'error': 'Prompts list is required'}), 400
        
        prompts = data['prompts']
        style = data.get('style', 'photorealistic')
        
        if not ultra_system:
            return jsonify({'error': 'Ultra-realistic system not available'}), 500
        
        logger.info(f"🔄 Batch generating {len(prompts)} images...")
        
        # Generate images
        images = ultra_system.batch_generate(prompts, style)
        
        # Save to static directory
        image_urls = []
        for i, image in enumerate(images):
            if image:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"batch_{i}_{style}_{timestamp}.png"
                static_path = Path("static") / filename
                image.save(static_path)
                image_urls.append(f'/static/{filename}')
        
        return jsonify({
            'success': True,
            'image_urls': image_urls,
            'count': len(image_urls),
            'style': style
        })
        
    except Exception as e:
        logger.error(f"❌ Error in batch generation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ultra-realistic/styles', methods=['GET'])
def get_available_styles():
    """Get available generation styles"""
    styles = [
        {
            'id': 'photorealistic',
            'name': 'Photorealistic',
            'description': 'Ultra-realistic photos with accurate lighting and details',
            'best_for': 'Portraits, landscapes, product photography'
        },
        {
            'id': 'artistic',
            'name': 'Artistic Realistic',
            'description': 'Realistic images with artistic flair and cinematic lighting',
            'best_for': 'Creative portraits, artistic scenes, concept art'
        },
        {
            'id': 'cinematic',
            'name': 'Cinematic',
            'description': 'Movie-like quality with dramatic lighting and composition',
            'best_for': 'Film stills, dramatic scenes, storytelling'
        },
        {
            'id': 'anime_realistic',
            'name': 'Anime Realistic',
            'description': 'Anime style with realistic elements and high detail',
            'best_for': 'Anime characters, stylized scenes, illustrations'
        }
    ]
    
    return jsonify({
        'success': True,
        'styles': styles
    })

@app.route('/api/ultra-realistic/status', methods=['GET'])
def get_system_status():
    """Get system status and capabilities"""
    if not ultra_system:
        return jsonify({
            'success': False,
            'status': 'unavailable',
            'error': 'Ultra-realistic system not initialized'
        })
    
    try:
        # Get system info
        device = ultra_system.device
        available_models = list(ultra_system.pipelines.keys())
        
        return jsonify({
            'success': True,
            'status': 'ready',
            'device': device,
            'available_models': available_models,
            'output_directory': str(ultra_system.output_dir),
            'cuda_available': torch.cuda.is_available() if hasattr(torch, 'cuda') else False
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'status': 'error',
            'error': str(e)
        })

@app.route('/api/ultra-realistic/static/<filename>')
def serve_static_file(filename):
    """Serve generated files"""
    static_path = Path("static") / filename
    if static_path.exists():
        return send_file(static_path)
    else:
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    # Ensure static directory exists
    Path("static").mkdir(exist_ok=True)
    
    print("🚀 Ultra-Realistic Generation API")
    print("=" * 40)
    print("Available endpoints:")
    print("  POST /api/ultra-realistic/generate-image")
    print("  POST /api/ultra-realistic/generate-video")
    print("  POST /api/ultra-realistic/batch-generate")
    print("  GET  /api/ultra-realistic/styles")
    print("  GET  /api/ultra-realistic/status")
    print("=" * 40)
    
    app.run(debug=True, host='0.0.0.0', port=5001) 