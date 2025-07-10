#!/usr/bin/env python3
"""
Ultra-Realistic Video Generation API
Flask API for the ultra-realistic video generation system
Integrates all top 5 open-source video models
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import logging
from pathlib import Path
from datetime import datetime
import json
import threading
import time
from ultra_realistic_video_system import UltraRealisticVideoSystem
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize the ultra-realistic video system
try:
    video_system = UltraRealisticVideoSystem()
    logger.info("✅ Ultra-realistic video system initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize ultra-realistic video system: {e}")
    video_system = None

# Global storage for generation status
generation_status = {}

@app.route('/api/ultra-realistic-video/status', methods=['GET'])
def get_video_system_status():
    """Get video system status and available pipelines"""
    try:
        if not video_system:
            return jsonify({
                'success': False,
                'error': 'Video system not available'
            }), 500
        
        # Get pipeline status
        pipeline_status = video_system.get_pipeline_status()
        available_pipelines = video_system.get_available_pipelines()
        
        return jsonify({
            'success': True,
            'device': video_system.device,
            'output_directory': str(video_system.output_dir),
            'pipeline_status': pipeline_status,
            'available_pipelines': available_pipelines,
            'loaded_models': list(video_system.video_pipelines.keys())
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting video system status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ultra-realistic-video/generate-from-image', methods=['POST'])
def generate_video_from_image():
    """Generate video from image using various pipelines"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Request data is required'}), 400
        
        # Required parameters
        if 'image_path' not in data:
            return jsonify({'error': 'Image path is required'}), 400
        
        image_path = data['image_path']
        motion_prompt = data.get('motion_prompt', 'gentle movement')
        duration_seconds = data.get('duration', 8)
        fps = data.get('fps', 24)
        width = data.get('width', 1024)
        height = data.get('height', 576)
        pipeline_type = data.get('pipeline_type', 'stable_video')
        
        if not video_system:
            return jsonify({'error': 'Video system not available'}), 500
        
        logger.info(f"🎬 Generating video from image: {motion_prompt}")
        
        # Generate video
        video_path = video_system.generate_video_from_image_pipeline(
            image=image_path,
            motion_prompt=motion_prompt,
            duration_seconds=duration_seconds,
            fps=fps,
            width=width,
            height=height,
            pipeline_type=pipeline_type
        )
        
        if video_path:
            # Extract filename for web access
            filename = Path(video_path).name
            
            return jsonify({
                'success': True,
                'video_path': video_path,
                'filename': filename,
                'motion_prompt': motion_prompt,
                'duration': duration_seconds,
                'fps': fps,
                'pipeline_type': pipeline_type
            })
        else:
            return jsonify({'error': 'Failed to generate video'}), 500
        
    except Exception as e:
        logger.error(f"❌ Error generating video from image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ultra-realistic-video/generate-direct', methods=['POST'])
def generate_direct_video():
    """Generate video directly from text prompt"""
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Prompt is required'}), 400
        
        prompt = data['prompt']
        duration_seconds = data.get('duration', 8)
        fps = data.get('fps', 24)
        width = data.get('width', 1024)
        height = data.get('height', 576)
        pipeline_type = data.get('pipeline_type', 'modelscope_t2v')
        
        if not video_system:
            return jsonify({'error': 'Video system not available'}), 500
        
        logger.info(f"🎬 Generating direct video: {prompt[:50]}...")
        
        # Generate video
        video_path = video_system.generate_direct_text_to_video(
            prompt=prompt,
            duration_seconds=duration_seconds,
            fps=fps,
            width=width,
            height=height,
            pipeline_type=pipeline_type
        )
        
        if video_path:
            filename = Path(video_path).name
            
            return jsonify({
                'success': True,
                'video_path': video_path,
                'filename': filename,
                'prompt': prompt,
                'duration': duration_seconds,
                'fps': fps,
                'pipeline_type': pipeline_type
            })
        else:
            return jsonify({'error': 'Failed to generate video'}), 500
        
    except Exception as e:
        logger.error(f"❌ Error generating direct video: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ultra-realistic-video/generate-pipeline', methods=['POST'])
def generate_ultra_realistic_video():
    """Generate ultra-realistic video using the complete pipeline"""
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Prompt is required'}), 400
        
        prompt = data['prompt']
        style = data.get('style', 'photorealistic')
        duration_seconds = data.get('duration', 8)
        fps = data.get('fps', 24)
        width = data.get('width', 1024)
        height = data.get('height', 576)
        pipeline_type = data.get('pipeline_type', 'auto')
        
        if not video_system:
            return jsonify({'error': 'Video system not available'}), 500
        
        logger.info(f"🎬 Starting ultra-realistic video pipeline: {prompt[:50]}...")
        
        # Generate video using the complete pipeline
        video_path = video_system.generate_ultra_realistic_video_pipeline(
            prompt=prompt,
            style=style,
            duration_seconds=duration_seconds,
            fps=fps,
            width=width,
            height=height,
            pipeline_type=pipeline_type
        )
        
        if video_path:
            filename = Path(video_path).name
            
            return jsonify({
                'success': True,
                'video_path': video_path,
                'filename': filename,
                'prompt': prompt,
                'style': style,
                'duration': duration_seconds,
                'fps': fps,
                'pipeline_type': pipeline_type
            })
        else:
            return jsonify({'error': 'Failed to generate video'}), 500
        
    except Exception as e:
        logger.error(f"❌ Error generating ultra-realistic video: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ultra-realistic-video/batch-generate', methods=['POST'])
def batch_generate_videos():
    """Generate multiple videos in batch"""
    try:
        data = request.get_json()
        
        if not data or 'prompts' not in data:
            return jsonify({'error': 'Prompts list is required'}), 400
        
        prompts = data['prompts']
        style = data.get('style', 'photorealistic')
        duration_seconds = data.get('duration', 8)
        fps = data.get('fps', 24)
        
        if not video_system:
            return jsonify({'error': 'Video system not available'}), 500
        
        logger.info(f"🎬 Batch generating {len(prompts)} videos...")
        
        # Generate videos in batch
        video_paths = video_system.batch_generate_videos(
            prompts=prompts,
            style=style,
            duration_seconds=duration_seconds,
            fps=fps
        )
        
        # Prepare response
        results = []
        for i, video_path in enumerate(video_paths):
            if video_path:
                filename = Path(video_path).name
                results.append({
                    'index': i,
                    'prompt': prompts[i],
                    'video_path': video_path,
                    'filename': filename,
                    'success': True
                })
            else:
                results.append({
                    'index': i,
                    'prompt': prompts[i],
                    'success': False,
                    'error': 'Generation failed'
                })
        
        successful_count = len([r for r in results if r['success']])
        
        return jsonify({
            'success': True,
            'total_prompts': len(prompts),
            'successful_generations': successful_count,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"❌ Error in batch video generation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ultra-realistic-video/video/<filename>', methods=['GET'])
def serve_video(filename):
    """Serve generated video files"""
    try:
        video_path = video_system.output_dir / filename
        
        if not video_path.exists():
            return jsonify({'error': 'Video not found'}), 404
        
        return send_file(str(video_path), mimetype='video/mp4')
        
    except Exception as e:
        logger.error(f"❌ Error serving video: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ultra-realistic-video/pipelines', methods=['GET'])
def get_available_pipelines():
    """Get detailed information about available pipelines"""
    try:
        if not video_system:
            return jsonify({'error': 'Video system not available'}), 500
        
        pipelines = {
            "stable_video": {
                "name": "Stable Video Diffusion",
                "description": "Best for image-to-video generation with realistic motion",
                "type": "Image-to-Video",
                "best_for": "Realistic motion, professional quality",
                "requirements": "High VRAM, CUDA recommended"
            },
            "modelscope_t2v": {
                "name": "ModelScope T2V",
                "description": "Stable text-to-video generation with good quality",
                "type": "Text-to-Video",
                "best_for": "Direct video generation, creative content",
                "requirements": "Medium VRAM, CPU/GPU"
            },
            "zeroscope": {
                "name": "Zeroscope v2 XL",
                "description": "HD text-to-video with high resolution output",
                "type": "Text-to-Video",
                "best_for": "HD quality, cinematic videos",
                "requirements": "High VRAM, CUDA recommended"
            },
            "animatediff": {
                "name": "AnimateDiff-like",
                "description": "Frame animation from images with motion control",
                "type": "Image Animation",
                "best_for": "Controlled motion, image-to-video",
                "requirements": "Medium VRAM, CPU/GPU"
            },
            "interpolation": {
                "name": "Frame Interpolation",
                "description": "Motion effects using frame interpolation",
                "type": "Motion Effects",
                "best_for": "Simple motion, always available",
                "requirements": "Low VRAM, CPU only"
            }
        }
        
        return jsonify({
            'success': True,
            'pipelines': pipelines,
            'status': video_system.get_pipeline_status()
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting pipeline info: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ultra-realistic-video/test', methods=['POST'])
def test_video_generation():
    """Test video generation with a simple example"""
    try:
        if not video_system:
            return jsonify({'error': 'Video system not available'}), 500
        
        # Create a simple test image
        from PIL import Image
        import numpy as np
        
        # Create a test image (gradient)
        test_image = Image.fromarray(
            np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        )
        
        # Save test image
        test_image_path = video_system.output_dir / "test_image.png"
        test_image.save(test_image_path)
        
        logger.info("🧪 Testing video generation with simple example...")
        
        # Generate video from test image
        video_path = video_system.generate_video_from_image_pipeline(
            image=str(test_image_path),
            motion_prompt="gentle zoom",
            duration_seconds=3,
            fps=8,
            width=512,
            height=512,
            pipeline_type="interpolation"  # Use interpolation for testing
        )
        
        if video_path:
            filename = Path(video_path).name
            
            return jsonify({
                'success': True,
                'message': 'Video generation test successful',
                'test_image_path': str(test_image_path),
                'video_path': video_path,
                'filename': filename
            })
        else:
            return jsonify({'error': 'Video generation test failed'}), 500
        
    except Exception as e:
        logger.error(f"❌ Error in video generation test: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ultra-realistic-video/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        if not video_system:
            return jsonify({
                'status': 'unhealthy',
                'error': 'Video system not initialized'
            }), 500
        
        # Check if any pipelines are loaded
        pipeline_status = video_system.get_pipeline_status()
        loaded_pipelines = [k for k, v in pipeline_status.items() if v]
        
        if not loaded_pipelines:
            return jsonify({
                'status': 'unhealthy',
                'error': 'No video pipelines loaded'
            }), 500
        
        return jsonify({
            'status': 'healthy',
            'device': video_system.device,
            'loaded_pipelines': loaded_pipelines,
            'output_directory': str(video_system.output_dir)
        })
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/api/ultra-realistic-video/list-videos', methods=['GET'])
def list_generated_videos():
    """List all generated videos"""
    try:
        if not video_system:
            return jsonify({'error': 'Video system not available'}), 500
        
        video_files = []
        for file_path in video_system.output_dir.glob("*.mp4"):
            stat = file_path.stat()
            video_files.append({
                'filename': file_path.name,
                'path': str(file_path),
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        # Sort by creation time (newest first)
        video_files.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify({
            'success': True,
            'total_videos': len(video_files),
            'videos': video_files
        })
        
    except Exception as e:
        logger.error(f"❌ Error listing videos: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Ultra-Realistic Video Generation API")
    print("=" * 50)
    print("🎬 Available Endpoints:")
    print("  POST /api/ultra-realistic-video/generate-from-image")
    print("  POST /api/ultra-realistic-video/generate-direct")
    print("  POST /api/ultra-realistic-video/generate-pipeline")
    print("  POST /api/ultra-realistic-video/batch-generate")
    print("  GET  /api/ultra-realistic-video/status")
    print("  GET  /api/ultra-realistic-video/pipelines")
    print("  GET  /api/ultra-realistic-video/health")
    print("  GET  /api/ultra-realistic-video/list-videos")
    print("  POST /api/ultra-realistic-video/test")
    print("")
    print("🌐 API will be available at: http://localhost:5007")
    print("📁 Videos will be saved in: ultra_realistic_video_outputs/")
    print("")
    
    # Run the API server
    app.run(host='0.0.0.0', port=5007, debug=True) 