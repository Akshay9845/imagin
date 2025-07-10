#!/usr/bin/env python3
"""
Real Video Generation API Server
Generates actual video content from text prompts like Veo/Sora
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import logging
import os
from pathlib import Path
from real_video_system import RealVideoSystem
from PIL import Image
import io
import base64

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize real video system
video_system = RealVideoSystem()

@app.route('/api/real-video/status', methods=['GET'])
def get_status():
    """Get system status"""
    try:
        info = video_system.get_system_info()
        return jsonify({
            "status": "running",
            "device": info['device'],
            "models_loaded": info['models_loaded'],
            "pipelines": info['available_pipelines'],
            "pipeline_descriptions": info['pipeline_descriptions'],
            "output_directory": info['output_directory']
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/real-video/generate', methods=['POST'])
def generate_real_video():
    """Generate real video from text prompt (like Veo)"""
    try:
        data = request.get_json()
        
        # Get parameters
        prompt = data.get('prompt')
        duration_seconds = data.get('duration_seconds', 8)
        fps = data.get('fps', 24)
        width = data.get('width', 512)
        height = data.get('height', 512)
        pipeline_type = data.get('pipeline_type', 'auto')
        
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        # Generate real video
        video_path = video_system.generate_real_video_from_text(
            prompt=prompt,
            duration_seconds=duration_seconds,
            fps=fps,
            width=width,
            height=height,
            pipeline_type=pipeline_type
        )
        
        return jsonify({
            "success": True,
            "video_path": video_path,
            "filename": Path(video_path).name,
            "prompt": prompt,
            "duration": duration_seconds,
            "fps": fps,
            "resolution": f"{width}x{height}"
        })
        
    except Exception as e:
        logger.error(f"Error generating real video: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/real-video/generate-dancing', methods=['POST'])
def generate_dancing_video():
    """Generate a person dancing video (like Veo examples)"""
    try:
        data = request.get_json()
        
        # Get parameters
        dance_style = data.get('dance_style', 'modern dance')
        duration_seconds = data.get('duration_seconds', 8)
        fps = data.get('fps', 24)
        width = data.get('width', 512)
        height = data.get('height', 512)
        
        # Generate dancing video
        video_path = video_system.generate_person_dancing_video(
            dance_style=dance_style,
            duration_seconds=duration_seconds,
            fps=fps,
            width=width,
            height=height
        )
        
        return jsonify({
            "success": True,
            "video_path": video_path,
            "filename": Path(video_path).name,
            "dance_style": dance_style,
            "duration": duration_seconds,
            "fps": fps,
            "resolution": f"{width}x{height}"
        })
        
    except Exception as e:
        logger.error(f"Error generating dancing video: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/real-video/generate-scene', methods=['POST'])
def generate_scene_video():
    """Generate scene-based video with specific motion"""
    try:
        data = request.get_json()
        
        # Get parameters
        scene_description = data.get('scene_description')
        motion_type = data.get('motion_type', 'gentle camera movement')
        duration_seconds = data.get('duration_seconds', 8)
        fps = data.get('fps', 24)
        width = data.get('width', 512)
        height = data.get('height', 512)
        
        if not scene_description:
            return jsonify({"error": "No scene description provided"}), 400
        
        # Generate scene video
        video_path = video_system.generate_scene_video(
            scene_description=scene_description,
            motion_type=motion_type,
            duration_seconds=duration_seconds,
            fps=fps,
            width=width,
            height=height
        )
        
        return jsonify({
            "success": True,
            "video_path": video_path,
            "filename": Path(video_path).name,
            "scene_description": scene_description,
            "motion_type": motion_type,
            "duration": duration_seconds,
            "fps": fps,
            "resolution": f"{width}x{height}"
        })
        
    except Exception as e:
        logger.error(f"Error generating scene video: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/real-video/generate-from-image', methods=['POST'])
def generate_from_image():
    """Generate video from image using real video models"""
    try:
        data = request.get_json()
        
        # Get parameters
        image_data = data.get('image')  # Base64 encoded image
        motion_prompt = data.get('motion_prompt', 'gentle movement')
        duration_seconds = data.get('duration_seconds', 8)
        fps = data.get('fps', 24)
        width = data.get('width', 512)
        height = data.get('height', 512)
        
        # Decode image
        if image_data:
            # Remove data URL prefix if present
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        else:
            return jsonify({"error": "No image provided"}), 400
        
        # Generate video from image
        video_path = video_system.generate_video_from_image(
            image=image,
            motion_prompt=motion_prompt,
            duration_seconds=duration_seconds,
            fps=fps,
            width=width,
            height=height
        )
        
        return jsonify({
            "success": True,
            "video_path": video_path,
            "filename": Path(video_path).name,
            "motion_prompt": motion_prompt,
            "duration": duration_seconds,
            "fps": fps,
            "resolution": f"{width}x{height}"
        })
        
    except Exception as e:
        logger.error(f"Error generating video from image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/real-video/batch-generate', methods=['POST'])
def batch_generate():
    """Generate multiple videos in batch"""
    try:
        data = request.get_json()
        
        # Get parameters
        prompts = data.get('prompts', [])
        duration_seconds = data.get('duration_seconds', 8)
        fps = data.get('fps', 24)
        
        if not prompts:
            return jsonify({"error": "No prompts provided"}), 400
        
        # Generate videos in batch
        video_paths = video_system.batch_generate_videos(
            prompts=prompts,
            duration_seconds=duration_seconds,
            fps=fps
        )
        
        # Prepare results
        results = []
        for i, video_path in enumerate(video_paths):
            if video_path:
                results.append({
                    "success": True,
                    "prompt": prompts[i],
                    "video_path": video_path,
                    "filename": Path(video_path).name
                })
            else:
                results.append({
                    "success": False,
                    "prompt": prompts[i],
                    "error": "Generation failed"
                })
        
        return jsonify({
            "success": True,
            "total_prompts": len(prompts),
            "successful": len([r for r in results if r['success']]),
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Error in batch generation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/real-video/videos', methods=['GET'])
def list_videos():
    """List all generated videos"""
    try:
        video_dir = video_system.output_dir
        videos = []
        
        if video_dir.exists():
            for video_file in video_dir.glob("*.mp4"):
                videos.append({
                    "filename": video_file.name,
                    "path": str(video_file),
                    "size": video_file.stat().st_size,
                    "created": video_file.stat().st_ctime
                })
        
        # Sort by creation time (newest first)
        videos.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify({
            "success": True,
            "videos": videos,
            "total": len(videos)
        })
        
    except Exception as e:
        logger.error(f"Error listing videos: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/real-video/download/<filename>', methods=['GET'])
def download_video(filename):
    """Download a generated video"""
    try:
        video_path = video_system.output_dir / filename
        
        if not video_path.exists():
            return jsonify({"error": "Video not found"}), 404
        
        return send_file(
            video_path,
            as_attachment=True,
            download_name=filename,
            mimetype='video/mp4'
        )
        
    except Exception as e:
        logger.error(f"Error downloading video: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/real-video/pipelines', methods=['GET'])
def get_pipelines():
    """Get available pipeline information"""
    try:
        pipelines = video_system.get_available_pipelines()
        status = video_system.get_pipeline_status()
        
        return jsonify({
            "success": True,
            "pipelines": pipelines,
            "status": status,
            "device": video_system.device
        })
        
    except Exception as e:
        logger.error(f"Error getting pipelines: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """API home page"""
    return jsonify({
        "name": "Real Video Generation API",
        "description": "Generate actual video content from text prompts like Veo/Sora",
        "version": "1.0.0",
        "endpoints": {
            "status": "GET /api/real-video/status",
            "generate": "POST /api/real-video/generate",
            "generate_dancing": "POST /api/real-video/generate-dancing",
            "generate_scene": "POST /api/real-video/generate-scene",
            "generate_from_image": "POST /api/real-video/generate-from-image",
            "batch_generate": "POST /api/real-video/batch-generate",
            "list_videos": "GET /api/real-video/videos",
            "download": "GET /api/real-video/download/<filename>",
            "pipelines": "GET /api/real-video/pipelines"
        },
        "examples": {
            "generate_dancing": {
                "url": "/api/real-video/generate-dancing",
                "method": "POST",
                "body": {
                    "dance_style": "modern dance",
                    "duration_seconds": 8,
                    "fps": 24,
                    "width": 512,
                    "height": 512
                }
            },
            "generate_scene": {
                "url": "/api/real-video/generate-scene",
                "method": "POST",
                "body": {
                    "scene_description": "A beautiful sunset over mountains",
                    "motion_type": "gentle camera movement",
                    "duration_seconds": 8,
                    "fps": 24
                }
            }
        }
    })

if __name__ == '__main__':
    print("🎬 Real Video Generation API Server")
    print("=" * 50)
    print("🚀 Starting server on http://localhost:5005")
    print("🎯 Generate real videos from text prompts like Veo/Sora")
    print("📡 Available endpoints:")
    print("   GET  /api/real-video/status")
    print("   POST /api/real-video/generate")
    print("   POST /api/real-video/generate-dancing")
    print("   POST /api/real-video/generate-scene")
    print("   POST /api/real-video/generate-from-image")
    print("   POST /api/real-video/batch-generate")
    print("   GET  /api/real-video/videos")
    print("   GET  /api/real-video/download/<filename>")
    print("   GET  /api/real-video/pipelines")
    print("")
    print("🎬 Video Generation Examples:")
    print("   • Person dancing (modern, hip hop, ballet, breakdance, salsa)")
    print("   • Scene videos with camera motion")
    print("   • Direct text-to-video generation")
    print("   • Image-to-video with motion")
    print("")
    print("Press Ctrl+C to stop the server")
    
    app.run(host='0.0.0.0', port=5005, debug=True) 