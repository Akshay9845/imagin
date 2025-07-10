#!/usr/bin/env python3
"""
Working Video Generation API Server
Uses the simple video system to provide both pipeline options reliably
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import logging
import os
from pathlib import Path
from simple_video_system import SimpleVideoSystem
from PIL import Image
import io
import base64

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize video system
video_system = SimpleVideoSystem()

@app.route('/api/video/status', methods=['GET'])
def get_status():
    """Get system status"""
    try:
        status = video_system.get_pipeline_status()
        pipelines = video_system.get_available_pipelines()
        
        return jsonify({
            "status": "running",
            "device": video_system.device,
            "pipelines": status,
            "pipeline_descriptions": pipelines,
            "output_directory": str(video_system.output_dir)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/video/generate-from-image', methods=['POST'])
def generate_from_image():
    """Option A: Generate video from image"""
    try:
        data = request.get_json()
        
        # Get parameters
        image_data = data.get('image')  # Base64 encoded image
        motion_prompt = data.get('motion_prompt', 'gentle movement')
        duration_seconds = data.get('duration_seconds', 5)
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
        
        # Generate video
        video_path = video_system.generate_video_from_image_pipeline(
            image=image,
            motion_prompt=motion_prompt,
            duration_seconds=duration_seconds,
            fps=fps,
            width=width,
            height=height,
            pipeline_type="interpolation"
        )
        
        return jsonify({
            "success": True,
            "video_path": video_path,
            "filename": Path(video_path).name,
            "pipeline": "Option A (Image → Motion → Video)"
        })
        
    except Exception as e:
        logger.error(f"Error generating video from image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/video/generate-direct', methods=['POST'])
def generate_direct():
    """Option B: Generate video directly from text"""
    try:
        data = request.get_json()
        
        # Get parameters
        prompt = data.get('prompt')
        duration_seconds = data.get('duration_seconds', 5)
        fps = data.get('fps', 24)
        width = data.get('width', 512)
        height = data.get('height', 512)
        
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        # Generate video
        video_path = video_system.generate_direct_text_to_video(
            prompt=prompt,
            duration_seconds=duration_seconds,
            fps=fps,
            width=width,
            height=height,
            pipeline_type="interpolation"
        )
        
        return jsonify({
            "success": True,
            "video_path": video_path,
            "filename": Path(video_path).name,
            "pipeline": "Option B (Text to Video Direct)"
        })
        
    except Exception as e:
        logger.error(f"Error generating direct video: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/video/generate-both', methods=['POST'])
def generate_both():
    """Generate video using both pipelines"""
    try:
        data = request.get_json()
        
        # Get parameters
        prompt = data.get('prompt')
        image_data = data.get('image')  # Optional
        motion_prompt = data.get('motion_prompt', 'gentle movement')
        duration_seconds = data.get('duration_seconds', 5)
        fps = data.get('fps', 24)
        width = data.get('width', 512)
        height = data.get('height', 512)
        
        results = {}
        
        # Option A: Image to Video (if image provided)
        if image_data:
            try:
                # Decode image
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                
                video_path_a = video_system.generate_video_from_image_pipeline(
                    image=image,
                    motion_prompt=motion_prompt,
                    duration_seconds=duration_seconds,
                    fps=fps,
                    width=width,
                    height=height,
                    pipeline_type="interpolation"
                )
                results['option_a'] = {
                    "success": True,
                    "video_path": video_path_a,
                    "filename": Path(video_path_a).name,
                    "pipeline": "Option A (Image → Motion → Video)"
                }
            except Exception as e:
                results['option_a'] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Option B: Direct Text to Video
        if prompt:
            try:
                video_path_b = video_system.generate_direct_text_to_video(
                    prompt=prompt,
                    duration_seconds=duration_seconds,
                    fps=fps,
                    width=width,
                    height=height,
                    pipeline_type="interpolation"
                )
                results['option_b'] = {
                    "success": True,
                    "video_path": video_path_b,
                    "filename": Path(video_path_b).name,
                    "pipeline": "Option B (Text to Video Direct)"
                }
            except Exception as e:
                results['option_b'] = {
                    "success": False,
                    "error": str(e)
                }
        
        return jsonify({
            "success": True,
            "results": results,
            "message": "Both pipeline options attempted"
        })
        
    except Exception as e:
        logger.error(f"Error generating with both pipelines: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/video/videos', methods=['GET'])
def list_videos():
    """List all generated videos"""
    try:
        video_files = list(video_system.output_dir.glob("*.mp4"))
        videos = []
        
        for video_file in video_files:
            size_mb = video_file.stat().st_size / (1024 * 1024)
            videos.append({
                "filename": video_file.name,
                "path": str(video_file),
                "size_mb": round(size_mb, 2),
                "created": video_file.stat().st_mtime
            })
        
        # Sort by creation time (newest first)
        videos.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify({
            "videos": videos,
            "count": len(videos)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/video/download/<filename>', methods=['GET'])
def download_video(filename):
    """Download a specific video file"""
    try:
        video_path = video_system.output_dir / filename
        
        if not video_path.exists():
            return jsonify({"error": "Video not found"}), 404
        
        return send_file(video_path, as_attachment=True)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/video/pipelines', methods=['GET'])
def get_pipelines():
    """Get available pipeline information"""
    try:
        pipelines = video_system.get_available_pipelines()
        status = video_system.get_pipeline_status()
        
        pipeline_info = {}
        for pipeline, description in pipelines.items():
            pipeline_info[pipeline] = {
                "description": description,
                "available": status.get(pipeline, False)
            }
        
        return jsonify({
            "pipelines": pipeline_info,
            "recommendations": {
                "option_a": "Best for controlled, realistic motion",
                "option_b": "Best for creative, dynamic content"
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """API home page"""
    return jsonify({
        "message": "Working Video Generation API",
        "version": "1.0",
        "pipelines": {
            "option_a": "Image → Motion → Video (Best for control)",
            "option_b": "Text to Video Direct (Best for creativity)"
        },
        "endpoints": {
            "status": "GET /api/video/status",
            "generate_from_image": "POST /api/video/generate-from-image",
            "generate_direct": "POST /api/video/generate-direct",
            "generate_both": "POST /api/video/generate-both",
            "list_videos": "GET /api/video/videos",
            "download_video": "GET /api/video/download/<filename>",
            "pipelines": "GET /api/video/pipelines"
        }
    })

if __name__ == '__main__':
    print("🎬 Working Video Generation API Server")
    print("=" * 50)
    print("🚀 Starting server on http://localhost:5003")
    print("")
    print("🎯 Available endpoints:")
    print("   GET  /api/video/status")
    print("   POST /api/video/generate-from-image")
    print("   POST /api/video/generate-direct")
    print("   POST /api/video/generate-both")
    print("   GET  /api/video/videos")
    print("   GET  /api/video/download/<filename>")
    print("   GET  /api/video/pipelines")
    print("")
    print("🔁 Pipeline Options:")
    print("   • Option A: Image → Motion → Video (Best for control)")
    print("   • Option B: Text to Video Direct (Best for creativity)")
    print("")
    print("Press Ctrl+C to stop the server")
    print("")
    
    app.run(host='0.0.0.0', port=5003, debug=True) 