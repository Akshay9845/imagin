#!/usr/bin/env python3
"""
Working Real Video Generation API Server
Uses updated libraries and working video generation system
"""

import os
import json
import base64
import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import traceback

# Import the working video system
from working_real_video_system import WorkingRealVideoSystem

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize video system
video_system = WorkingRealVideoSystem(output_dir="working_real_video_outputs")

@app.route('/')
def home():
    """Home page with API information"""
    return """
    <h1>🎬 Working Real Video Generation API</h1>
    <p>Real video generation with updated libraries</p>
    <h2>Available Endpoints:</h2>
    <ul>
        <li>GET /api/video/status - System status</li>
        <li>POST /api/video/generate-from-text - Generate video from text</li>
        <li>POST /api/video/generate-from-image - Generate video from image</li>
        <li>POST /api/video/generate-dancing - Generate dancing video</li>
        <li>POST /api/video/generate-scene - Generate scene video</li>
        <li>GET /api/video/videos - List generated videos</li>
        <li>GET /api/video/download/&lt;filename&gt; - Download video</li>
        <li>GET /api/video/pipelines - Available pipelines</li>
    </ul>
    """

@app.route('/api/video/status')
def get_status():
    """Get system status"""
    try:
        info = video_system.get_system_info()
        pipelines = video_system.get_available_pipelines()
        
        return jsonify({
            "status": "running",
            "device": info["device"],
            "output_directory": info["output_directory"],
            "pipelines_loaded": info["pipelines_loaded"],
            "image_pipeline_loaded": info["image_pipeline_loaded"],
            "diffusers_available": info["diffusers_available"],
            "available_pipelines": list(pipelines.keys()),
            "pipeline_descriptions": {
                name: details["description"] 
                for name, details in pipelines.items()
            }
        })
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/video/pipelines')
def get_pipelines():
    """Get available pipelines"""
    try:
        pipelines = video_system.get_available_pipelines()
        
        return jsonify({
            "pipelines": {
                name: {
                    "available": details["available"],
                    "description": details["description"],
                    "type": details["type"]
                }
                for name, details in pipelines.items()
            },
            "recommendations": {
                "text_to_video": "Best for creative content and scenes",
                "image_to_video": "Best for animating static images"
            }
        })
    except Exception as e:
        logger.error(f"Error getting pipelines: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/video/generate-from-text', methods=['POST'])
def generate_from_text():
    """Generate video from text prompt"""
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' parameter"}), 400
        
        prompt = data['prompt']
        duration = data.get('duration_seconds', 8)
        fps = data.get('fps', 24)
        width = data.get('width', 512)
        height = data.get('height', 512)
        pipeline = data.get('pipeline', 'auto')
        
        logger.info(f"Generating video from text: {prompt[:50]}...")
        
        video_path = video_system.generate_real_video_from_text(
            prompt=prompt,
            duration_seconds=duration,
            fps=fps,
            width=width,
            height=height,
            pipeline_name=pipeline
        )
        
        # Get filename for response
        filename = Path(video_path).name
        
        return jsonify({
            "success": True,
            "message": "Video generated successfully",
            "video_path": video_path,
            "filename": filename,
            "download_url": f"/api/video/download/{filename}"
        })
        
    except Exception as e:
        logger.error(f"Error generating video from text: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/video/generate-from-image', methods=['POST'])
def generate_from_image():
    """Generate video from uploaded image"""
    try:
        # Check if image file was uploaded
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file.filename == '':
                return jsonify({"error": "No image file selected"}), 400
            
            # Save uploaded image temporarily
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_image_path = f"temp_image_{timestamp}.png"
            image_file.save(temp_image_path)
            
        else:
            # Check for base64 encoded image
            data = request.get_json()
            if not data or 'image_base64' not in data:
                return jsonify({"error": "Missing image file or base64 data"}), 400
            
            # Decode base64 image
            image_data = base64.b64decode(data['image_base64'])
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_image_path = f"temp_image_{timestamp}.png"
            
            with open(temp_image_path, 'wb') as f:
                f.write(image_data)
        
        # Get parameters
        data = request.get_json() or {}
        motion_prompt = data.get('motion_prompt', 'gentle movement')
        duration = data.get('duration_seconds', 8)
        fps = data.get('fps', 24)
        pipeline = data.get('pipeline', 'auto')
        
        logger.info(f"Generating video from image: {temp_image_path}")
        
        video_path = video_system.generate_video_from_image(
            image_path=temp_image_path,
            motion_prompt=motion_prompt,
            duration_seconds=duration,
            fps=fps,
            pipeline_name=pipeline
        )
        
        # Clean up temporary image
        try:
            os.remove(temp_image_path)
        except:
            pass
        
        # Get filename for response
        filename = Path(video_path).name
        
        return jsonify({
            "success": True,
            "message": "Video generated successfully",
            "video_path": video_path,
            "filename": filename,
            "download_url": f"/api/video/download/{filename}"
        })
        
    except Exception as e:
        logger.error(f"Error generating video from image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/video/generate-dancing', methods=['POST'])
def generate_dancing():
    """Generate dancing video"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Missing parameters"}), 400
        
        dance_style = data.get('dance_style', 'modern dance')
        duration = data.get('duration_seconds', 8)
        fps = data.get('fps', 24)
        width = data.get('width', 512)
        height = data.get('height', 512)
        
        logger.info(f"Generating dancing video: {dance_style}")
        
        video_path = video_system.generate_person_dancing_video(
            dance_style=dance_style,
            duration_seconds=duration,
            fps=fps,
            width=width,
            height=height
        )
        
        # Get filename for response
        filename = Path(video_path).name
        
        return jsonify({
            "success": True,
            "message": "Dancing video generated successfully",
            "video_path": video_path,
            "filename": filename,
            "download_url": f"/api/video/download/{filename}"
        })
        
    except Exception as e:
        logger.error(f"Error generating dancing video: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/video/generate-scene', methods=['POST'])
def generate_scene():
    """Generate scene video"""
    try:
        data = request.get_json()
        
        if not data or 'scene_description' not in data:
            return jsonify({"error": "Missing 'scene_description' parameter"}), 400
        
        scene_description = data['scene_description']
        motion_type = data.get('motion_type', 'gentle camera movement')
        duration = data.get('duration_seconds', 8)
        fps = data.get('fps', 24)
        width = data.get('width', 512)
        height = data.get('height', 512)
        
        logger.info(f"Generating scene video: {scene_description[:50]}...")
        
        video_path = video_system.generate_scene_video(
            scene_description=scene_description,
            motion_type=motion_type,
            duration_seconds=duration,
            fps=fps,
            width=width,
            height=height
        )
        
        # Get filename for response
        filename = Path(video_path).name
        
        return jsonify({
            "success": True,
            "message": "Scene video generated successfully",
            "video_path": video_path,
            "filename": filename,
            "download_url": f"/api/video/download/{filename}"
        })
        
    except Exception as e:
        logger.error(f"Error generating scene video: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/video/videos')
def list_videos():
    """List all generated videos"""
    try:
        output_dir = Path(video_system.output_dir)
        videos = []
        
        if output_dir.exists():
            for video_file in output_dir.glob("*.mp4"):
                videos.append({
                    "filename": video_file.name,
                    "size": video_file.stat().st_size,
                    "created": datetime.fromtimestamp(video_file.stat().st_mtime).isoformat(),
                    "download_url": f"/api/video/download/{video_file.name}"
                })
        
        return jsonify({
            "videos": videos,
            "count": len(videos)
        })
        
    except Exception as e:
        logger.error(f"Error listing videos: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/video/download/<filename>')
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

if __name__ == '__main__':
    print("🎬 Working Real Video Generation API Server")
    print("=" * 50)
    print("🚀 Starting server on http://localhost:5005")
    print("🎯 Available endpoints:")
    print("   GET  /api/video/status")
    print("   POST /api/video/generate-from-text")
    print("   POST /api/video/generate-from-image")
    print("   POST /api/video/generate-dancing")
    print("   POST /api/video/generate-scene")
    print("   GET  /api/video/videos")
    print("   GET  /api/video/download/<filename>")
    print("   GET  /api/video/pipelines")
    print("🔁 Real Video Generation:")
    print("   • Text-to-Video (Real content generation)")
    print("   • Image-to-Video (Real motion animation)")
    print("   • Person dancing videos")
    print("   • Scene videos with motion")
    print("Press Ctrl+C to stop the server")
    
    app.run(host='0.0.0.0', port=5005, debug=True) 