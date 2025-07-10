#!/usr/bin/env python3
"""
Ultra-Realistic Video Generation Web Interface
Simplified web interface for the video generation system
"""

from flask import Flask, render_template_string, request, jsonify, send_file
from flask_cors import CORS
import os
import logging
from pathlib import Path
from datetime import datetime
import json
from ultra_realistic_video_system import UltraRealisticVideoSystem

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize the video system
try:
    video_system = UltraRealisticVideoSystem()
    logger.info("✅ Video system initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize video system: {e}")
    video_system = None

# Simple HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ultra-Realistic Video Generator</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .container {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        h1 {
            text-align: center;
            color: #4a5568;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
        }
        input[type="text"], textarea, select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 16px;
        }
        textarea {
            height: 100px;
            resize: vertical;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
        }
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .status {
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            text-align: center;
            font-weight: 600;
        }
        .status.loading {
            background: #fef5e7;
            color: #d69e2e;
            border: 2px solid #f6ad55;
        }
        .status.success {
            background: #f0fff4;
            color: #38a169;
            border: 2px solid #68d391;
        }
        .status.error {
            background: #fed7d7;
            color: #e53e3e;
            border: 2px solid #fc8181;
        }
        .result {
            margin-top: 30px;
            text-align: center;
        }
        .result video {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .warning {
            background: #fef5e7;
            color: #d69e2e;
            padding: 15px;
            border-radius: 8px;
            border: 2px solid #f6ad55;
            margin: 20px 0;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 Ultra-Realistic Video Generator</h1>
        
        <div class="warning">
            <strong>⚠️ Important:</strong> Video generation may take 10-30 minutes. Please be patient!
        </div>
        
        <form id="video-form">
            <div class="form-group">
                <label for="prompt">Video Prompt:</label>
                <textarea id="prompt" name="prompt" placeholder="Describe the video you want to generate... (e.g., A beautiful sunset over mountains with gentle camera movement, ultra-realistic, cinematic quality)"></textarea>
            </div>
            
            <div class="grid">
                <div class="form-group">
                    <label for="style">Style:</label>
                    <select id="style" name="style">
                        <option value="photorealistic">Photorealistic</option>
                        <option value="artistic">Artistic Realistic</option>
                        <option value="cinematic">Cinematic</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="pipeline_type">Pipeline:</label>
                    <select id="pipeline_type" name="pipeline_type">
                        <option value="auto">Auto (Recommended)</option>
                        <option value="stable_video">Stable Video Diffusion</option>
                        <option value="modelscope_t2v">ModelScope T2V</option>
                        <option value="interpolation">Frame Interpolation</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="duration">Duration (seconds):</label>
                    <select id="duration" name="duration">
                        <option value="3">3 seconds</option>
                        <option value="5">5 seconds</option>
                        <option value="8" selected>8 seconds</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="fps">Frame Rate:</label>
                    <select id="fps" name="fps">
                        <option value="8">8 FPS (Faster)</option>
                        <option value="24" selected>24 FPS (Smooth)</option>
                    </select>
                </div>
            </div>
            
            <button type="submit" id="generate-btn">🚀 Generate Ultra-Realistic Video</button>
        </form>
        
        <div id="status" class="status" style="display: none;"></div>
        <div id="result" class="result"></div>
        
        <div style="margin-top: 40px;">
            <h3>📊 System Status</h3>
            <div id="system-status">Loading...</div>
        </div>
    </div>

    <script>
        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = 'status ' + type;
            status.style.display = 'block';
        }

        async function loadSystemStatus() {
            try {
                const response = await fetch('/api/ultra-realistic-video/status');
                const data = await response.json();
                
                if (data.success) {
                    const statusHtml = `
                        <p><strong>Device:</strong> ${data.device}</p>
                        <p><strong>Output Directory:</strong> ${data.output_directory}</p>
                        <p><strong>Loaded Models:</strong> ${data.loaded_models.join(', ')}</p>
                    `;
                    document.getElementById('system-status').innerHTML = statusHtml;
                }
            } catch (error) {
                document.getElementById('system-status').innerHTML = '<p>Error loading status</p>';
            }
        }

        document.getElementById('video-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData);
            
            showStatus('🎬 Starting video generation... This may take 10-30 minutes.', 'loading');
            document.getElementById('generate-btn').disabled = true;
            
            try {
                const response = await fetch('/api/ultra-realistic-video/generate-pipeline', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showStatus('✅ Video generated successfully!', 'success');
                    document.getElementById('result').innerHTML = `
                        <h3>Generated Video:</h3>
                        <video controls style="max-width: 100%;">
                            <source src="/api/ultra-realistic-video/video/${result.filename}" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                        <p><strong>Prompt:</strong> ${data.prompt}</p>
                        <p><strong>Style:</strong> ${data.style}</p>
                        <p><strong>Pipeline:</strong> ${data.pipeline_type}</p>
                        <p><strong>Duration:</strong> ${data.duration} seconds</p>
                        <p><strong>FPS:</strong> ${data.fps}</p>
                    `;
                } else {
                    showStatus(`❌ Error: ${result.error}`, 'error');
                }
            } catch (error) {
                showStatus(`❌ Generation failed: ${error.message}`, 'error');
            } finally {
                document.getElementById('generate-btn').disabled = false;
            }
        });

        // Load status on page load
        window.addEventListener('load', () => {
            loadSystemStatus();
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/ultra-realistic-video/status', methods=['GET'])
def get_status():
    """Get video system status"""
    try:
        if not video_system:
            return jsonify({
                'success': False,
                'error': 'Video system not available'
            }), 500
        
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
        logger.error(f"❌ Error getting status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ultra-realistic-video/generate-pipeline', methods=['POST'])
def generate_pipeline():
    """Generate video using the complete pipeline"""
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Prompt is required'}), 400
        
        if not video_system:
            return jsonify({'error': 'Video system not available'}), 500
        
        logger.info(f"🎬 Starting pipeline generation: {data['prompt'][:50]}...")
        
        video_path = video_system.generate_ultra_realistic_video_pipeline(
            prompt=data['prompt'],
            style=data.get('style', 'photorealistic'),
            duration_seconds=int(data.get('duration', 8)),
            fps=int(data.get('fps', 24)),
            width=int(data.get('width', 1024)),
            height=int(data.get('height', 576)),
            pipeline_type=data.get('pipeline_type', 'auto')
        )
        
        if video_path:
            filename = Path(video_path).name
            
            return jsonify({
                'success': True,
                'video_path': video_path,
                'filename': filename,
                'prompt': data['prompt'],
                'style': data.get('style', 'photorealistic'),
                'duration': data.get('duration', 8),
                'fps': data.get('fps', 24),
                'pipeline_type': data.get('pipeline_type', 'auto')
            })
        else:
            return jsonify({'error': 'Failed to generate video'}), 500
        
    except Exception as e:
        logger.error(f"❌ Error generating video: {e}")
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

if __name__ == '__main__':
    print("🎬 Ultra-Realistic Video Generation Web Interface")
    print("=" * 50)
    print("🌐 Web interface will be available at: http://localhost:5004")
    print("📁 Videos will be saved in: ultra_realistic_video_outputs/")
    print("")
    
    app.run(host='0.0.0.0', port=5004, debug=True) 