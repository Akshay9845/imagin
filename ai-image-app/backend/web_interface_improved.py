#!/usr/bin/env python3
"""
Improved web interface for the ultra-realistic generation system
with better handling of long-running requests
"""

from flask import Flask, render_template_string, request, jsonify, send_file
import os
from pathlib import Path
from datetime import datetime
from ultra_realistic_system import UltraRealisticSystem
import logging
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize the ultra-realistic system
system = None

def init_system():
    """Initialize the ultra-realistic system"""
    global system
    if system is None:
        logger.info("Initializing ultra-realistic system...")
        system = UltraRealisticSystem()
        logger.info("✅ Ultra-realistic system initialized")

# HTML template for the improved web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ultra-Realistic AI Generator</title>
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
            font-size: 2.5em;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #2d3748;
        }
        input[type="text"], textarea, select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        input[type="text"]:focus, textarea:focus, select:focus {
            outline: none;
            border-color: #667eea;
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
            transition: transform 0.2s;
            width: 100%;
        }
        button:hover {
            transform: translateY(-2px);
        }
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
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
        .result img, .result video {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .tabs {
            display: flex;
            margin-bottom: 20px;
            border-bottom: 2px solid #e2e8f0;
        }
        .tab {
            padding: 15px 30px;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.3s;
        }
        .tab.active {
            border-bottom-color: #667eea;
            color: #667eea;
            font-weight: 600;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .info-card {
            background: #f7fafc;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background-color: #e2e8f0;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            width: 0%;
            transition: width 0.3s ease;
        }
        .warning {
            background: #fef5e7;
            color: #d69e2e;
            padding: 15px;
            border-radius: 8px;
            border: 2px solid #f6ad55;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 Ultra-Realistic AI Generator</h1>
        
        <div class="warning">
            <strong>⚠️ Important:</strong> Image generation may take 5-15 minutes on CPU. Please be patient and don't close the browser tab.
        </div>
        
        <div class="tabs">
            <div class="tab active" onclick="switchTab('image')">🖼️ Generate Image</div>
            <div class="tab" onclick="switchTab('video')">🎬 Generate Video</div>
            <div class="tab" onclick="switchTab('status')">📊 System Status</div>
        </div>

        <!-- Image Generation Tab -->
        <div id="image-tab" class="tab-content active">
            <form id="image-form">
                <div class="form-group">
                    <label for="prompt">Image Prompt:</label>
                    <textarea id="prompt" name="prompt" placeholder="Describe the image you want to generate... (e.g., A beautiful sunset over mountains with golden hour lighting, ultra-realistic, high quality)"></textarea>
                </div>
                
                <div class="grid">
                    <div class="form-group">
                        <label for="style">Style:</label>
                        <select id="style" name="style">
                            <option value="photorealistic">Photorealistic</option>
                            <option value="artistic">Artistic Realistic</option>
                            <option value="cinematic">Cinematic</option>
                            <option value="anime_realistic">Anime Realistic</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="width">Width:</label>
                        <select id="width" name="width">
                            <option value="512" selected>512px (Faster)</option>
                            <option value="768">768px</option>
                            <option value="1024">1024px (Slower)</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="height">Height:</label>
                        <select id="height" name="height">
                            <option value="512" selected>512px (Faster)</option>
                            <option value="768">768px</option>
                            <option value="1024">1024px (Slower)</option>
                        </select>
                    </div>
                </div>
                
                <button type="submit" id="generate-btn">🚀 Generate Ultra-Realistic Image</button>
            </form>
            
            <div id="image-status" class="status" style="display: none;"></div>
            <div id="progress-container" style="display: none;">
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-fill"></div>
                </div>
                <p id="progress-text">Initializing...</p>
            </div>
            <div id="image-result" class="result"></div>
        </div>

        <!-- Video Generation Tab -->
        <div id="video-tab" class="tab-content">
            <div class="warning">
                <strong>⚠️ Video Generation:</strong> This creates a video from a generated image with motion effects. Generation may take 10-20 minutes.
            </div>
            
            <form id="video-form">
                <div class="form-group">
                    <label for="video-prompt">Video Prompt:</label>
                    <textarea id="video-prompt" name="prompt" placeholder="Describe the video you want to generate... (e.g., A beautiful sunset over mountains with gentle camera movement)"></textarea>
                </div>
                
                <div class="grid">
                    <div class="form-group">
                        <label for="video-style">Style:</label>
                        <select id="video-style" name="style">
                            <option value="photorealistic">Photorealistic</option>
                            <option value="artistic">Artistic Realistic</option>
                            <option value="cinematic">Cinematic</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="motion-type">Motion Type:</label>
                        <select id="motion-type" name="motion_type">
                            <option value="gentle">Gentle Movement</option>
                            <option value="zoom">Zoom Effect</option>
                            <option value="pan">Pan Effect</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="duration">Duration (seconds):</label>
                        <select id="duration" name="duration">
                            <option value="3">3 seconds</option>
                            <option value="5" selected>5 seconds</option>
                            <option value="8">8 seconds</option>
                        </select>
                    </div>
                </div>
                
                <button type="submit" id="video-btn">🎬 Generate Ultra-Realistic Video</button>
            </form>
            
            <div id="video-status" class="status" style="display: none;"></div>
            <div id="video-progress-container" style="display: none;">
                <div class="progress-bar">
                    <div class="progress-fill" id="video-progress-fill"></div>
                </div>
                <p id="video-progress-text">Initializing...</p>
            </div>
            <div id="video-result" class="result"></div>
        </div>

        <!-- Status Tab -->
        <div id="status-tab" class="tab-content">
            <div class="info-card">
                <h3>🖥️ System Information</h3>
                <p><strong>Device:</strong> <span id="device">Loading...</span></p>
                <p><strong>Available Models:</strong> <span id="models">Loading...</span></p>
                <p><strong>Output Directory:</strong> <span id="output-dir">Loading...</span></p>
            </div>
            
            <div class="info-card">
                <h3>🎨 Available Styles</h3>
                <div id="styles-list">Loading...</div>
            </div>
            
            <button onclick="refreshStatus()" style="width: auto; margin-top: 20px;">🔄 Refresh Status</button>
        </div>
    </div>

    <script>
        let generationInProgress = false;
        let progressInterval = null;

        function switchTab(tabName) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab content
            document.getElementById(tabName + '-tab').classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
            
            // Load status if status tab
            if (tabName === 'status') {
                loadStatus();
            }
        }

        function showStatus(elementId, message, type) {
            const status = document.getElementById(elementId);
            status.textContent = message;
            status.className = 'status ' + type;
            status.style.display = 'block';
        }

        function hideStatus(elementId) {
            document.getElementById(elementId).style.display = 'none';
        }

        function showProgress() {
            document.getElementById('progress-container').style.display = 'block';
            document.getElementById('progress-fill').style.width = '0%';
            document.getElementById('progress-text').textContent = 'Starting generation...';
            
            // Simulate progress
            let progress = 0;
            progressInterval = setInterval(() => {
                progress += Math.random() * 5;
                if (progress > 90) progress = 90; // Don't go to 100% until complete
                
                document.getElementById('progress-fill').style.width = progress + '%';
                document.getElementById('progress-text').textContent = `Generating... ${Math.round(progress)}%`;
            }, 2000);
        }

        function hideProgress() {
            document.getElementById('progress-container').style.display = 'none';
            if (progressInterval) {
                clearInterval(progressInterval);
                progressInterval = null;
            }
        }

        function completeProgress() {
            document.getElementById('progress-fill').style.width = '100%';
            document.getElementById('progress-text').textContent = 'Generation complete!';
            
            setTimeout(() => {
                hideProgress();
            }, 2000);
        }

        async function loadStatus() {
            try {
                const response = await fetch('/api/ultra-realistic/status');
                const data = await response.json();
                
                if (data.success) {
                    document.getElementById('device').textContent = data.device;
                    document.getElementById('models').textContent = data.available_models.join(', ');
                    document.getElementById('output-dir').textContent = data.output_directory;
                }
                
                const stylesResponse = await fetch('/api/ultra-realistic/styles');
                const stylesData = await stylesResponse.json();
                
                if (stylesData.success) {
                    const stylesList = document.getElementById('styles-list');
                    stylesList.innerHTML = stylesData.styles.map(style => 
                        `<div style="margin: 10px 0; padding: 10px; background: white; border-radius: 5px;">
                            <strong>${style.name}</strong><br>
                            <small>${style.description}</small><br>
                            <small style="color: #666;">Best for: ${style.best_for}</small>
                        </div>`
                    ).join('');
                }
            } catch (error) {
                console.error('Error loading status:', error);
            }
        }

        function refreshStatus() {
            loadStatus();
        }

        // Image generation form
        document.getElementById('image-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            if (generationInProgress) {
                alert('Generation already in progress. Please wait.');
                return;
            }
            
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData);
            
            generationInProgress = true;
            showStatus('image-status', '🎨 Starting ultra-realistic image generation... This may take 5-15 minutes.', 'loading');
            showProgress();
            document.getElementById('generate-btn').disabled = true;
            
            try {
                const response = await fetch('/api/ultra-realistic/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const result = await response.json();
                
                if (result.success) {
                    completeProgress();
                    showStatus('image-status', '✅ Image generated successfully!', 'success');
                    document.getElementById('image-result').innerHTML = `
                        <h3>Generated Image:</h3>
                        <img src="/api/ultra-realistic/image/${result.filename}" alt="Generated Image">
                        <p><strong>Prompt:</strong> ${data.prompt}</p>
                        <p><strong>Style:</strong> ${data.style}</p>
                        <p><strong>Size:</strong> ${data.width}x${data.height}</p>
                    `;
                } else {
                    hideProgress();
                    showStatus('image-status', `❌ Error: ${result.error}`, 'error');
                }
            } catch (error) {
                hideProgress();
                showStatus('image-status', `❌ Generation failed: ${error.message}`, 'error');
                console.error('Generation error:', error);
            } finally {
                generationInProgress = false;
                document.getElementById('generate-btn').disabled = false;
            }
        });

        // Video generation form (disabled for now)
        document.getElementById('video-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            alert('Video generation is coming soon! Please use image generation for now.');
        });

        // Load status on page load
        window.addEventListener('load', () => {
            loadStatus();
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main web interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/ultra-realistic/status')
def api_status():
    """Get system status"""
    try:
        return jsonify({
            "success": True,
            "status": "ready",
            "device": system.device,
            "cuda_available": False,  # We're using CPU
            "available_models": list(system.pipelines.keys()),
            "output_directory": str(system.output_dir)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/ultra-realistic/styles')
def api_styles():
    """Get available styles"""
    styles = [
        {
            "id": "photorealistic",
            "name": "Photorealistic",
            "description": "Ultra-realistic photos with accurate lighting and details",
            "best_for": "Portraits, landscapes, product photography"
        },
        {
            "id": "artistic",
            "name": "Artistic Realistic",
            "description": "Realistic images with artistic flair and cinematic lighting",
            "best_for": "Creative portraits, artistic scenes, concept art"
        },
        {
            "id": "cinematic",
            "name": "Cinematic",
            "description": "Movie-like quality with dramatic lighting and composition",
            "best_for": "Film stills, dramatic scenes, storytelling"
        },
        {
            "id": "anime_realistic",
            "name": "Anime Realistic",
            "description": "Anime style with realistic elements and high detail",
            "best_for": "Anime characters, stylized scenes, illustrations"
        }
    ]
    return jsonify({"success": True, "styles": styles})

@app.route('/api/ultra-realistic/generate', methods=['POST'])
def api_generate_image():
    """Generate ultra-realistic image with better error handling"""
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({"success": False, "error": "Prompt is required"})
        
        logger.info(f"Starting image generation: {data['prompt'][:50]}...")
        
        # Generate the image with reduced steps for faster generation
        image = system.generate_ultra_realistic_image(
            prompt=data['prompt'],
            style=data.get('style', 'photorealistic'),
            width=int(data.get('width', 512)),
            height=int(data.get('height', 512)),
            num_inference_steps=int(data.get('num_inference_steps', 15)),  # Reduced for speed
            guidance_scale=float(data.get('guidance_scale', 7.5))
        )
        
        # Save the image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"web_generated_{data.get('style', 'photorealistic')}_{timestamp}.png"
        image_path = system.output_dir / filename
        image.save(image_path)
        
        logger.info(f"✅ Image generated successfully: {filename}")
        
        return jsonify({
            "success": True,
            "filename": filename,
            "message": "Image generated successfully"
        })
        
    except Exception as e:
        logger.error(f"❌ Image generation failed: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/ultra-realistic/generate-video', methods=['POST'])
def api_generate_video():
    """Generate ultra-realistic video (placeholder)"""
    return jsonify({"success": False, "error": "Video generation is coming soon!"})

@app.route('/api/ultra-realistic/image/<filename>')
def serve_image(filename):
    """Serve generated images"""
    try:
        return send_file(f"ultra_realistic_outputs/{filename}")
    except FileNotFoundError:
        return "Image not found", 404

@app.route('/api/ultra-realistic/video/<filename>')
def serve_video(filename):
    """Serve generated videos"""
    try:
        return send_file(f"ultra_realistic_outputs/{filename}")
    except FileNotFoundError:
        return "Video not found", 404

if __name__ == '__main__':
    # Initialize the system
    init_system()
    
    print("🌐 Starting Improved Ultra-Realistic Web Interface...")
    print("📱 Access the interface at: http://localhost:5002")
    print("🔗 API endpoints available at: http://localhost:5002/api/ultra-realistic/")
    print("⚠️  Note: Image generation may take 5-15 minutes on CPU")
    
    app.run(host='0.0.0.0', port=5002, debug=False, threaded=True) 