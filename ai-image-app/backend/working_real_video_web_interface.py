#!/usr/bin/env python3
"""
Working Real Video Generation Web Interface
Provides a user-friendly interface for real video generation
"""

import os
import json
import base64
import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string, send_file
from flask_cors import CORS
import requests

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# API configuration
API_URL = "http://localhost:5005"

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎬 Real Video Generation</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        .tabs {
            display: flex;
            margin-bottom: 30px;
            border-bottom: 2px solid #eee;
        }
        
        .tab {
            padding: 15px 30px;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.3s ease;
            font-weight: 600;
        }
        
        .tab.active {
            border-bottom-color: #667eea;
            color: #667eea;
        }
        
        .tab:hover {
            background: #f8f9fa;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .form-group {
            margin-bottom: 25px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        
        .form-group input, .form-group textarea, .form-group select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s ease;
        }
        
        .form-group input:focus, .form-group textarea:focus, .form-group select:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .form-group textarea {
            resize: vertical;
            min-height: 100px;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease;
            width: 100%;
        }
        
        .btn:hover {
            transform: translateY(-2px);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .progress {
            width: 100%;
            height: 20px;
            background: #e1e5e9;
            border-radius: 10px;
            overflow: hidden;
            margin: 20px 0;
            display: none;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            width: 0%;
            transition: width 0.3s ease;
        }
        
        .status {
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            display: none;
        }
        
        .status.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .status.info {
            background: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }
        
        .video-gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        
        .video-item {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }
        
        .video-item video {
            width: 100%;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        
        .video-item h4 {
            margin-bottom: 10px;
            color: #333;
        }
        
        .video-item p {
            color: #666;
            margin-bottom: 15px;
        }
        
        .download-btn {
            background: #28a745;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
        }
        
        .system-status {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
        }
        
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .status-item {
            text-align: center;
            padding: 15px;
            background: white;
            border-radius: 8px;
            border: 1px solid #e1e5e9;
        }
        
        .status-item .icon {
            font-size: 2em;
            margin-bottom: 10px;
        }
        
        .status-item .label {
            font-weight: 600;
            color: #333;
            margin-bottom: 5px;
        }
        
        .status-item .value {
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 Real Video Generation</h1>
            <p>Generate real videos from text and images using AI</p>
        </div>
        
        <div class="content">
            <!-- System Status -->
            <div class="system-status">
                <h3>🖥️ System Status</h3>
                <div class="status-grid" id="systemStatus">
                    <div class="status-item">
                        <div class="icon">⏳</div>
                        <div class="label">Loading...</div>
                        <div class="value">Checking system</div>
                    </div>
                </div>
            </div>
            
            <!-- Tabs -->
            <div class="tabs">
                <div class="tab active" onclick="showTab('text-to-video')">📝 Text to Video</div>
                <div class="tab" onclick="showTab('image-to-video')">🖼️ Image to Video</div>
                <div class="tab" onclick="showTab('dancing')">💃 Dancing Videos</div>
                <div class="tab" onclick="showTab('scenes')">🎭 Scene Videos</div>
                <div class="tab" onclick="showTab('gallery')">🎬 Video Gallery</div>
            </div>
            
            <!-- Text to Video Tab -->
            <div id="text-to-video" class="tab-content active">
                <h3>Generate Video from Text</h3>
                <p>Create real videos from text descriptions (like Veo/Sora)</p>
                
                <form id="textToVideoForm">
                    <div class="form-group">
                        <label for="textPrompt">Video Description:</label>
                        <textarea id="textPrompt" placeholder="Describe the video you want to generate... (e.g., 'A person dancing modern dance in a studio with professional lighting')" required></textarea>
                    </div>
                    
                    <div class="form-group">
                        <label for="textDuration">Duration (seconds):</label>
                        <input type="number" id="textDuration" value="8" min="1" max="30">
                    </div>
                    
                    <div class="form-group">
                        <label for="textFPS">FPS:</label>
                        <select id="textFPS">
                            <option value="8">8 FPS (Fast)</option>
                            <option value="16">16 FPS (Medium)</option>
                            <option value="24" selected>24 FPS (Smooth)</option>
                            <option value="30">30 FPS (High Quality)</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="textPipeline">Pipeline:</label>
                        <select id="textPipeline">
                            <option value="auto">Auto-select best</option>
                            <option value="damo_t2v">Damo T2V (High Quality)</option>
                        </select>
                    </div>
                    
                    <button type="submit" class="btn">🎬 Generate Video</button>
                </form>
            </div>
            
            <!-- Image to Video Tab -->
            <div id="image-to-video" class="tab-content">
                <h3>Generate Video from Image</h3>
                <p>Animate static images into videos with real motion</p>
                
                <form id="imageToVideoForm">
                    <div class="form-group">
                        <label for="imageFile">Upload Image:</label>
                        <input type="file" id="imageFile" accept="image/*" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="imageMotion">Motion Type:</label>
                        <select id="imageMotion">
                            <option value="gentle movement">Gentle Movement</option>
                            <option value="zoom">Zoom Effect</option>
                            <option value="pan">Pan Effect</option>
                            <option value="rotation">Rotation</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="imageDuration">Duration (seconds):</label>
                        <input type="number" id="imageDuration" value="8" min="1" max="30">
                    </div>
                    
                    <div class="form-group">
                        <label for="imageFPS">FPS:</label>
                        <select id="imageFPS">
                            <option value="8">8 FPS (Fast)</option>
                            <option value="16">16 FPS (Medium)</option>
                            <option value="24" selected>24 FPS (Smooth)</option>
                            <option value="30">30 FPS (High Quality)</option>
                        </select>
                    </div>
                    
                    <button type="submit" class="btn">🎬 Generate Video</button>
                </form>
            </div>
            
            <!-- Dancing Videos Tab -->
            <div id="dancing" class="tab-content">
                <h3>Generate Dancing Videos</h3>
                <p>Create videos of people dancing in different styles</p>
                
                <form id="dancingForm">
                    <div class="form-group">
                        <label for="danceStyle">Dance Style:</label>
                        <select id="danceStyle">
                            <option value="modern dance">Modern Dance</option>
                            <option value="hip hop">Hip Hop</option>
                            <option value="ballet">Ballet</option>
                            <option value="breakdance">Breakdance</option>
                            <option value="salsa">Salsa</option>
                            <option value="contemporary">Contemporary</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="danceDuration">Duration (seconds):</label>
                        <input type="number" id="danceDuration" value="8" min="1" max="30">
                    </div>
                    
                    <div class="form-group">
                        <label for="danceFPS">FPS:</label>
                        <select id="danceFPS">
                            <option value="8">8 FPS (Fast)</option>
                            <option value="16">16 FPS (Medium)</option>
                            <option value="24" selected>24 FPS (Smooth)</option>
                            <option value="30">30 FPS (High Quality)</option>
                        </select>
                    </div>
                    
                    <button type="submit" class="btn">💃 Generate Dancing Video</button>
                </form>
            </div>
            
            <!-- Scene Videos Tab -->
            <div id="scenes" class="tab-content">
                <h3>Generate Scene Videos</h3>
                <p>Create videos of scenes with camera movement</p>
                
                <form id="sceneForm">
                    <div class="form-group">
                        <label for="sceneDescription">Scene Description:</label>
                        <textarea id="sceneDescription" placeholder="Describe the scene... (e.g., 'A beautiful sunset over mountains')" required></textarea>
                    </div>
                    
                    <div class="form-group">
                        <label for="sceneMotion">Camera Movement:</label>
                        <select id="sceneMotion">
                            <option value="gentle camera movement">Gentle Camera Movement</option>
                            <option value="zoom">Zoom</option>
                            <option value="pan">Pan</option>
                            <option value="tilt">Tilt</option>
                            <option value="orbit">Orbit</option>
                            <option value="dolly">Dolly</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="sceneDuration">Duration (seconds):</label>
                        <input type="number" id="sceneDuration" value="8" min="1" max="30">
                    </div>
                    
                    <div class="form-group">
                        <label for="sceneFPS">FPS:</label>
                        <select id="sceneFPS">
                            <option value="8">8 FPS (Fast)</option>
                            <option value="16">16 FPS (Medium)</option>
                            <option value="24" selected>24 FPS (Smooth)</option>
                            <option value="30">30 FPS (High Quality)</option>
                        </select>
                    </div>
                    
                    <button type="submit" class="btn">🎭 Generate Scene Video</button>
                </form>
            </div>
            
            <!-- Video Gallery Tab -->
            <div id="gallery" class="tab-content">
                <h3>Generated Videos</h3>
                <p>View and download your generated videos</p>
                
                <button onclick="loadVideos()" class="btn">🔄 Refresh Gallery</button>
                
                <div id="videoGallery" class="video-gallery">
                    <div style="text-align: center; grid-column: 1 / -1; padding: 40px; color: #666;">
                        Loading videos...
                    </div>
                </div>
            </div>
            
            <!-- Progress and Status -->
            <div class="progress" id="progress">
                <div class="progress-bar" id="progressBar"></div>
            </div>
            
            <div class="status" id="status"></div>
        </div>
    </div>
    
    <script>
        // Global variables
        let currentProgress = 0;
        let progressInterval;
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            loadSystemStatus();
            loadVideos();
        });
        
        // Tab switching
        function showTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }
        
        // Load system status
        async function loadSystemStatus() {
            try {
                const response = await fetch('http://localhost:5005/api/video/status');
                const data = await response.json();
                
                const statusGrid = document.getElementById('systemStatus');
                statusGrid.innerHTML = `
                    <div class="status-item">
                        <div class="icon">${data.status === 'running' ? '✅' : '❌'}</div>
                        <div class="label">System Status</div>
                        <div class="value">${data.status}</div>
                    </div>
                    <div class="status-item">
                        <div class="icon">🖥️</div>
                        <div class="label">Device</div>
                        <div class="value">${data.device}</div>
                    </div>
                    <div class="status-item">
                        <div class="icon">🎬</div>
                        <div class="label">Pipelines</div>
                        <div class="value">${data.pipelines_loaded} loaded</div>
                    </div>
                    <div class="status-item">
                        <div class="icon">🖼️</div>
                        <div class="label">Image Model</div>
                        <div class="value">${data.image_pipeline_loaded ? '✅' : '❌'}</div>
                    </div>
                `;
            } catch (error) {
                console.error('Error loading system status:', error);
                showStatus('Error loading system status', 'error');
            }
        }
        
        // Form submissions
        document.getElementById('textToVideoForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            await generateVideo('text', {
                prompt: document.getElementById('textPrompt').value,
                duration_seconds: parseInt(document.getElementById('textDuration').value),
                fps: parseInt(document.getElementById('textFPS').value),
                pipeline: document.getElementById('textPipeline').value
            });
        });
        
        document.getElementById('imageToVideoForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const file = document.getElementById('imageFile').files[0];
            if (!file) {
                showStatus('Please select an image file', 'error');
                return;
            }
            
            const base64 = await fileToBase64(file);
            await generateVideo('image', {
                image_base64: base64,
                motion_prompt: document.getElementById('imageMotion').value,
                duration_seconds: parseInt(document.getElementById('imageDuration').value),
                fps: parseInt(document.getElementById('imageFPS').value)
            });
        });
        
        document.getElementById('dancingForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            await generateVideo('dancing', {
                dance_style: document.getElementById('danceStyle').value,
                duration_seconds: parseInt(document.getElementById('danceDuration').value),
                fps: parseInt(document.getElementById('danceFPS').value)
            });
        });
        
        document.getElementById('sceneForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            await generateVideo('scene', {
                scene_description: document.getElementById('sceneDescription').value,
                motion_type: document.getElementById('sceneMotion').value,
                duration_seconds: parseInt(document.getElementById('sceneDuration').value),
                fps: parseInt(document.getElementById('sceneFPS').value)
            });
        });
        
        // Generate video
        async function generateVideo(type, data) {
            try {
                showStatus('Starting video generation...', 'info');
                showProgress();
                
                let endpoint;
                switch(type) {
                    case 'text':
                        endpoint = '/api/video/generate-from-text';
                        break;
                    case 'image':
                        endpoint = '/api/video/generate-from-image';
                        break;
                    case 'dancing':
                        endpoint = '/api/video/generate-dancing';
                        break;
                    case 'scene':
                        endpoint = '/api/video/generate-scene';
                        break;
                }
                
                const response = await fetch('http://localhost:5005' + endpoint, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showStatus('Video generated successfully!', 'success');
                    hideProgress();
                    loadVideos();
                } else {
                    showStatus('Error: ' + result.error, 'error');
                    hideProgress();
                }
                
            } catch (error) {
                console.error('Error generating video:', error);
                showStatus('Error generating video: ' + error.message, 'error');
                hideProgress();
            }
        }
        
        // Load videos
        async function loadVideos() {
            try {
                const response = await fetch('http://localhost:5005/api/video/videos');
                const data = await response.json();
                
                const gallery = document.getElementById('videoGallery');
                
                if (data.videos.length === 0) {
                    gallery.innerHTML = '<div style="text-align: center; grid-column: 1 / -1; padding: 40px; color: #666;">No videos generated yet. Create your first video!</div>';
                    return;
                }
                
                gallery.innerHTML = data.videos.map(video => `
                    <div class="video-item">
                        <video controls>
                            <source src="http://localhost:5005/api/video/download/${video.filename}" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                        <h4>${video.filename}</h4>
                        <p>Size: ${formatFileSize(video.size)}<br>Created: ${new Date(video.created).toLocaleString()}</p>
                        <a href="http://localhost:5005/api/video/download/${video.filename}" class="download-btn" download>📥 Download</a>
                    </div>
                `).join('');
                
            } catch (error) {
                console.error('Error loading videos:', error);
                document.getElementById('videoGallery').innerHTML = '<div style="text-align: center; grid-column: 1 / -1; padding: 40px; color: #666;">Error loading videos</div>';
            }
        }
        
        // Utility functions
        function fileToBase64(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.readAsDataURL(file);
                reader.onload = () => {
                    const base64 = reader.result.split(',')[1];
                    resolve(base64);
                };
                reader.onerror = error => reject(error);
            });
        }
        
        function formatFileSize(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }
        
        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = 'status ' + type;
            status.style.display = 'block';
            
            setTimeout(() => {
                status.style.display = 'none';
            }, 5000);
        }
        
        function showProgress() {
            document.getElementById('progress').style.display = 'block';
            currentProgress = 0;
            updateProgress(0);
            
            progressInterval = setInterval(() => {
                currentProgress += Math.random() * 10;
                if (currentProgress > 90) currentProgress = 90;
                updateProgress(currentProgress);
            }, 1000);
        }
        
        function hideProgress() {
            clearInterval(progressInterval);
            updateProgress(100);
            setTimeout(() => {
                document.getElementById('progress').style.display = 'none';
            }, 1000);
        }
        
        function updateProgress(percent) {
            document.getElementById('progressBar').style.width = percent + '%';
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Serve the web interface"""
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    print("🌐 Working Real Video Generation Web Interface")
    print("=" * 50)
    print("🚀 Starting web interface on http://localhost:5006")
    print("🎯 Features:")
    print("   • Text-to-Video (Real content generation)")
    print("   • Image-to-Video (Real motion animation)")
    print("   • Dancing videos")
    print("   • Scene videos with motion")
    print("   • Video gallery and downloads")
    print(f"🔗 API Server: {API_URL}")
    print("Press Ctrl+C to stop the web interface")
    
    app.run(host='0.0.0.0', port=5006, debug=True) 