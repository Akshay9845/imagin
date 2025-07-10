#!/usr/bin/env python3
"""
Real Video Generation Web Interface
Generate actual video content from text prompts like Veo/Sora
"""

from flask import Flask, render_template_string, request, jsonify, send_file
from flask_cors import CORS
import logging
import requests
import json
from pathlib import Path
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    <title>🎬 Real Video Generation - Like Veo/Sora</title>
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
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
            color: white;
        }
        
        .header h1 {
            font-size: 3rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 40px;
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        .card h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #555;
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
            transition: all 0.3s ease;
            width: 100%;
            margin-top: 10px;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .progress-container {
            display: none;
            margin-top: 20px;
        }
        
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #e1e5e9;
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 10px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            width: 0%;
            transition: width 0.3s ease;
        }
        
        .status {
            text-align: center;
            font-weight: 600;
            color: #667eea;
        }
        
        .results {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-top: 30px;
        }
        
        .results h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8rem;
        }
        
        .video-item {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
        }
        
        .video-item h3 {
            color: #333;
            margin-bottom: 10px;
        }
        
        .video-item p {
            color: #666;
            margin-bottom: 10px;
        }
        
        .video-item video {
            width: 100%;
            border-radius: 8px;
            margin-top: 10px;
        }
        
        .download-btn {
            background: #28a745;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
            margin-top: 10px;
        }
        
        .download-btn:hover {
            background: #218838;
        }
        
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            border-left: 4px solid #dc3545;
        }
        
        .success {
            background: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            border-left: 4px solid #28a745;
        }
        
        .system-status {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }
        
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        
        .status-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        
        .status-item .label {
            font-weight: 600;
            color: #555;
            margin-bottom: 5px;
        }
        
        .status-item .value {
            color: #667eea;
            font-size: 1.1rem;
        }
        
        .status-item .status-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        
        .status-badge.online {
            background: #d4edda;
            color: #155724;
        }
        
        .status-badge.offline {
            background: #f8d7da;
            color: #721c24;
        }
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 Real Video Generation</h1>
            <p>Generate actual video content from text prompts like Veo/Sora</p>
        </div>
        
        <div class="system-status">
            <h2>🔧 System Status</h2>
            <div class="status-grid" id="systemStatus">
                <div class="status-item">
                    <div class="label">API Status</div>
                    <div class="value">
                        <span class="status-badge" id="apiStatus">Checking...</span>
                    </div>
                </div>
                <div class="status-item">
                    <div class="label">Models Loaded</div>
                    <div class="value" id="modelsLoaded">-</div>
                </div>
                <div class="status-item">
                    <div class="label">Device</div>
                    <div class="value" id="device">-</div>
                </div>
                <div class="status-item">
                    <div class="label">Videos Generated</div>
                    <div class="value" id="videosCount">-</div>
                </div>
            </div>
        </div>
        
        <div class="main-content">
            <div class="card">
                <h2>💃 Dancing Video</h2>
                <p style="margin-bottom: 20px; color: #666;">Generate a person dancing video like Veo examples</p>
                
                <div class="form-group">
                    <label for="danceStyle">Dance Style:</label>
                    <select id="danceStyle">
                        <option value="modern dance">Modern Dance</option>
                        <option value="hip hop">Hip Hop</option>
                        <option value="ballet">Ballet</option>
                        <option value="breakdance">Breakdance</option>
                        <option value="salsa">Salsa</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="danceDuration">Duration (seconds):</label>
                    <input type="number" id="danceDuration" value="8" min="3" max="15">
                </div>
                
                <div class="form-group">
                    <label for="danceFPS">Frame Rate (FPS):</label>
                    <input type="number" id="danceFPS" value="24" min="8" max="30">
                </div>
                
                <button class="btn" onclick="generateDancingVideo()">🎬 Generate Dancing Video</button>
                
                <div class="progress-container" id="danceProgress">
                    <div class="progress-bar">
                        <div class="progress-fill" id="danceProgressFill"></div>
                    </div>
                    <div class="status" id="danceStatus">Preparing...</div>
                </div>
            </div>
            
            <div class="card">
                <h2>🎥 Scene Video</h2>
                <p style="margin-bottom: 20px; color: #666;">Generate scene-based video with camera motion</p>
                
                <div class="form-group">
                    <label for="sceneDescription">Scene Description:</label>
                    <textarea id="sceneDescription" placeholder="A beautiful sunset over mountains with gentle camera movement..."></textarea>
                </div>
                
                <div class="form-group">
                    <label for="motionType">Camera Motion:</label>
                    <select id="motionType">
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
                    <input type="number" id="sceneDuration" value="8" min="3" max="15">
                </div>
                
                <div class="form-group">
                    <label for="sceneFPS">Frame Rate (FPS):</label>
                    <input type="number" id="sceneFPS" value="24" min="8" max="30">
                </div>
                
                <button class="btn" onclick="generateSceneVideo()">🎬 Generate Scene Video</button>
                
                <div class="progress-container" id="sceneProgress">
                    <div class="progress-bar">
                        <div class="progress-fill" id="sceneProgressFill"></div>
                    </div>
                    <div class="status" id="sceneStatus">Preparing...</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>🎬 Custom Video Generation</h2>
            <p style="margin-bottom: 20px; color: #666;">Generate any video from text prompt</p>
            
            <div class="form-group">
                <label for="customPrompt">Video Prompt:</label>
                <textarea id="customPrompt" placeholder="A person walking through a magical forest with glowing butterflies..."></textarea>
            </div>
            
            <div class="form-group">
                <label for="customDuration">Duration (seconds):</label>
                <input type="number" id="customDuration" value="8" min="3" max="15">
            </div>
            
            <div class="form-group">
                <label for="customFPS">Frame Rate (FPS):</label>
                <input type="number" id="customFPS" value="24" min="8" max="30">
            </div>
            
            <button class="btn" onclick="generateCustomVideo()">🎬 Generate Custom Video</button>
            
            <div class="progress-container" id="customProgress">
                <div class="progress-bar">
                    <div class="progress-fill" id="customProgressFill"></div>
                </div>
                <div class="status" id="customStatus">Preparing...</div>
            </div>
        </div>
        
        <div class="results" id="results" style="display: none;">
            <h2>📹 Generated Videos</h2>
            <div id="videoList"></div>
        </div>
    </div>
    
    <script>
        const API_URL = 'http://localhost:5005';
        
        // Check system status on load
        window.onload = function() {
            checkSystemStatus();
            loadVideos();
        };
        
        async function checkSystemStatus() {
            try {
                const response = await fetch(`${API_URL}/api/real-video/status`);
                const data = await response.json();
                
                document.getElementById('apiStatus').textContent = 'Online';
                document.getElementById('apiStatus').className = 'status-badge online';
                document.getElementById('modelsLoaded').textContent = data.models_loaded;
                document.getElementById('device').textContent = data.device;
                
                // Get video count
                const videosResponse = await fetch(`${API_URL}/api/real-video/videos`);
                const videosData = await videosResponse.json();
                document.getElementById('videosCount').textContent = videosData.total || 0;
                
            } catch (error) {
                document.getElementById('apiStatus').textContent = 'Offline';
                document.getElementById('apiStatus').className = 'status-badge offline';
                console.error('Error checking system status:', error);
            }
        }
        
        async function generateDancingVideo() {
            const danceStyle = document.getElementById('danceStyle').value;
            const duration = parseInt(document.getElementById('danceDuration').value);
            const fps = parseInt(document.getElementById('danceFPS').value);
            
            showProgress('dance');
            
            try {
                const response = await fetch(`${API_URL}/api/real-video/generate-dancing`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        dance_style: danceStyle,
                        duration_seconds: duration,
                        fps: fps
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    hideProgress('dance');
                    showSuccess(`Dancing video generated successfully! Filename: ${result.filename}`);
                    loadVideos();
                } else {
                    hideProgress('dance');
                    showError(`Error: ${result.error}`);
                }
                
            } catch (error) {
                hideProgress('dance');
                showError(`Network error: ${error.message}`);
            }
        }
        
        async function generateSceneVideo() {
            const sceneDescription = document.getElementById('sceneDescription').value;
            const motionType = document.getElementById('motionType').value;
            const duration = parseInt(document.getElementById('sceneDuration').value);
            const fps = parseInt(document.getElementById('sceneFPS').value);
            
            if (!sceneDescription.trim()) {
                showError('Please enter a scene description');
                return;
            }
            
            showProgress('scene');
            
            try {
                const response = await fetch(`${API_URL}/api/real-video/generate-scene`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        scene_description: sceneDescription,
                        motion_type: motionType,
                        duration_seconds: duration,
                        fps: fps
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    hideProgress('scene');
                    showSuccess(`Scene video generated successfully! Filename: ${result.filename}`);
                    loadVideos();
                } else {
                    hideProgress('scene');
                    showError(`Error: ${result.error}`);
                }
                
            } catch (error) {
                hideProgress('scene');
                showError(`Network error: ${error.message}`);
            }
        }
        
        async function generateCustomVideo() {
            const prompt = document.getElementById('customPrompt').value;
            const duration = parseInt(document.getElementById('customDuration').value);
            const fps = parseInt(document.getElementById('customFPS').value);
            
            if (!prompt.trim()) {
                showError('Please enter a video prompt');
                return;
            }
            
            showProgress('custom');
            
            try {
                const response = await fetch(`${API_URL}/api/real-video/generate`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        prompt: prompt,
                        duration_seconds: duration,
                        fps: fps
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    hideProgress('custom');
                    showSuccess(`Custom video generated successfully! Filename: ${result.filename}`);
                    loadVideos();
                } else {
                    hideProgress('custom');
                    showError(`Error: ${result.error}`);
                }
                
            } catch (error) {
                hideProgress('custom');
                showError(`Network error: ${error.message}`);
            }
        }
        
        async function loadVideos() {
            try {
                const response = await fetch(`${API_URL}/api/real-video/videos`);
                const data = await response.json();
                
                if (data.success && data.videos.length > 0) {
                    const videoList = document.getElementById('videoList');
                    videoList.innerHTML = '';
                    
                    data.videos.forEach(video => {
                        const videoItem = document.createElement('div');
                        videoItem.className = 'video-item';
                        videoItem.innerHTML = `
                            <h3>${video.filename}</h3>
                            <p>Size: ${(video.size / 1024 / 1024).toFixed(2)} MB</p>
                            <p>Created: ${new Date(video.created * 1000).toLocaleString()}</p>
                            <video controls>
                                <source src="${API_URL}/api/real-video/download/${video.filename}" type="video/mp4">
                                Your browser does not support the video tag.
                            </video>
                            <br>
                            <a href="${API_URL}/api/real-video/download/${video.filename}" class="download-btn">📥 Download</a>
                        `;
                        videoList.appendChild(videoItem);
                    });
                    
                    document.getElementById('results').style.display = 'block';
                }
                
            } catch (error) {
                console.error('Error loading videos:', error);
            }
        }
        
        function showProgress(type) {
            const progressContainer = document.getElementById(`${type}Progress`);
            const progressFill = document.getElementById(`${type}ProgressFill`);
            const status = document.getElementById(`${type}Status`);
            
            progressContainer.style.display = 'block';
            progressFill.style.width = '0%';
            status.textContent = 'Starting generation...';
            
            // Simulate progress
            let progress = 0;
            const interval = setInterval(() => {
                progress += Math.random() * 15;
                if (progress > 90) progress = 90;
                progressFill.style.width = progress + '%';
                status.textContent = `Generating video... ${Math.round(progress)}%`;
            }, 1000);
            
            // Store interval for clearing
            progressContainer.dataset.interval = interval;
        }
        
        function hideProgress(type) {
            const progressContainer = document.getElementById(`${type}Progress`);
            const progressFill = document.getElementById(`${type}ProgressFill`);
            const status = document.getElementById(`${type}Status`);
            
            // Clear interval
            if (progressContainer.dataset.interval) {
                clearInterval(parseInt(progressContainer.dataset.interval));
            }
            
            progressFill.style.width = '100%';
            status.textContent = 'Complete!';
            
            setTimeout(() => {
                progressContainer.style.display = 'none';
            }, 2000);
        }
        
        function showSuccess(message) {
            const successDiv = document.createElement('div');
            successDiv.className = 'success';
            successDiv.textContent = message;
            document.querySelector('.container').appendChild(successDiv);
            
            setTimeout(() => {
                successDiv.remove();
            }, 5000);
        }
        
        function showError(message) {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error';
            errorDiv.textContent = message;
            document.querySelector('.container').appendChild(errorDiv);
            
            setTimeout(() => {
                errorDiv.remove();
            }, 5000);
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
    print("🌐 Real Video Generation Web Interface")
    print("=" * 50)
    print("🚀 Starting web interface on http://localhost:5006")
    print("🎯 Generate real videos from text prompts like Veo/Sora")
    print("📡 API Server: http://localhost:5005")
    print("")
    print("🎬 Video Generation Features:")
    print("   • Person dancing videos (modern, hip hop, ballet, breakdance, salsa)")
    print("   • Scene videos with camera motion")
    print("   • Custom text-to-video generation")
    print("   • Real video content, not just motion effects")
    print("")
    print("Press Ctrl+C to stop the web interface")
    
    app.run(host='0.0.0.0', port=5006, debug=True) 