#!/usr/bin/env python3
"""
Working Video Generation Web Interface
Simple web interface to test both pipeline options
"""

from flask import Flask, render_template_string, request, jsonify
import requests
import base64
from PIL import Image
import io

app = Flask(__name__)

# API server URL
API_URL = "http://localhost:5003"

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎬 Video Generation - Both Pipeline Options</title>
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
        .pipeline-options {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        .pipeline-card {
            background: #f7fafc;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            padding: 25px;
            transition: all 0.3s ease;
        }
        .pipeline-card:hover {
            border-color: #667eea;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .pipeline-card h3 {
            color: #2d3748;
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        .pipeline-card p {
            color: #4a5568;
            margin-bottom: 20px;
            line-height: 1.6;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #2d3748;
        }
        input, textarea, select {
            width: 100%;
            padding: 10px;
            border: 2px solid #e2e8f0;
            border-radius: 5px;
            font-size: 14px;
            transition: border-color 0.3s ease;
        }
        input:focus, textarea:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
        }
        button:hover {
            transform: translateY(-1px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        button:disabled {
            background: #cbd5e0;
            cursor: not-allowed;
            transform: none;
        }
        .status {
            margin-top: 20px;
            padding: 15px;
            border-radius: 5px;
            font-weight: 600;
        }
        .status.success {
            background: #c6f6d5;
            color: #22543d;
            border: 1px solid #9ae6b4;
        }
        .status.error {
            background: #fed7d7;
            color: #742a2a;
            border: 1px solid #feb2b2;
        }
        .status.info {
            background: #bee3f8;
            color: #2a4365;
            border: 1px solid #90cdf4;
        }
        .results {
            margin-top: 30px;
            padding: 20px;
            background: #f7fafc;
            border-radius: 10px;
            border: 2px solid #e2e8f0;
        }
        .video-result {
            margin-top: 15px;
            padding: 15px;
            background: white;
            border-radius: 5px;
            border: 1px solid #e2e8f0;
        }
        .hybrid-section {
            grid-column: 1 / -1;
            background: #f0fff4;
            border: 2px solid #9ae6b4;
        }
        .hybrid-section h3 {
            color: #22543d;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 Video Generation System</h1>
        <p style="text-align: center; color: #4a5568; margin-bottom: 30px;">
            Generate videos using both pipeline options for maximum flexibility
        </p>
        
        <div class="pipeline-options">
            <!-- Option A: Image → Motion → Video -->
            <div class="pipeline-card">
                <h3>🔁 Option A: Image → Motion → Video</h3>
                <p><strong>Best for:</strong> Controlled, realistic motion<br>
                <strong>Use case:</strong> Professional portraits, product demos, landscapes</p>
                
                <form id="optionAForm">
                    <div class="form-group">
                        <label for="imageFile">Upload Image:</label>
                        <input type="file" id="imageFile" accept="image/*" required>
                    </div>
                    <div class="form-group">
                        <label for="motionPrompt">Motion Type:</label>
                        <select id="motionPrompt">
                            <option value="gentle movement">Gentle Movement</option>
                            <option value="zoom">Zoom Effect</option>
                            <option value="pan">Pan Effect</option>
                            <option value="rotate">Rotation</option>
                            <option value="wave">Wave Effect</option>
                            <option value="pulse">Pulse Effect</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="durationA">Duration (seconds):</label>
                        <input type="number" id="durationA" value="5" min="1" max="30">
                    </div>
                    <button type="submit">Generate Video (Option A)</button>
                </form>
                
                <div id="statusA" class="status" style="display: none;"></div>
            </div>
            
            <!-- Option B: Text to Video Direct -->
            <div class="pipeline-card">
                <h3>🎥 Option B: Text to Video Direct</h3>
                <p><strong>Best for:</strong> Creative, dynamic content<br>
                <strong>Use case:</strong> Artistic scenes, animations, storytelling</p>
                
                <form id="optionBForm">
                    <div class="form-group">
                        <label for="prompt">Video Prompt:</label>
                        <textarea id="prompt" rows="3" placeholder="Describe the video you want to generate..." required></textarea>
                    </div>
                    <div class="form-group">
                        <label for="durationB">Duration (seconds):</label>
                        <input type="number" id="durationB" value="5" min="1" max="30">
                    </div>
                    <button type="submit">Generate Video (Option B)</button>
                </form>
                
                <div id="statusB" class="status" style="display: none;"></div>
            </div>
            
            <!-- Hybrid Option: Both Pipelines -->
            <div class="pipeline-card hybrid-section">
                <h3>🔄 Hybrid: Use Both Pipelines</h3>
                <p><strong>Generate with both options and compare results</strong></p>
                
                <form id="hybridForm">
                    <div class="form-group">
                        <label for="hybridPrompt">Video Prompt:</label>
                        <textarea id="hybridPrompt" rows="2" placeholder="Describe the video you want to generate..." required></textarea>
                    </div>
                    <div class="form-group">
                        <label for="hybridImageFile">Upload Image (Optional):</label>
                        <input type="file" id="hybridImageFile" accept="image/*">
                    </div>
                    <div class="form-group">
                        <label for="hybridMotion">Motion Type:</label>
                        <select id="hybridMotion">
                            <option value="gentle movement">Gentle Movement</option>
                            <option value="zoom">Zoom Effect</option>
                            <option value="pan">Pan Effect</option>
                            <option value="rotate">Rotation</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="hybridDuration">Duration (seconds):</label>
                        <input type="number" id="hybridDuration" value="5" min="1" max="30">
                    </div>
                    <button type="submit">Generate Both Videos</button>
                </form>
                
                <div id="statusHybrid" class="status" style="display: none;"></div>
            </div>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Generating video... Please wait...</p>
        </div>
        
        <div class="results" id="results" style="display: none;">
            <h3>🎬 Generated Videos</h3>
            <div id="videoResults"></div>
        </div>
    </div>

    <script>
        // Show status message
        function showStatus(elementId, message, type) {
            const element = document.getElementById(elementId);
            element.textContent = message;
            element.className = `status ${type}`;
            element.style.display = 'block';
        }
        
        // Hide status message
        function hideStatus(elementId) {
            document.getElementById(elementId).style.display = 'none';
        }
        
        // Show/hide loading
        function showLoading() {
            document.getElementById('loading').style.display = 'block';
        }
        
        function hideLoading() {
            document.getElementById('loading').style.display = 'none';
        }
        
        // Convert image to base64
        function imageToBase64(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve(reader.result);
                reader.onerror = reject;
                reader.readAsDataURL(file);
            });
        }
        
        // Add video result
        function addVideoResult(pipeline, videoPath, filename) {
            const resultsDiv = document.getElementById('videoResults');
            const videoDiv = document.createElement('div');
            videoDiv.className = 'video-result';
            videoDiv.innerHTML = `
                <h4>${pipeline}</h4>
                <p><strong>File:</strong> ${filename}</p>
                <p><strong>Path:</strong> ${videoPath}</p>
                <a href="http://localhost:5003/api/video/download/${filename}" target="_blank">
                    <button style="width: auto; margin-top: 10px;">Download Video</button>
                </a>
            `;
            resultsDiv.appendChild(videoDiv);
        }
        
        // Option A: Image to Video
        document.getElementById('optionAForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const imageFile = document.getElementById('imageFile').files[0];
            const motionPrompt = document.getElementById('motionPrompt').value;
            const duration = document.getElementById('durationA').value;
            
            if (!imageFile) {
                showStatus('statusA', 'Please select an image file', 'error');
                return;
            }
            
            try {
                showLoading();
                hideStatus('statusA');
                
                const imageData = await imageToBase64(imageFile);
                
                const response = await fetch(`http://localhost:5003/api/video/generate-from-image`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        image: imageData,
                        motion_prompt: motionPrompt,
                        duration_seconds: parseInt(duration),
                        fps: 24,
                        width: 512,
                        height: 512
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showStatus('statusA', `✅ Video generated successfully!`, 'success');
                    addVideoResult('Option A (Image → Video)', result.video_path, result.filename);
                    document.getElementById('results').style.display = 'block';
                } else {
                    showStatus('statusA', `❌ Error: ${result.error}`, 'error');
                }
            } catch (error) {
                showStatus('statusA', `❌ Error: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        });
        
        // Option B: Direct Text to Video
        document.getElementById('optionBForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const prompt = document.getElementById('prompt').value;
            const duration = document.getElementById('durationB').value;
            
            if (!prompt) {
                showStatus('statusB', 'Please enter a video prompt', 'error');
                return;
            }
            
            try {
                showLoading();
                hideStatus('statusB');
                
                const response = await fetch(`http://localhost:5003/api/video/generate-direct`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        prompt: prompt,
                        duration_seconds: parseInt(duration),
                        fps: 24,
                        width: 512,
                        height: 512
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showStatus('statusB', `✅ Video generated successfully!`, 'success');
                    addVideoResult('Option B (Direct)', result.video_path, result.filename);
                    document.getElementById('results').style.display = 'block';
                } else {
                    showStatus('statusB', `❌ Error: ${result.error}`, 'error');
                }
            } catch (error) {
                showStatus('statusB', `❌ Error: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        });
        
        // Hybrid: Both Pipelines
        document.getElementById('hybridForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const prompt = document.getElementById('hybridPrompt').value;
            const imageFile = document.getElementById('hybridImageFile').files[0];
            const motionPrompt = document.getElementById('hybridMotion').value;
            const duration = document.getElementById('hybridDuration').value;
            
            if (!prompt) {
                showStatus('statusHybrid', 'Please enter a video prompt', 'error');
                return;
            }
            
            try {
                showLoading();
                hideStatus('statusHybrid');
                
                const requestData = {
                    prompt: prompt,
                    motion_prompt: motionPrompt,
                    duration_seconds: parseInt(duration),
                    fps: 24,
                    width: 512,
                    height: 512
                };
                
                if (imageFile) {
                    requestData.image = await imageToBase64(imageFile);
                }
                
                const response = await fetch(`http://localhost:5003/api/video/generate-both`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(requestData)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showStatus('statusHybrid', `✅ Both pipeline options attempted!`, 'success');
                    
                    // Clear previous results
                    document.getElementById('videoResults').innerHTML = '';
                    
                    // Add results
                    if (result.results.option_a && result.results.option_a.success) {
                        addVideoResult('Option A (Image → Video)', 
                                     result.results.option_a.video_path, 
                                     result.results.option_a.filename);
                    }
                    
                    if (result.results.option_b && result.results.option_b.success) {
                        addVideoResult('Option B (Direct)', 
                                     result.results.option_b.video_path, 
                                     result.results.option_b.filename);
                    }
                    
                    document.getElementById('results').style.display = 'block';
                } else {
                    showStatus('statusHybrid', `❌ Error: ${result.error}`, 'error');
                }
            } catch (error) {
                showStatus('statusHybrid', `❌ Error: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    print("🌐 Working Video Generation Web Interface")
    print("=" * 50)
    print("🚀 Starting web interface on http://localhost:5004")
    print("")
    print("🎯 Features:")
    print("   • Option A: Image → Motion → Video (Best for control)")
    print("   • Option B: Text to Video Direct (Best for creativity)")
    print("   • Hybrid: Use both pipelines together")
    print("   • Download generated videos")
    print("")
    print("🔗 API Server: http://localhost:5003")
    print("")
    print("Press Ctrl+C to stop the web interface")
    print("")
    
    app.run(host='0.0.0.0', port=5004, debug=True) 