from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from generate import generate_image
try:
    from generate_fast import generate_image as generate_image_fast
    FAST_MODEL_AVAILABLE = True
except ImportError:
    FAST_MODEL_AVAILABLE = False

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "AI Image Generator API is running"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "ai-image-generator"})

@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.json["prompt"]
    image_path = generate_image(prompt)
    return jsonify({"image_path": image_path})

@app.route("/generate_fast", methods=["POST"])
def generate_fast():
    if not FAST_MODEL_AVAILABLE:
        return jsonify({"error": "Fast model not available"}), 400
    
    prompt = request.json["prompt"]
    image_path = generate_image_fast(prompt)
    return jsonify({"image_path": image_path})

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    print("🚀 Starting AI Image Generator API on port 5001...")
    app.run(host="0.0.0.0", port=5001, debug=True) 