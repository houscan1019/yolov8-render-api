import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import uuid
import gc
import torch

app = Flask(__name__)
CORS(app)

# Railway port configuration
PORT = int(os.environ.get('PORT', 5000))

# Create static directory for processed images
static_dir = os.path.join(os.getcwd(), 'static', 'processed')
os.makedirs(static_dir, exist_ok=True)

# Download and load YOLO model
import urllib.request

def download_model_from_release():
    """Download YOLO model from GitHub release if not present"""
    model_path = 'weights.pt'
    
    if not os.path.exists(model_path):
        print("📥 Downloading YOLO model from GitHub release...")
        try:
            # Your GitHub release URL for weights.pt
            model_url = "https://github.com/houscani019/yolov8-render-api/releases/download/v1.0/weights.pt"
            urllib.request.urlretrieve(model_url, model_path)
            print("✅ YOLO model downloaded successfully")
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            return False
    else:
        print("✅ YOLO model already exists")
    
    return True

# Download model and load
try:
    if download_model_from_release():
        model = YOLO('weights.pt')
        print("✅ YOLO model loaded successfully")
    else:
        model = None
        print("❌ Failed to download/load YOLO model")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    model = None

@app.route('/')
def home():
    return jsonify({
        "message": "YOLOv8 Document Processing API", 
        "status": "running",
        "model_loaded": model is not None
    }), 200

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy", 
        "service": "document-processor",
        "model_status": "loaded" if model else "failed"
    }), 200

@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        if not model:
            return jsonify({"error": "YOLO model not loaded"}), 500
            
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_np.shape) == 3:
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_cv = image_np
        
        # Run YOLO inference
        results = model(image_cv)
        
        # Process results (your existing logic here)
        processed_image = process_yolo_results(image_cv, results)
        
        # Save processed image
        filename = f"processed_{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(static_dir, filename)
        cv2.imwrite(filepath, processed_image)
        
        # Generate URL
        base_url = request.url_root.rstrip('/')
        image_url = f"{base_url}/static/processed/{filename}"
        
        return jsonify({
            "success": True,
            "processed_image_url": image_url,
            "filename": filename
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def process_yolo_results(image, results):
    """
    Process YOLO results - add your existing processing logic here
    """
    # Placeholder - replace with your actual processing code
    return image

@app.route('/static/processed/<filename>')
def serve_processed_image(filename):
    return send_from_directory(static_dir, filename)

@app.after_request
def cleanup(response):
    """Clean up memory after each request"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return response

if __name__ == '__main__':
    print(f"🚀 Starting Flask app on port {PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=False)
