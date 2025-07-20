import os
import sys
import urllib.request
from flask import Flask, jsonify, request
import base64
import io
from PIL import Image
import numpy as np

app = Flask(__name__)

# Railway port configuration
PORT = int(os.environ.get('PORT', 5000))

# Global model variable
model = None
model_status = "not_loaded"

def download_model_from_release():
    """Download YOLO model from GitHub release if not present"""
    global model_status
    model_path = 'weights.pt'
    
    try:
        if not os.path.exists(model_path):
            print("📥 Downloading YOLO model from GitHub release...")
            model_status = "downloading"
            
            model_url = "https://github.com/houscani019/yolov8-render-api/releases/download/v1.0/weights.pt"
            urllib.request.urlretrieve(model_url, model_path)
            print("✅ YOLO model downloaded successfully")
            model_status = "downloaded"
        else:
            print("✅ YOLO model already exists")
            model_status = "exists"
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        model_status = f"download_failed: {str(e)}"
        return False

def load_yolo_model():
    """Load YOLO model with error handling"""
    global model, model_status
    
    try:
        print("🔄 Loading YOLO model...")
        model_status = "loading"
        
        # Import ultralytics here to catch import errors
        from ultralytics import YOLO
        
        model = YOLO('weights.pt')
        print("✅ YOLO model loaded successfully")
        model_status = "loaded"
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        model_status = f"import_error: {str(e)}"
        return False
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        model_status = f"load_error: {str(e)}"
        return False

# Initialize model on startup
print("🚀 Starting model initialization...")
if download_model_from_release():
    load_yolo_model()
else:
    print("❌ Model initialization failed")

@app.route('/')
def home():
    return jsonify({
        "message": "Railway YOLO API", 
        "status": "running",
        "port": PORT,
        "model_status": model_status,
        "model_loaded": model is not None
    }), 200

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy", 
        "service": "document-processor",
        "model_status": model_status,
        "model_ready": model is not None
    }), 200

@app.route('/test')
def test():
    return f"✅ Railway YOLO API - Model Status: {model_status}"

@app.route('/model-info')
def model_info():
    """Detailed model information"""
    return jsonify({
        "model_loaded": model is not None,
        "model_status": model_status,
        "weights_file_exists": os.path.exists('weights.pt'),
        "weights_file_size": os.path.getsize('weights.pt') if os.path.exists('weights.pt') else 0
    }), 200

@app.route('/process-image', methods=['POST'])
def process_image():
    """Basic image processing endpoint"""
    try:
        if not model:
            return jsonify({
                "error": "YOLO model not loaded",
                "model_status": model_status
            }), 500
            
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Basic image validation
        try:
            image_data = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(image_data))
            image_np = np.array(image)
        except Exception as e:
            return jsonify({"error": f"Invalid image data: {str(e)}"}), 400
        
        # Simple response for now (no actual processing yet)
        return jsonify({
            "success": True,
            "message": "Image received successfully",
            "image_shape": image_np.shape,
            "model_status": model_status,
            "note": "Full processing will be added next"
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print(f"🚀 Starting Flask app on port {PORT}")
    print(f"🤖 Model status: {model_status}")
    app.run(host='0.0.0.0', port=PORT, debug=False)
