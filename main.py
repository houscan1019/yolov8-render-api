import os
from flask import Flask, jsonify, request
import base64
import io
import urllib.request
from PIL import Image
import numpy as np

app = Flask(__name__)

# Railway port configuration
PORT = int(os.environ.get('PORT', 5000))
APP_VERSION = "YOLOv11-v1.0"

# Global model variable
model = None
model_status = "not_loaded"

def download_yolov11_model():
    """Download YOLOv11 model from GitHub release"""
    global model_status
    model_path = 'weights.pt'
    
    try:
        if not os.path.exists(model_path):
            print("📥 Downloading YOLOv11 model from GitHub release...")
            model_status = "downloading"
            
            # Your YOLOv11 model from v1.0 release
            model_url = "https://github.com/houscan1019/yolov8-render-api/releases/download/v1.0/weights.pt"
            urllib.request.urlretrieve(model_url, model_path)
            print("✅ YOLOv11 model downloaded successfully")
            model_status = "downloaded"
        else:
            print("✅ YOLOv11 model already exists")
            model_status = "exists"
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading YOLOv11 model: {e}")
        model_status = f"download_failed: {str(e)}"
        return False

def load_yolov11_model():
    """Load YOLOv11 model with minimal dependencies"""
    global model, model_status
    
    try:
        print("🔄 Loading YOLOv11 model...")
        model_status = "loading"
        
        from ultralytics import YOLO
        
        # Load your custom YOLOv11 model
        model = YOLO('weights.pt')
        print("✅ YOLOv11 model loaded successfully")
        model_status = "loaded_yolov11"
        return True
        
    except Exception as e:
        print(f"❌ Error loading YOLOv11 model: {e}")
        model_status = f"load_error: {str(e)}"
        return False

# Initialize model on startup
print("🚀 Starting YOLOv11 model initialization...")
if download_yolov11_model():
    load_yolov11_model()
else:
    print("❌ YOLOv11 model initialization failed")

@app.route('/')
def home():
    return jsonify({
        "message": "Railway YOLOv11 Document Processing API", 
        "status": "running",
        "port": PORT,
        "model_status": model_status,
        "model_loaded": model is not None,
        "app_version": APP_VERSION,
        "model_type": "YOLOv11 Instance Segmentation (Custom Trained)"
    }), 200

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy", 
        "service": "document-processor",
        "model_status": model_status,
        "model_ready": model is not None,
        "app_version": APP_VERSION,
        "model_version": "YOLOv11"
    }), 200

@app.route('/test')
def test():
    return f"✅ Railway YOLOv11 API ({APP_VERSION}) - Model Status: {model_status}"

@app.route('/model-info')
def model_info():
    return jsonify({
        "app_version": APP_VERSION,
        "deployment_type": "nixpacks",
        "model_loaded": model is not None,
        "model_status": model_status,
        "weights_file_exists": os.path.exists('weights.pt'),
        "weights_file_size": os.path.getsize('weights.pt') if os.path.exists('weights.pt') else 0,
        "model_type": "YOLOv11 Instance Segmentation",
        "model_source": "Custom trained in Roboflow",
        "github_release": "v1.0",
        "advantages": "Better accuracy than YOLOv8, latest architecture"
    }), 200

@app.route('/process-image', methods=['POST'])
def process_image():
    """Document processing with YOLOv11 model"""
    try:
        if not model:
            return jsonify({
                "error": "YOLOv11 model not loaded",
                "model_status": model_status
            }), 500
            
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Decode image
        try:
            image_data = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(image_data))
            image_np = np.array(image)
        except Exception as e:
            return jsonify({"error": f"Invalid image data: {str(e)}"}), 400
        
        print(f"Processing image with YOLOv11: {image_np.shape}")
        
        # Run YOLOv11 inference (PIL image works directly with ultralytics)
        results = model(image, conf=0.5)
        
        # Extract detection information
        detection_count = 0
        confidence_scores = []
        has_masks = False
        
        if results and len(results) > 0:
            result = results[0]
            
            # Check for instance segmentation masks
            if hasattr(result, 'masks') and result.masks is not None:
                has_masks = True
                detection_count = len(result.masks)
                if hasattr(result, 'boxes') and result.boxes is not None:
                    confidence_scores = [float(conf) for conf in result.boxes.conf.cpu().numpy()]
            
            # Fallback to bounding boxes
            elif hasattr(result, 'boxes') and result.boxes is not None:
                detection_count = len(result.boxes)
                confidence_scores = [float(conf) for conf in result.boxes.conf.cpu().numpy()]
        
        return jsonify({
            "success": True,
            "message": "Document processed with YOLOv11 model",
            "detections": detection_count,
            "confidence_scores": confidence_scores,
            "has_instance_segmentation": has_masks,
            "model_type": "YOLOv11 Instance Segmentation (Custom)",
            "image_shape": image_np.shape,
            "app_version": APP_VERSION,
            "model_performance": "Enhanced accuracy with YOLOv11"
        }), 200
        
    except Exception as e:
        print(f"Error in YOLOv11 processing: {e}")
        return jsonify({"error": str(e)}), 500

# Create static directory for future use
os.makedirs('static/processed', exist_ok=True)

if __name__ == '__main__':
    print(f"🚀 Starting YOLOv11 Flask app on port {PORT}")
    print(f"🤖 Model status: {model_status}")
    print(f"🆕 YOLOv11 Document Processing API")
    app.run(host='0.0.0.0', port=PORT, debug=False)
