import os

# CRITICAL: Set these environment variables BEFORE any other imports
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
os.environ['OPENCV_IO_ENABLE_JASPER'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'

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
            
            model_url = "https://github.com/houscan1019/yolov8-render-api/releases/download/v1.0/weights.pt"
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
    """Load YOLO model with forced headless environment"""
    global model, model_status
    
    try:
        print("🔄 Loading YOLO model with headless environment...")
        model_status = "loading"
        
        # Import cv2 first with explicit headless setup
        print("🔧 Setting up headless OpenCV environment...")
        
        try:
            import cv2
            print(f"✅ OpenCV {cv2.__version__} imported successfully (headless)")
        except Exception as cv2_error:
            print(f"❌ OpenCV import failed: {cv2_error}")
            model_status = f"opencv_error: {str(cv2_error)}"
            return False
        
        # Now import ultralytics
        print("🔧 Importing ultralytics...")
        try:
            from ultralytics import YOLO
            print("✅ Ultralytics imported successfully")
        except Exception as ultralytics_error:
            print(f"❌ Ultralytics import failed: {ultralytics_error}")
            model_status = f"ultralytics_error: {str(ultralytics_error)}"
            return False
        
        # Load the model
        print("🔧 Loading YOLO model from weights.pt...")
        model = YOLO('weights.pt')
        print("✅ YOLO model loaded successfully")
        model_status = "loaded"
        return True
        
    except Exception as e:
        print(f"❌ Error in load_yolo_model: {e}")
        model_status = f"general_error: {str(e)}"
        return False

# Initialize model on startup
print("🚀 Starting model initialization with headless environment...")
if download_model_from_release():
    load_yolo_model()
else:
    print("❌ Model initialization failed")

@app.route('/')
def home():
    return jsonify({
        "message": "Railway YOLO API - Headless Environment", 
        "status": "running",
        "port": PORT,
        "model_status": model_status,
        "model_loaded": model is not None,
        "environment": "headless_forced"
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
    return f"✅ Railway YOLO API (Headless) - Model Status: {model_status}"

@app.route('/debug-env')
def debug_env():
    """Debug endpoint to check environment variables"""
    env_vars = {
        'OPENCV_IO_ENABLE_OPENEXR': os.environ.get('OPENCV_IO_ENABLE_OPENEXR'),
        'QT_QPA_PLATFORM': os.environ.get('QT_QPA_PLATFORM'),
        'MPLBACKEND': os.environ.get('MPLBACKEND'),
        'DISPLAY': os.environ.get('DISPLAY'),
    }
    
    # Test cv2 import
    cv2_test = "not_tested"
    try:
        import cv2
        cv2_test = f"success_v{cv2.__version__}"
    except Exception as e:
        cv2_test = f"failed: {str(e)}"
    
    return jsonify({
        "environment_variables": env_vars,
        "cv2_import_test": cv2_test,
        "python_version": sys.version
    }), 200

@app.route('/model-info')
def model_info():
    """Detailed model information"""
    return jsonify({
        "model_loaded": model is not None,
        "model_status": model_status,
        "weights_file_exists": os.path.exists('weights.pt'),
        "weights_file_size": os.path.getsize('weights.pt') if os.path.exists('weights.pt') else 0,
        "model_source": "Custom trained YOLOv8 instance segmentation",
        "github_release": "v1.0",
        "environment_type": "forced_headless"
    }), 200

@app.route('/process-image', methods=['POST'])
def process_image():
    """Image processing endpoint"""
    try:
        if not model:
            return jsonify({
                "error": "YOLO model not loaded",
                "model_status": model_status
            }), 500
            
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Decode and validate image
        try:
            image_data = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(image_data))
            image_np = np.array(image)
        except Exception as e:
            return jsonify({"error": f"Invalid image data: {str(e)}"}), 400
        
        # Convert RGB to BGR for OpenCV processing
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            import cv2
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_cv = image_np
        
        print(f"Processing image with shape: {image_cv.shape}")
        
        # Run YOLO inference
        results = model(image_cv, conf=0.5)
        
        # Extract detection information
        detection_count = 0
        confidence_scores = []
        has_masks = False
        
        if results and len(results) > 0:
            result = results[0]
            
            if hasattr(result, 'masks') and result.masks is not None:
                has_masks = True
                detection_count = len(result.masks)
                if hasattr(result, 'boxes') and result.boxes is not None:
                    confidence_scores = [float(conf) for conf in result.boxes.conf.cpu().numpy()]
            elif hasattr(result, 'boxes') and result.boxes is not None:
                detection_count = len(result.boxes)
                confidence_scores = [float(conf) for conf in result.boxes.conf.cpu().numpy()]
        
        return jsonify({
            "success": True,
            "message": "Image processed with custom YOLO model (headless)",
            "detections": detection_count,
            "confidence_scores": confidence_scores,
            "has_instance_segmentation": has_masks,
            "model_type": "Custom YOLOv8 instance segmentation",
            "image_shape": image_np.shape
        }), 200
        
    except Exception as e:
        print(f"Error in process_image: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print(f"🚀 Starting Flask app on port {PORT}")
    print(f"🤖 Model status: {model_status}")
    print(f"🖥️ Environment: Forced headless mode")
    app.run(host='0.0.0.0', port=PORT, debug=False)
