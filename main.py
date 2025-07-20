import os
from flask import Flask, jsonify, request
import base64
import io
import urllib.request
from PIL import Image
import numpy as np

# Set environment variables before any imports
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'

app = Flask(__name__)

# Railway port configuration
PORT = int(os.environ.get('PORT', 5000))
APP_VERSION = "Pure-Ultralytics-v1.0"

# Global model variable
model = None
model_status = "not_loaded"

def download_model():
    """Download model from GitHub release"""
    global model_status
    model_path = 'weights.pt'
    
    try:
        if not os.path.exists(model_path):
            print("📥 Downloading model from GitHub release...")
            model_status = "downloading"
            
            model_url = "https://github.com/houscan1019/yolov8-render-api/releases/download/v1.0/weights.pt"
            urllib.request.urlretrieve(model_url, model_path)
            print("✅ Model downloaded successfully")
            model_status = "downloaded"
        else:
            print("✅ Model already exists")
            model_status = "exists"
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        model_status = f"download_failed: {str(e)}"
        return False

def load_model():
    """Load model with pure ultralytics approach"""
    global model, model_status
    
    try:
        print("🔄 Loading model with pure ultralytics...")
        model_status = "loading"
        
        # Try to import ultralytics without triggering OpenCV
        try:
            import ultralytics
            print(f"✅ Ultralytics {ultralytics.__version__} imported")
            
            # Import YOLO class
            from ultralytics import YOLO
            print("✅ YOLO class imported successfully")
            
        except Exception as import_error:
            print(f"❌ Ultralytics import failed: {import_error}")
            model_status = f"import_error: {str(import_error)}"
            return False
        
        # Load the model
        print("🔧 Loading model weights...")
        try:
            model = YOLO('weights.pt')
            print("✅ Model loaded successfully with pure ultralytics")
            model_status = "loaded_pure_ultralytics"
            return True
        except Exception as load_error:
            print(f"❌ Model loading failed: {load_error}")
            model_status = f"load_error: {str(load_error)}"
            return False
        
    except Exception as e:
        print(f"❌ General error in load_model: {e}")
        model_status = f"general_error: {str(e)}"
        return False

# Initialize model on startup
print("🚀 Starting pure ultralytics initialization...")
if download_model():
    load_model()
else:
    print("❌ Model initialization failed")

@app.route('/')
def home():
    return jsonify({
        "message": "Railway Pure Ultralytics API", 
        "status": "running",
        "port": PORT,
        "model_status": model_status,
        "model_loaded": model is not None,
        "app_version": APP_VERSION,
        "approach": "Pure ultralytics - no OpenCV operations"
    }), 200

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy", 
        "service": "document-processor",
        "model_status": model_status,
        "model_ready": model is not None,
        "app_version": APP_VERSION
    }), 200

@app.route('/test')
def test():
    return f"✅ Railway Pure Ultralytics API ({APP_VERSION}) - Model Status: {model_status}"

@app.route('/debug')
def debug():
    """Debug ultralytics installation"""
    debug_info = {
        "app_version": APP_VERSION,
        "approach": "pure_ultralytics"
    }
    
    # Test ultralytics import
    try:
        import ultralytics
        debug_info["ultralytics_version"] = ultralytics.__version__
        debug_info["ultralytics_import"] = "success"
    except Exception as e:
        debug_info["ultralytics_import"] = f"failed: {str(e)}"
    
    # Test YOLO import
    try:
        from ultralytics import YOLO
        debug_info["yolo_import"] = "success"
    except Exception as e:
        debug_info["yolo_import"] = f"failed: {str(e)}"
    
    return jsonify(debug_info), 200

@app.route('/model-info')
def model_info():
    return jsonify({
        "app_version": APP_VERSION,
        "deployment_type": "nixpacks",
        "approach": "pure_ultralytics",
        "model_loaded": model is not None,
        "model_status": model_status,
        "weights_file_exists": os.path.exists('weights.pt'),
        "weights_file_size": os.path.getsize('weights.pt') if os.path.exists('weights.pt') else 0,
        "model_type": "Custom Instance Segmentation",
        "model_source": "Trained model from Roboflow",
        "github_release": "v1.0",
        "note": "Using pure ultralytics approach to avoid OpenCV"
    }), 200

@app.route('/process-image', methods=['POST'])
def process_image():
    """Document processing with pure ultralytics"""
    try:
        if not model:
            return jsonify({
                "error": "Model not loaded",
                "model_status": model_status
            }), 500
            
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Decode image using PIL only
        try:
            image_data = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            print(f"Processing image: {image.size}")
        except Exception as e:
            return jsonify({"error": f"Invalid image data: {str(e)}"}), 400
        
        # Run inference with PIL image directly
        try:
            results = model(image, conf=0.5)
            print("✅ Inference completed successfully")
        except Exception as e:
            return jsonify({"error": f"Inference failed: {str(e)}"}), 500
        
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
                print(f"Found {detection_count} masks")
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    confidence_scores = [float(conf) for conf in result.boxes.conf.cpu().numpy()]
            
            # Fallback to bounding boxes
            elif hasattr(result, 'boxes') and result.boxes is not None:
                detection_count = len(result.boxes)
                confidence_scores = [float(conf) for conf in result.boxes.conf.cpu().numpy()]
                print(f"Found {detection_count} boxes")
        
        return jsonify({
            "success": True,
            "message": "Document processed with pure ultralytics",
            "detections": detection_count,
            "confidence_scores": confidence_scores,
            "has_instance_segmentation": has_masks,
            "model_type": "Custom Instance Segmentation",
            "image_size": image.size,
            "app_version": APP_VERSION,
            "processing_method": "Pure ultralytics - no OpenCV"
        }), 200
        
    except Exception as e:
        print(f"Error in process_image: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print(f"🚀 Starting Pure Ultralytics Flask app on port {PORT}")
    print(f"🤖 Model status: {model_status}")
    print(f"🔧 Approach: Pure ultralytics without OpenCV operations")
    app.run(host='0.0.0.0', port=PORT, debug=False)
