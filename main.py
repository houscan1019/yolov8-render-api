import os
from flask import Flask, request, jsonify

app = Flask(__name__)

print("[INFO] Starting Document Processing API...")

# Global variables
model = None
model_error = None

def download_weights():
    """Download weights file if it doesn't exist"""
    if os.path.exists("weights.pt"):
        print("[INFO] weights.pt already exists")
        return True
        
    weights_url = os.getenv('WEIGHTS_URL')
    if not weights_url:
        print("[ERROR] No WEIGHTS_URL environment variable set")
        return False
        
    try:
        import requests
        print(f"[INFO] Downloading weights from GitHub...")
        response = requests.get(weights_url, stream=True, timeout=300)
        response.raise_for_status()
        
        with open('weights.pt', 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print("[SUCCESS] Weights downloaded successfully")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to download weights: {e}")
        return False

def load_model():
    """Try to load the YOLO model"""
    global model, model_error
    
    try:
        if not download_weights():
            model_error = "Failed to download weights.pt"
            return False
            
        print("[INFO] Loading YOLO model...")
        from ultralytics import YOLO
        model = YOLO("weights.pt")
        print("[SUCCESS] Your trained model loaded successfully!")
        return True
        
    except ImportError as e:
        model_error = f"Failed to import ultralytics: {e}"
        print(f"[ERROR] {model_error}")
        return False
        
    except Exception as e:
        model_error = f"Model loading failed: {e}"
        print(f"[ERROR] {model_error}")
        return False

# Try to load model on startup
print("[INFO] Attempting to load model...")
load_model()

@app.route('/')
def home():
    return jsonify({
        'message': 'Document Processing API',
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_error': model_error,
        'instructions': 'Use POST /detect-and-process to process documents'
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_error': model_error,
        'weights_exist': os.path.exists('weights.pt'),
        'weights_url_configured': os.getenv('WEIGHTS_URL') is not None
    })

@app.route('/test-model')
def test_model():
    """Simple endpoint to test if model is working"""
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'details': model_error
        }), 500
    
    return jsonify({
        'message': 'Model is loaded and ready!',
        'model_type': str(type(model)),
        'model_loaded': True
    })

@app.route('/detect-and-process', methods=['POST'])
def detect_and_process():
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'details': model_error,
            'suggestion': 'Check /health endpoint for details'
        }), 500
        
    return jsonify({
        'message': 'Model is working! File upload processing coming soon...',
        'model_loaded': True,
        'status': 'ready'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"[INFO] Starting Flask app on port {port}")
    print(f"[INFO] Model status: {'Loaded' if model else 'Failed to load'}")
    
    # Run the app
    app.run(
        debug=False, 
        host='0.0.0.0', 
        port=port,
        threaded=True
    )
