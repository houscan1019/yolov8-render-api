import os
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Download weights if they don't exist
def download_weights():
    if not os.path.exists("weights.pt"):
        weights_url = os.getenv('WEIGHTS_URL')
        if weights_url:
            try:
                import requests
                print(f"[INFO] Downloading weights from {weights_url}")
                response = requests.get(weights_url, stream=True, timeout=300)
                response.raise_for_status()
                
                with open('weights.pt', 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print("[SUCCESS] Weights downloaded successfully")
                return True
            except Exception as e:
                print(f"[ERROR] Download failed: {e}")
                return False
        else:
            print("[INFO] No WEIGHTS_URL provided")
            return False
    return True

# Try to load model
model = None
model_error = None

try:
    if download_weights():
        print("[INFO] Attempting to load YOLO model...")
        from ultralytics import YOLO
        model = YOLO("weights.pt")
        print("[SUCCESS] Your trained model loaded successfully!")
    else:
        model_error = "weights.pt download failed"
except Exception as e:
    model_error = str(e)
    print(f"[ERROR] Model loading failed: {e}")

@app.route('/')
def home():
    return jsonify({
        'message': 'Document Processing API',
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_error': model_error,
        'endpoints': {
            'detect': '/detect-and-process (POST)',
            'health': '/health (GET)'
        }
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

@app.route('/detect-and-process', methods=['POST'])
def detect_and_process():
    if model is None:
        return jsonify({
            'error': 'Model not loaded', 
            'details': model_error
        }), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, f"raw_{filename}")
        file.save(file_path)
        
        # Try basic model prediction
        print(f"[INFO] Processing image: {filename}")
        
        # Simple prediction with your trained model
        results = model.predict(file_path, save=False, conf=0.4, verbose=False)
        
        processed_files = []
        
        for r in results:
            if hasattr(r, 'masks') and r.masks is not None:
                print(f"[INFO] Found {len(r.masks)} detections")
                # For now, just return detection info
                processed_files.append({
                    'detection_count': len(r.masks),
                    'message': 'Instance segmentation working!'
                })
            elif hasattr(r, 'boxes') and r.boxes is not None:
                print(f"[INFO] Found {len(r.boxes)} bounding boxes")
                processed_files.append({
                    'detection_count': len(r.boxes),
                    'message': 'Object detection working!'
                })
        
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return jsonify({
            'processed_files': processed_files,
            'message': 'Your trained model is working!',
            'model_type': 'YOLOv11 Instance Segmentation'
        })
        
    except Exception as e:
        print(f"[ERROR] Processing failed: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"[INFO] Starting on port {port}")
    print(f"[INFO] Model loaded: {model is not None}")
    app.run(debug=False, host='0.0.0.0', port=port)
