import os
import uuid
import cv2
import numpy as np
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
                response = requests.get(weights_url, stream=True)
                with open('weights.pt', 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("[SUCCESS] Weights downloaded successfully")
                return True
            except Exception as e:
                print(f"[ERROR] Download failed: {e}")
                return False
        else:
            print("[ERROR] No WEIGHTS_URL provided")
            return False
    return True

# Load model
model = None
try:
    if download_weights():
        from ultralytics import YOLO
        model = YOLO("weights.pt")
        print("[SUCCESS] Model loaded successfully")
    else:
        print("[ERROR] Cannot load model")
except Exception as e:
    print(f"[ERROR] Model loading failed: {e}")

@app.route('/')
def home():
    return jsonify({
        'message': 'Document Processing API is running',
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model is not None,
        'weights_exist': os.path.exists('weights.pt')
    })

@app.route('/detect-and-process', methods=['POST'])
def detect_and_process():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, f"raw_{filename}")
        file.save(file_path)
        
        image = cv2.imread(file_path)
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        print(f"[INFO] Processing: {filename}")
        
        # Simple prediction
        results = model.predict(image, save=False, conf=0.4, verbose=False)
        
        processed_files = []
        
        for r in results:
            if r.masks is not None and len(r.masks) > 0:
                for i, mask in enumerate(r.masks.segments):
                    points = mask.reshape(-1, 2)
                    x, y, w, h = cv2.boundingRect(points.astype(int))
                    
                    # Simple crop
                    cropped = image[y:y+h, x:x+w]
                    
                    if cropped.size > 0:
                        out_filename = f"doc_{uuid.uuid4().hex[:8]}.jpg"
                        out_path = os.path.join(UPLOAD_FOLDER, out_filename)
                        cv2.imwrite(out_path, cropped)
                        
                        processed_files.append(f"/processed/{out_filename}")
        
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return jsonify({
            'processed_files': processed_files,
            'detection_count': len(processed_files)
        })
        
    except Exception as e:
        print(f"[ERROR] Processing failed: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/processed/<path:filename>')
def serve_processed(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"[INFO] Starting on port {port}")
    print(f"[INFO] Model loaded: {model is not None}")
    app.run(debug=False, host='0.0.0.0', port=port)
