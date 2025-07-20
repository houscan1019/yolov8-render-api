import os
import uuid
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Try to load model, but don't crash if it fails
model = None
try:
    from ultralytics import YOLO
    if os.path.exists("weights.pt"):
        model = YOLO("weights.pt")
        print("[SUCCESS] Model loaded successfully")
    else:
        print("[ERROR] weights.pt not found")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")

def simple_crop_and_process(image, detection_data):
    """Simple processing without complex rotation"""
    try:
        # Get bounding box from detection
        x, y, w, h = detection_data
        
        # Crop with some padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2*padding)
        h = min(image.shape[0] - y, h + 2*padding)
        
        cropped = image[y:y+h, x:x+w]
        return cropped
    except Exception as e:
        print(f"[ERROR] Crop failed: {e}")
        return image

@app.route('/')
def home():
    """Simple home page"""
    return jsonify({
        'message': 'Document Processing API is running',
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model is not None,
        'weights_exist': os.path.exists('weights.pt')
    })

@app.route('/detect-and-process', methods=['POST'])
def detect_and_process():
    if model is None:
        return jsonify({'error': 'Model not loaded - check weights.pt'}), 500
        
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
        
        # Load image
        image = cv2.imread(file_path)
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        print(f"[INFO] Processing: {filename}")
        
        # Run simple prediction
        results = model.predict(image, save=False, conf=0.4, verbose=False)
        
        processed_files = []
        
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                # Use bounding boxes for simple processing
                for i, box in enumerate(r.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Simple crop
                    cropped = image[int(y1):int(y2), int(x1):int(x2)]
                    
                    if cropped.size > 0:
                        # Save processed image
                        out_filename = f"doc_{uuid.uuid4().hex[:8]}.jpg"
                        out_path = os.path.join(UPLOAD_FOLDER, out_filename)
                        cv2.imwrite(out_path, cropped)
                        
                        processed_files.append(f"/processed/{out_filename}")
            
            elif r.masks is not None and len(r.masks) > 0:
                # Fallback to masks if available
                for i, mask in enumerate(r.masks.segments):
                    points = mask.reshape(-1, 2)
                    x, y, w, h = cv2.boundingRect(points.astype(int))
                    
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
    """Serve processed files"""
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"[INFO] Starting on port {port}")
    print(f"[INFO] Model loaded: {model is not None}")
    app.run(debug=False, host='0.0.0.0', port=port)
