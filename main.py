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
    else:
        print("[INFO] weights.pt already exists")
        return True

# Load model
model = None
try:
    if download_weights():
        from ultralytics import YOLO
        model = YOLO("weights.pt")
        print("[SUCCESS] Your trained model loaded successfully!")
    else:
        print("[ERROR] Cannot load model - weights.pt missing")
except Exception as e:
    print(f"[ERROR] Model loading failed: {e}")

def deskew_with_angle(image, angle):
    """Apply rotation correction to image"""
    if abs(angle) < 0.5:
        return image
    
    # Quantize angle for stability
    angle = round(angle * 2) / 2.0
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Calculate new dimensions to avoid cropping
    cos_a, sin_a = abs(np.cos(np.radians(angle))), abs(np.sin(np.radians(angle)))
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    
    # Create rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    
    # Apply rotation
    rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                           flags=cv2.INTER_CUBIC, 
                           borderMode=cv2.BORDER_REPLICATE)
    
    print(f"[INFO] Applied rotation: {angle:.1f}°")
    return rotated

def get_rotation_angle(points):
    """Calculate rotation angle from polygon points"""
    if len(points) < 4:
        return 0
    
    # Get minimum area rectangle
    rect = cv2.minAreaRect(np.array(points, dtype=np.float32))
    angle = rect[-1]
    
    # Normalize angle to [-45, 45] range
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    
    return angle

def crop_with_padding_removal(image, points, padding=12):
    """Crop image and remove annotation padding"""
    x, y, w, h = cv2.boundingRect(np.array(points, dtype=np.int32))
    
    # Remove annotation padding
    x = max(0, x + padding)
    y = max(0, y + padding)  
    w = max(1, w - 2 * padding)
    h = max(1, h - 2 * padding)
    
    # Ensure bounds are within image
    x = min(x, image.shape[1] - 1)
    y = min(y, image.shape[0] - 1)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    
    return image[y:y+h, x:x+w]

@app.route('/')
def home():
    return jsonify({
        'message': 'Document Processing API with Trained Model',
        'status': 'healthy',
        'model_loaded': model is not None,
        'endpoints': {
            'detect': '/detect-and-process (POST)',
            'health': '/health (GET)', 
            'files': '/processed/<filename> (GET)'
        }
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'weights_exist': os.path.exists('weights.pt'),
        'weights_url_configured': os.getenv('WEIGHTS_URL') is not None
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
        
        # Load and process image
        image = cv2.imread(file_path)
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        print(f"[INFO] Processing image: {filename}")
        
        # Run your trained model
        results = model.predict(image, save=False, conf=0.4, verbose=False)
        
        processed_files = []
        detection_count = 0
        
        for r in results:
            # Use instance segmentation masks from your training
            if r.masks is not None and len(r.masks.segments) > 0:
                print(f"[INFO] Found {len(r.masks.segments)} documents")
                
                for i, seg in enumerate(r.masks.segments):
                    detection_count += 1
                    
                    # Get polygon points from your trained model
                    points = seg.reshape(-1, 2)
                    
                    # Crop with padding removal (removes training annotation padding)
                    cropped = crop_with_padding_removal(image, points)
                    
                    if cropped.size == 0:
                        print(f"[WARN] Empty crop for detection {detection_count}")
                        continue
                    
                    # Apply rotation correction based on polygon shape
                    if len(points) >= 4:
                        rotation_angle = get_rotation_angle(points)
                        cropped = deskew_with_angle(cropped, rotation_angle)
                    
                    # Save processed document
                    out_filename = f"doc_{uuid.uuid4().hex[:8]}.jpg"
                    out_path = os.path.join(UPLOAD_FOLDER, out_filename)
                    
                    # Save with high quality
                    cv2.imwrite(out_path, cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    url = f"/processed/{out_filename}"
                    processed_files.append(url)
                    print(f"[INFO] Processed document {detection_count}: {url}")
                    
            else:
                print("[INFO] No instance segmentation masks found")
        
        # Cleanup raw file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        if not processed_files:
            return jsonify({
                'message': 'No documents detected in image',
                'processed_files': []
            })
        
        print(f"[SUCCESS] Processed {len(processed_files)} documents")
        return jsonify({
            'processed_files': processed_files,
            'detection_count': len(processed_files),
            'model_confidence': 'Using your trained YOLOv11 model'
        })
        
    except Exception as e:
        print(f"[ERROR] Processing failed: {str(e)}")
        # Cleanup on error
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/processed/<path:filename>')
def serve_processed(filename):
    """Serve processed document files"""
    try:
        return send_from_directory(UPLOAD_FOLDER, filename)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"[INFO] Starting Document Processing API on port {port}")
    print(f"[INFO] Model loaded: {model is not None}")
    print(f"[INFO] Upload folder: {UPLOAD_FOLDER}")
    app.run(debug=False, host='0.0.0.0', port=port)
