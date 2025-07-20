import os
import uuid
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
from PIL import Image
from werkzeug.utils import secure_filename
import requests

app = Flask(__name__)
UPLOAD_FOLDER = 'static/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Download weights if they don't exist (backup method)
def ensure_weights_exist():
    if not os.path.exists("weights.pt"):
        print("[WARNING] weights.pt not found. Please upload it manually to Render.")
        # You can add a download URL here if you put weights.pt in GitHub releases
        return False
    return True

# Load model
try:
    if ensure_weights_exist():
        model = YOLO("weights.pt")
        print("[SUCCESS] Model loaded successfully")
    else:
        model = None
        print("[ERROR] Cannot load model - weights.pt missing")
except Exception as e:
    model = None
    print(f"[ERROR] Failed to load model: {e}")

def deskew_with_known_angle(image, angle):
    """Rotate image by specified angle"""
    if abs(angle) < 0.5:  # Skip very small rotations
        print(f"[DEBUG] Angle too small ({angle:.2f}°), skipping deskew")
        return image
    
    # Round angle for stability
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
    deskewed = cv2.warpAffine(image, M, (new_w, new_h), 
                             flags=cv2.INTER_CUBIC, 
                             borderMode=cv2.BORDER_REPLICATE)
    
    print(f"[INFO] Deskewed image with angle {angle:.2f}°")
    return deskewed

def get_rectangle_from_points(points):
    """Get the best rectangle from polygon points"""
    # Convert to numpy array
    contour = np.array(points, dtype=np.float32)
    
    # Get minimum area rectangle
    rect = cv2.minAreaRect(contour)
    
    # Extract angle
    angle = rect[-1]
    
    # Normalize angle to [-45, 45] range
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    
    print(f"[DEBUG] Rectangle angle: {angle:.2f}°")
    return angle

def crop_with_padding_removal(image, points, padding_px=12):
    """Crop image and remove annotation padding"""
    x, y, w, h = cv2.boundingRect(np.array(points, dtype=np.int32))
    
    # Remove padding from bounds
    x = max(0, x + padding_px)
    y = max(0, y + padding_px)
    w = max(1, w - 2 * padding_px)
    h = max(1, h - 2 * padding_px)
    
    # Ensure bounds are within image
    x = min(x, image.shape[1] - 1)
    y = min(y, image.shape[0] - 1)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    
    cropped = image[y:y+h, x:x+w]
    print(f"[DEBUG] Cropped to size: {w}x{h}")
    return cropped

@app.route('/detect-and-process', methods=['POST'])
def detect_and_process():
    if model is None:
        return jsonify({'error': 'Model not loaded - weights.pt missing'}), 500
        
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
        
        print(f"[INFO] Processing image: {filename}")
        
        # Run prediction
        results = model.predict(image, save=False, conf=0.4, verbose=False)
        
        processed_files = []
        detection_count = 0
        
        for r in results:
            if r.masks is not None and len(r.masks.segments) > 0:
                print(f"[INFO] Found {len(r.masks.segments)} detections")
                
                for i, seg in enumerate(r.masks.segments):
                    detection_count += 1
                    print(f"[DEBUG] Processing detection {detection_count}")
                    
                    # Convert segment points to proper format
                    points = seg.reshape(-1, 2)
                    
                    # Crop image using polygon points
                    cropped = crop_with_padding_removal(image, points)
                    
                    if cropped.size == 0:
                        print(f"[WARN] Empty crop for detection {detection_count}, skipping")
                        continue
                    
                    # Get rotation angle from polygon
                    if len(points) >= 4:
                        angle = get_rectangle_from_points(points)
                        cropped = deskew_with_known_angle(cropped, angle)
                    else:
                        print(f"[WARN] Not enough points for angle calculation ({len(points)} points)")
                    
                    # Save processed image
                    out_filename = f"doc_{uuid.uuid4().hex[:8]}.jpg"
                    out_path = os.path.join(UPLOAD_FOLDER, out_filename)
                    
                    # Save with high quality
                    cv2.imwrite(out_path, cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    url = f"/processed/{out_filename}"
                    processed_files.append(url)
                    print(f"[INFO] Saved: {url}")
            else:
                print("[WARN] No masks detected in results")
        
        # Cleanup raw file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        if not processed_files:
            return jsonify({
                'message': 'No documents detected in image',
                'processed_files': []
            }), 200
        
        print(f"[SUCCESS] Processed {len(processed_files)} documents")
        return jsonify({
            'processed_files': processed_files,
            'detection_count': len(processed_files)
        })
        
    except Exception as e:
        print(f"[ERROR] Processing failed: {str(e)}")
        # Cleanup on error
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/processed/<path:filename>')
def serve_processed(filename):
    """Serve processed files"""
    try:
        return send_from_directory(UPLOAD_FOLDER, filename, 
                                 mimetype='image/jpeg',
                                 as_attachment=False)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model is not None,
        'weights_exist': os.path.exists('weights.pt')
    })

@app.route('/')
def home():
    """Simple home page"""
    return jsonify({
        'message': 'Document Processing API is running',
        'endpoints': {
            'detect': '/detect-and-process (POST)',
            'health': '/health (GET)',
            'files': '/processed/<filename> (GET)'
        }
    })

if __name__ == '__main__':
    print("[INFO] Starting Document Processing API...")
    print(f"[INFO] Model loaded: {model is not None}")
    print(f"[INFO] Upload folder: {UPLOAD_FOLDER}")
    if model is None:
        print("[WARNING] Upload weights.pt to Render manually!")
    
    # Use PORT environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    print(f"[INFO] Starting on port {port}")
    app.run(debug=False, host='0.0.0.0', port=port)
