import os
import cv2
import numpy as np
import base64
import urllib.request
import uuid
import gc
import io
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import torch

app = Flask(__name__)
CORS(app)

# Railway port configuration
PORT = int(os.environ.get('PORT', 5000))

# Create static directory for processed images
static_dir = os.path.join(os.getcwd(), 'static', 'processed')
os.makedirs(static_dir, exist_ok=True)

def download_model_from_release():
    """Download YOLO model from GitHub release if not present"""
    model_path = 'weights.pt'
    
    if not os.path.exists(model_path):
        print("📥 Downloading YOLO model from GitHub release...")
        try:
            # Your GitHub release URL for weights.pt
            model_url = "https://github.com/houscani019/yolov8-render-api/releases/download/v1.0/weights.pt"
            urllib.request.urlretrieve(model_url, model_path)
            print("✅ YOLO model downloaded successfully")
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            return False
    else:
        print("✅ YOLO model already exists")
    
    return True

# Download and load YOLO model
try:
    if download_model_from_release():
        model = YOLO('weights.pt')
        print("✅ YOLO model loaded successfully")
    else:
        model = None
        print("❌ Failed to download/load YOLO model")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    model = None

@app.route('/')
def home():
    return jsonify({
        "message": "YOLOv8 Document Processing API", 
        "status": "running",
        "model_loaded": model is not None,
        "version": "2.0 - Railway Deployment"
    }), 200

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy", 
        "service": "document-processor",
        "model_status": "loaded" if model else "failed",
        "static_dir": static_dir
    }), 200

def detect_faces(image):
    """Detect faces in the image to determine orientation"""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return len(faces) > 0
    except:
        return False

def smart_rotate_image(image, angle, has_faces=False):
    """Smart rotation based on content analysis"""
    if has_faces:
        # For photos with faces, be more conservative with rotation
        if abs(angle) < 5:
            return image
    
    # Get image dimensions
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions
    cos_angle = abs(rotation_matrix[0, 0])
    sin_angle = abs(rotation_matrix[0, 1])
    new_width = int((height * sin_angle) + (width * cos_angle))
    new_height = int((height * cos_angle) + (width * sin_angle))
    
    # Adjust rotation matrix for new dimensions
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    
    # Perform rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    return rotated

def process_yolo_detection(image, results):
    """Process YOLO detection results and extract document"""
    try:
        if not results or len(results) == 0:
            return image, "No detection results"
        
        result = results[0]
        
        # Check if we have masks (instance segmentation)
        if hasattr(result, 'masks') and result.masks is not None:
            # Instance segmentation - use mask
            mask = result.masks.data[0].cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
            
            # Resize mask to match image dimensions
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour
                main_contour = max(contours, key=cv2.contourArea)
                
                # Get minimum area rectangle (preserves rotation)
                rect = cv2.minAreaRect(main_contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # Extract rotation angle
                angle = rect[2]
                if angle < -45:
                    angle = 90 + angle
                
                # Detect if image contains faces
                has_faces = detect_faces(image)
                
                # Apply smart rotation
                if not has_faces and abs(angle) > 1:  # Only rotate documents, not photos
                    image = smart_rotate_image(image, angle, has_faces)
                
                # Get bounding rectangle for cropping
                x, y, w, h = cv2.boundingRect(main_contour)
                
                # Add padding
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                
                # Crop the document
                cropped = image[y:y+h, x:x+w]
                
                return cropped, "Instance segmentation successful"
        
        # Fallback to bounding box detection
        elif hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
            box = result.boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Convert to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Add padding
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)
            
            # Crop the image
            cropped = image[y1:y2, x1:x2]
            
            return cropped, "Bounding box detection successful"
        
        return image, "No valid detection found"
        
    except Exception as e:
        print(f"Error in process_yolo_detection: {e}")
        return image, f"Processing error: {str(e)}"

@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        if not model:
            return jsonify({"error": "YOLO model not loaded"}), 500
            
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(image_data))
            image_np = np.array(image)
        except Exception as e:
            return jsonify({"error": f"Invalid image data: {str(e)}"}), 400
        
        # Convert RGB to BGR for OpenCV
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        elif len(image_np.shape) == 3 and image_np.shape[2] == 4:
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
        else:
            image_cv = image_np
        
        print(f"Processing image with shape: {image_cv.shape}")
        
        # Run YOLO inference
        results = model(image_cv, conf=0.5)
        
        # Process results
        processed_image, processing_info = process_yolo_detection(image_cv, results)
        
        # Generate unique filename
        filename = f"processed_{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(static_dir, filename)
        
        # Save processed image
        cv2.imwrite(filepath, processed_image)
        
        # Generate URL
        base_url = request.url_root.rstrip('/')
        image_url = f"{base_url}/static/processed/{filename}"
        
        # Get detection info
        detection_count = 0
        confidence_scores = []
        
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                detection_count = len(result.boxes)
                confidence_scores = [float(conf) for conf in result.boxes.conf.cpu().numpy()]
        
        return jsonify({
            "success": True,
            "processed_image_url": image_url,
            "filename": filename,
            "processing_info": processing_info,
            "detections": detection_count,
            "confidence_scores": confidence_scores,
            "original_shape": image_cv.shape[:2],
            "processed_shape": processed_image.shape[:2]
        }), 200
        
    except Exception as e:
        print(f"Error in process_image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/static/processed/<filename>')
def serve_processed_image(filename):
    """Serve processed images"""
    return send_from_directory(static_dir, filename)

@app.route('/test')
def test():
    """Simple test endpoint"""
    return "API is working! Railway deployment successful."

@app.after_request
def cleanup(response):
    """Clean up memory after each request"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return response

if __name__ == '__main__':
    print(f"🚀 Starting Document Processing API on port {PORT}")
    print(f"📁 Static directory: {static_dir}")
    print(f"🤖 Model loaded: {model is not None}")
    app.run(host='0.0.0.0', port=PORT, debug=False)
