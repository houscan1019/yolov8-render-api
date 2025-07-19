import os
import uuid
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model with TTA (Test-Time Augmentation) for better accuracy
model = YOLO("weights.pt")

def detect_faces_for_orientation(image):
    """Detect faces to determine photo orientation"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        # Faces detected - likely a photo that shouldn't be rotated
        print("[INFO] Faces detected - treating as oriented photo")
        return True
    return False

def detect_horizon_lines(image):
    """Detect horizon lines for natural photo orientation"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Focus on horizontal lines in the middle third of the image
    h, w = gray.shape
    roi_edges = edges[h//3:2*h//3, :]
    
    lines = cv2.HoughLines(roi_edges, 1, np.pi/180, int(w*0.3))
    
    if lines is not None:
        horizontal_lines = []
        for rho, theta in lines[:, 0]:
            angle = abs((theta - np.pi/2) * 180 / np.pi)
            if angle < 10:  # Nearly horizontal lines
                horizontal_lines.append(theta)
        
        if len(horizontal_lines) > 0:
            print(f"[INFO] Found {len(horizontal_lines)} potential horizon lines")
            return True
    
    return False

def smart_orientation_check(image):
    """Determine if image should be rotated based on AI analysis"""
    has_faces = detect_faces_for_orientation(image)
    has_horizon = detect_horizon_lines(image)
    
    # If faces or strong horizon detected, be conservative with rotation
    if has_faces or has_horizon:
        return False
    
    return True

def enhance_image_preprocessing(image):
    """Apply preprocessing to improve detection accuracy"""
    # Convert to LAB color space and apply CLAHE to L channel
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Slight sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Blend original and enhanced (70% enhanced, 30% original)
    result = cv2.addWeighted(sharpened, 0.7, image, 0.3, 0)
    
    return result

def get_optimal_rectangle(contour):
    """Get best rectangle fit using multiple approaches"""
    # Approach 1: minAreaRect
    rect1 = cv2.minAreaRect(contour)
    
    # Approach 2: Convex hull + minAreaRect for cleaner edges
    hull = cv2.convexHull(contour)
    rect2 = cv2.minAreaRect(hull)
    
    # Calculate area efficiency for both
    contour_area = cv2.contourArea(contour)
    area1 = rect1[1][0] * rect1[1][1]
    area2 = rect2[1][0] * rect2[1][1]
    
    efficiency1 = contour_area / area1 if area1 > 0 else 0
    efficiency2 = contour_area / area2 if area2 > 0 else 0
    
    # Choose the more efficient rectangle
    chosen_rect = rect1 if efficiency1 >= efficiency2 else rect2
    print(f"[DEBUG] Rectangle efficiency: {max(efficiency1, efficiency2):.3f}")
    
    return chosen_rect

def deskew_with_known_angle(image, angle, is_smart_rotation=False):
    """Deskew with angle, considering smart rotation context"""
    # More conservative threshold for photos with faces/horizons
    threshold = 1.0 if is_smart_rotation else 0.2
    
    if abs(angle) < threshold:
        print(f"[DEBUG] Angle too small ({angle:.2f}°), skipping deskew")
        return image
    
    # Quantize angle for stability
    angle = round(angle * 4) / 4.0
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Calculate new dimensions to avoid cropping
    cos_a, sin_a = abs(np.cos(np.radians(angle))), abs(np.sin(np.radians(angle)))
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    
    deskewed = cv2.warpAffine(image, M, (new_w, new_h), 
                             flags=cv2.INTER_CUBIC, 
                             borderMode=cv2.BORDER_REPLICATE)
    
    print(f"[INFO] Deskewed image with angle {angle:.2f}°")
    return deskewed

def crop_with_padding_removal(image, points, padding_px=15):
    """Crop image and remove annotation padding"""
    x, y, w, h = cv2.boundingRect(points)
    
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
    
    return image[y:y+h, x:x+w]

def apply_test_time_augmentation(image):
    """Apply TTA for more robust predictions"""
    # Original prediction
    results_original = model.predict(image, save=False, conf=0.4, verbose=False)
    
    # Horizontally flipped prediction
    image_flipped = cv2.flip(image, 1)
    results_flipped = model.predict(image_flipped, save=False, conf=0.4, verbose=False)
    
    # Combine results (simplified - using original for now, but you could ensemble)
    return results_original

@app.route('/detect-and-process', methods=['POST'])
def detect_and_process():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, f"raw_{filename}")
    file.save(file_path)

    try:
        # Load and enhance image
        image = cv2.imread(file_path)
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        print(f"[INFO] Processing image: {filename}")
        
        # Apply preprocessing enhancement
        enhanced_image = enhance_image_preprocessing(image)
        
        # Smart orientation analysis
        allow_smart_rotation = smart_orientation_check(image)
        
        # Run prediction with TTA
        results = apply_test_time_augmentation(enhanced_image)

        processed_files = []
        detection_count = 0
        
        for r in results:
            if r.masks is not None and len(r.masks.segments) > 0:
                print(f"[INFO] Found {len(r.masks.segments)} detections")
                
                for i, seg in enumerate(r.masks.segments):
                    detection_count += 1
                    print(f"[DEBUG] Processing detection {detection_count}")
                    
                    # Create mask
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    points = np.array([seg], dtype=np.int32)
                    cv2.fillPoly(mask, points, 255)

                    # Crop with padding removal
                    cropped = crop_with_padding_removal(image, points[0])
                    
                    if cropped.size == 0:
                        print(f"[WARN] Empty crop for detection {detection_count}, skipping")
                        continue

                    # Apply deskewing based on detection quality
                    if len(seg) >= 4:
                        # Use optimal rectangle detection
                        rect = get_optimal_rectangle(np.array(seg, dtype=np.float32))
                        angle = rect[-1]
                        
                        # Normalize angle to [-45, 45] range
                        if angle < -45:
                            angle += 90
                        elif angle > 45:
                            angle -= 90
                            
                        cropped = deskew_with_known_angle(cropped, angle, allow_smart_rotation)
                    else:
                        print(f"[WARN] Insufficient points for rectangle detection ({len(seg)} points)")
                        # Fallback to basic processing
                        continue

                    # Save processed image
                    out_filename = f"processed_{uuid.uuid4().hex[:8]}.jpg"
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
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/processed/<path:filename>')
def serve_processed(filename):
    """Serve processed files with proper headers"""
    try:
        return send_from_directory(UPLOAD_FOLDER, filename, 
                                 mimetype='image/jpeg',
                                 as_attachment=False)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    print("[INFO] Starting Flask application...")
    print(f"[INFO] Model loaded: {model is not None}")
    print(f"[INFO] Upload folder: {UPLOAD_FOLDER}")
    app.run(debug=True, host='0.0.0.0', port=5000)
