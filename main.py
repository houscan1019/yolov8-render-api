import os
import uuid
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify, url_for, send_from_directory
from ultralytics import YOLO

# Create Flask app with explicit static folder configuration
app = Flask(__name__, static_url_path='/static', static_folder='static')

# Load YOLOv8 model
model = YOLO("weights.pt")

# Output directory for processed images
OUTPUT_DIR = os.path.join(app.static_folder, 'processed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def decode_base64_image(image_base64):
    try:
        image_data = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"[ERROR] Decoding base64 failed: {e}")
        return None

def crop_using_bounding_box(image, bbox):
    """
    Crop image using bounding box coordinates
    """
    try:
        x1, y1, x2, y2 = bbox
        
        # Ensure coordinates are within image bounds
        height, width = image.shape[:2]
        x1 = max(0, min(int(x1), width))
        y1 = max(0, min(int(y1), height))
        x2 = max(0, min(int(x2), width))
        y2 = max(0, min(int(y2), height))
        
        # Ensure x2 > x1 and y2 > y1
        if x2 <= x1 or y2 <= y1:
            print(f"[ERROR] Invalid bounding box coordinates")
            return None
        
        cropped = image[y1:y2, x1:x2]
        print(f"[DEBUG] Cropped using bbox: ({x1},{y1}) to ({x2},{y2}), size: {cropped.shape}")
        return cropped
        
    except Exception as e:
        print(f"[ERROR] Bounding box cropping failed: {e}")
        return None

def trim_border(image, border_pixels=15):
    """
    Remove border padding that might be included in bounding box
    """
    try:
        h, w = image.shape[:2]
        
        # Only trim if image is large enough
        if h > border_pixels * 2 and w > border_pixels * 2:
            trimmed = image[border_pixels:h-border_pixels, border_pixels:w-border_pixels]
            print(f"[DEBUG] Trimmed {border_pixels}px border, new size: {trimmed.shape}")
            return trimmed
        else:
            print(f"[DEBUG] Image too small to trim border")
            return image
            
    except Exception as e:
        print(f"[ERROR] Border trimming failed: {e}")
        return image

def detect_faces_and_orientation(image):
    """
    Detect faces to determine photo orientation
    """
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        face_count = len(faces)
        face_quality = 0
        
        if face_count > 0:
            total_area = sum(w * h for (x, y, w, h) in faces)
            avg_face_area = total_area / face_count
            image_area = image.shape[0] * image.shape[1]
            face_quality = (avg_face_area / image_area) * face_count * 100
        
        print(f"[DEBUG] Detected {face_count} faces, quality score: {face_quality:.2f}")
        return face_count, face_quality
        
    except Exception as e:
        print(f"[ERROR] Face detection failed: {e}")
        return 0, 0

def analyze_photo_composition(image):
    """
    Analyze natural photo composition (sky/ground distribution)
    """
    try:
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Analyze upper and lower thirds
        upper_third = hsv[:height//3, :]
        lower_third = hsv[2*height//3:, :]
        
        # Sky colors (blues) in upper region
        sky_mask_upper = cv2.inRange(upper_third, (100, 50, 50), (130, 255, 255))
        sky_ratio_upper = np.sum(sky_mask_upper > 0) / sky_mask_upper.size
        
        # Sky colors in lower region (should be less)
        sky_mask_lower = cv2.inRange(lower_third, (100, 50, 50), (130, 255, 255))
        sky_ratio_lower = np.sum(sky_mask_lower > 0) / sky_mask_lower.size
        
        # Green colors (vegetation) in lower region
        green_mask_lower = cv2.inRange(lower_third, (40, 50, 50), (80, 255, 255))
        green_ratio_lower = np.sum(green_mask_lower > 0) / green_mask_lower.size
        
        green_mask_upper = cv2.inRange(upper_third, (40, 50, 50), (80, 255, 255))
        green_ratio_upper = np.sum(green_mask_upper > 0) / green_mask_upper.size
        
        # Score natural composition
        composition_score = (
            sky_ratio_upper * 2 +
            green_ratio_lower * 2 +
            max(0, 0.1 - sky_ratio_lower) * 5 +
            max(0, 0.1 - green_ratio_upper) * 3
        )
        
        print(f"[DEBUG] Composition - Sky upper: {sky_ratio_upper:.3f}, Green lower: {green_ratio_lower:.3f}, Score: {composition_score:.3f}")
        return composition_score
        
    except Exception as e:
        print(f"[ERROR] Composition analysis failed: {e}")
        return 0

def detect_horizon_line(image):
    """
    Detect horizontal lines that could be horizons
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        horizon_quality = 0
        
        if lines is not None:
            height = image.shape[0]
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta)
                
                # Look for horizontal lines
                if abs(angle) < 10 or abs(angle - 180) < 10:
                    # Lines in middle third are more likely to be horizon
                    if height * 0.3 < abs(rho) < height * 0.7:
                        horizon_quality += 2
                    else:
                        horizon_quality += 1
        
        print(f"[DEBUG] Horizon quality: {horizon_quality}")
        return horizon_quality
        
    except Exception as e:
        print(f"[ERROR] Horizon detection failed: {e}")
        return 0

def analyze_aspect_ratio_preference(image):
    """
    Score aspect ratio preference for photos
    """
    height, width = image.shape[:2]
    aspect_ratio = width / height
    
    if aspect_ratio > 1.2:  # Landscape
        aspect_score = min(aspect_ratio, 2.0)
    elif aspect_ratio < 0.8:  # Portrait
        aspect_score = aspect_ratio * 0.8
    else:  # Square-ish
        aspect_score = 0.9
    
    print(f"[DEBUG] Aspect ratio: {aspect_ratio:.2f}, Score: {aspect_score:.2f}")
    return aspect_score

def detect_best_photo_orientation(image):
    """
    Test all 4 orientations and find the best one for photos
    """
    try:
        orientations = {
            0: image,
            90: cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
            180: cv2.rotate(image, cv2.ROTATE_180),
            270: cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        }
        
        best_angle = 0
        best_score = 0
        
        print(f"[DEBUG] Testing photo orientations...")
        
        for angle, rotated_img in orientations.items():
            face_count, face_quality = detect_faces_and_orientation(rotated_img)
            composition_score = analyze_photo_composition(rotated_img)
            horizon_score = detect_horizon_line(rotated_img)
            aspect_score = analyze_aspect_ratio_preference(rotated_img)
            
            # Weighted scoring
            total_score = (
                face_quality * 3 +
                composition_score * 2 +
                horizon_score * 1.5 +
                aspect_score * 1
            )
            
            print(f"[DEBUG] Angle {angle}°: total score {total_score:.2f}")
            
            if total_score > best_score:
                best_score = total_score
                best_angle = angle
        
        print(f"[DEBUG] Best photo orientation: {best_angle}° (score: {best_score:.2f})")
        return orientations[best_angle], best_angle
        
    except Exception as e:
        print(f"[ERROR] Photo orientation detection failed: {e}")
        return image, 0

def smart_rotate_image(image):
    """
    Apply intelligent rotation based on image content
    """
    try:
        rotated_image, angle = detect_best_photo_orientation(image)
        
        if angle == 0:
            print(f"[DEBUG] Image is already correctly oriented")
        else:
            print(f"[DEBUG] Image rotated {angle}° for optimal orientation")
            
        return rotated_image
        
    except Exception as e:
        print(f"[ERROR] Smart rotation failed: {e}")
        return image

def deskew_rectangular_image(image):
    """
    Deskew rectangular images using edge detection
    Works well with clean rectangular crops from bounding boxes
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use edge detection to find document/photo edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
        
        if lines is None:
            print(f"[DEBUG] No lines detected for deskewing")
            return image
        
        # Collect angles from detected lines
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta)
            
            # Normalize angle to -45 to 45 degrees
            if angle > 135:
                angle = angle - 180
            elif angle > 45:
                angle = angle - 90
            elif angle < -45:
                angle = angle + 90
            
            # Only consider reasonable skew angles
            if abs(angle) < 45:
                angles.append(angle)
        
        if not angles:
            print(f"[DEBUG] No valid angles found for deskewing")
            return image
        
        # Use median angle for robustness
        median_angle = np.median(angles)
        
        print(f"[DEBUG] Detected skew angle: {median_angle:.2f}°")
        
        # Only deskew if angle is significant but reasonable
        if abs(median_angle) < 0.5:
            print(f"[DEBUG] Angle too small ({median_angle:.1f}°), skipping deskew")
            return image
        elif abs(median_angle) > 15:
            print(f"[DEBUG] Angle too large ({median_angle:.1f}°), skipping deskew")
            return image
        
        # Apply rotation to correct skew
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        
        # Calculate new image size to avoid cropping
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust transformation matrix
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        print(f"[DEBUG] Applied deskew rotation: {median_angle:.2f}°")
        return rotated
        
    except Exception as e:
        print(f"[ERROR] Rectangular deskewing failed: {e}")
        return image

# Route to serve processed images
@app.route('/processed/<filename>')
def serve_processed_file(filename):
    try:
        if not filename.endswith('.png'):
            png_filename = filename + '.png'
            png_path = os.path.join(OUTPUT_DIR, png_filename)
            if os.path.exists(png_path):
                return send_from_directory(OUTPUT_DIR, png_filename)
        
        exact_path = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(exact_path):
            return send_from_directory(OUTPUT_DIR, filename)
        
        print(f"[DEBUG] File not found: {filename}")
        if os.path.exists(OUTPUT_DIR):
            files = os.listdir(OUTPUT_DIR)
            print(f"[DEBUG] Available files: {files}")
        
        return f"File not found: {filename}", 404
        
    except Exception as e:
        print(f"[ERROR] Error serving file {filename}: {str(e)}")
        return f"Error serving file: {str(e)}", 500

@app.route("/detect-and-process", methods=["POST"])
def detect_and_process():
    try:
        data = request.get_json()
        image_base64 = data.get("image")
        if not image_base64:
            return jsonify({"error": "No image provided."}), 400
        
        img_np = decode_base64_image(image_base64)
        if img_np is None:
            return jsonify({"error": "Could not decode image from base64."}), 400
        
        results = model(img_np)
        public_urls = []
        
        # Process results - try both bounding boxes and masks
        for r in results:
            processed_objects = []
            
            # Method 1: Try to use bounding boxes (preferred for rectangular objects)
            if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
                print(f"[DEBUG] Using bounding boxes for processing")
                boxes = r.boxes.xyxy.cpu().numpy()  # Get bounding box coordinates
                
                for box in boxes:
                    x1, y1, x2, y2 = box
                    processed_objects.append(('bbox', [x1, y1, x2, y2]))
            
            # Method 2: Fallback to polygon masks if no boxes available
            elif hasattr(r, 'masks') and r.masks is not None:
                print(f"[DEBUG] Falling back to polygon masks")
                masks = r.masks.xy
                
                for polygon in masks:
                    if len(polygon) < 3:
                        continue
                    # Convert polygon to bounding box
                    x_coords = polygon[:, 0]
                    y_coords = polygon[:, 1]
                    x1, y1 = np.min(x_coords), np.min(y_coords)
                    x2, y2 = np.max(x_coords), np.max(y_coords)
                    processed_objects.append(('polygon_to_bbox', [x1, y1, x2, y2]))
            
            # Process each detected object
            for obj_type, bbox in processed_objects:
                print(f"[DEBUG] Processing {obj_type}: {bbox}")
                
                # Crop using bounding box
                cropped = crop_using_bounding_box(img_np, bbox)
                if cropped is None or cropped.size == 0:
                    continue
                
                # Optional: trim border padding
                trimmed = trim_border(cropped, border_pixels=10)
                if trimmed is None or trimmed.size == 0:
                    trimmed = cropped
                
                # Apply intelligent rotation
                rotated = smart_rotate_image(trimmed)
                if rotated is None or rotated.size == 0:
                    continue
                
                # Apply rectangular deskewing
                deskewed = deskew_rectangular_image(rotated)
                if deskewed is None or deskewed.size == 0:
                    deskewed = rotated
                
                # Save processed image
                filename = f"{uuid.uuid4().hex}.png"
                filepath = os.path.join(OUTPUT_DIR, filename)
                cv2.imwrite(filepath, deskewed)
                
                # Generate URL
                filename_without_ext = filename[:-4]
                public_url = f"https://yolov8-render-api.onrender.com/processed/{filename_without_ext}"
                public_urls.append(public_url)
                
                print(f"[DEBUG] Processed and saved: {filename}")
                print(f"[DEBUG] Public URL: {public_url}")
        
        if not public_urls:
            return jsonify({"message": "No valid objects found."}), 200
        
        return jsonify({
            "processed_image_urls": public_urls
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Processing error: {e}")
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

# Health check route
@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "output_dir": OUTPUT_DIR})

# Debug route to list files
@app.route('/debug/files')
def debug_files():
    try:
        if os.path.exists(OUTPUT_DIR):
            files = os.listdir(OUTPUT_DIR)
            return jsonify({"files": files, "output_dir": OUTPUT_DIR})
        else:
            return jsonify({"error": "Output directory does not exist", "output_dir": OUTPUT_DIR})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
