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

def crop_using_polygon(image, polygon):
    try:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        x, y, w, h = cv2.boundingRect(polygon)
        cropped = cv2.bitwise_and(image, image, mask=mask)[y:y+h, x:x+w]
        return cropped
    except Exception as e:
        print(f"[ERROR] Cropping failed: {e}")
        return None

def deskew_image(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(binary > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle += 90
        
        if abs(angle) < 1 or abs(angle) > 10:
            return image  # Skip deskew
        
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except Exception as e:
        print(f"[ERROR] Deskewing failed: {e}")
        return image

# Route to serve processed images with multiple fallback methods
@app.route('/processed/<filename>')
def serve_processed_file(filename):
    try:
        # Method 1: Try with .png extension
        if not filename.endswith('.png'):
            png_filename = filename + '.png'
            png_path = os.path.join(OUTPUT_DIR, png_filename)
            if os.path.exists(png_path):
                return send_from_directory(OUTPUT_DIR, png_filename)
        
        # Method 2: Try exact filename
        exact_path = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(exact_path):
            return send_from_directory(OUTPUT_DIR, filename)
        
        # Method 3: Debug - list available files
        print(f"[DEBUG] File not found: {filename}")
        print(f"[DEBUG] Looking in directory: {OUTPUT_DIR}")
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
        
        for r in results:
            if not r.masks:
                continue
            masks = r.masks.xy  # list of [N, 2] polygons
            for polygon in masks:
                polygon_np = np.array(polygon, dtype=np.int32)
                if len(polygon_np) < 3:
                    continue
                
                cropped = crop_using_polygon(img_np, polygon_np)
                if cropped is None or cropped.size == 0:
                    continue
                
                deskewed = deskew_image(cropped)
                if deskewed is None or deskewed.size == 0:
                    continue
                
                # Generate filename with full UUID
                filename = f"{uuid.uuid4().hex}.png"
                filepath = os.path.join(OUTPUT_DIR, filename)
                cv2.imwrite(filepath, deskewed)
                
                # Generate URL with full filename (without .png extension)
                filename_without_ext = filename[:-4]  # Remove .png extension properly
                public_url = f"https://yolov8-render-api.onrender.com/processed/{filename_without_ext}"
                public_urls.append(public_url)
                
                # Debug logging
                print(f"[DEBUG] Generated filename: {filename}")
                print(f"[DEBUG] Filename without ext: {filename_without_ext}")
                print(f"[DEBUG] Saved to: {filepath}")
                print(f"[DEBUG] File exists: {os.path.exists(filepath)}")
                print(f"[DEBUG] File size: {os.path.getsize(filepath) if os.path.exists(filepath) else 'N/A'} bytes")
                print(f"[DEBUG] Public URL: {public_url}")
        
        if not public_urls:
            return jsonify({"message": "No valid objects found."}), 200
        
        return jsonify({
            "processed_image_urls": public_urls
        }), 200
        
    except Exception as e:
        print(f"[ERROR] {e}")
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
