import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
import uuid
import os

app = Flask(__name__)

# Output directory for processed images
OUTPUT_DIR = "processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLOv8 model
try:
    model = YOLO("weights.pt")
    print("YOLOv8 model loaded successfully!")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    model = None

def deskew_image(image_np):
    if image_np is None or image_np.size == 0:
        return image_np

    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) if len(image_np.shape) == 3 else image_np
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) if (x2 - x1) != 0 else 90.0
            angles.append(angle)

    filtered_angles = [a for a in angles if abs(a % 90) > 1 and abs(a % 90) < 89]
    if filtered_angles:
        skew_angle = np.median(filtered_angles)
        print(f"[DEBUG] Detected skew angle: {skew_angle:.2f} degrees")
        (h, w) = image_np.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
        return cv2.warpAffine(image_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return image_np

def crop_using_polygon(image, polygon):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    masked = cv2.bitwise_and(image, image, mask=mask)
    x, y, w, h = cv2.boundingRect(pts)
    return masked[y:y+h, x:x+w]

@app.route('/processed/<filename>')
def serve_processed_file(filename):
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/detect-and-process', methods=['POST'])
def detect_and_process():
    if model is None:
        return jsonify({"error": "YOLOv8 model not loaded."}), 500

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided."}), 400

    try:
        base64_image = data['image']
        if "," in base64_image:
            base64_image = base64_image.split(",")[1]

        np_arr = np.frombuffer(base64.b64decode(base64_image), np.uint8)
        img_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
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

                filename = f"{uuid.uuid4().hex}.png"
                filepath = os.path.join(OUTPUT_DIR, filename)
                cv2.imwrite(filepath, deskewed)

                print(f"[DEBUG] Saved filename: {filename}")
                public_url = f"https://yolov8-render-api.onrender.com/processed/{filename}"
                print(f"[DEBUG] Public URL: {public_url}")
                public_urls.append(public_url)

        if not public_urls:
            return jsonify({"message": "No valid objects found."}), 200

        return jsonify({"processed_image_urls": public_urls}), 200

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": f"Processing error: {str(e)}"}), 500
from flask import send_from_directory
import os

# Serve static processed files
@app.route('/processed/<path:filename>')
def serve_processed_file(filename):
    return send_from_directory(os.path.abspath("processed"), filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
