import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import uuid
import os

app = Flask(__name__)

# Create output folder if it doesn't exist
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

    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_np

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) if (x2 - x1) != 0 else 90.0
            angles.append(angle)

    if angles:
        filtered_angles = [a for a in angles if abs(a % 90) > 1 and abs(a % 90) < 89]
        if filtered_angles:
            skew_angle = np.median(filtered_angles)
            print(f"Detected skew angle: {skew_angle:.2f} degrees")
            (h, w) = image_np.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
            return cv2.warpAffine(image_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return image_np

@app.route('/detect-and-process', methods=['POST'])
def detect_and_process():
    if model is None:
        return jsonify({"error": "YOLOv8 model not loaded."}), 500

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided."}), 400

    base64_image = data['image']
    if "," in base64_image:
        base64_image = base64_image.split(",")[1]

    try:
        nparr = np.frombuffer(base64.b64decode(base64_image), np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_np is None:
            return jsonify({"error": "Could not decode image from base64."}), 400

        results = model(img_np)
        processed_images_base64 = []
        saved_file_paths = []

        for r in results:
            boxes = r.boxes.xyxy.tolist()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(img_np.shape[1], x2); y2 = min(img_np.shape[0], y2)

                cropped = img_np[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue

                deskewed = deskew_image(cropped)
                if deskewed is None or deskewed.size == 0:
                    continue

                # Save image to disk
                filename = f"{uuid.uuid4().hex}.png"
                filepath = os.path.join(OUTPUT_DIR, filename)
                cv2.imwrite(filepath, deskewed)
                saved_file_paths.append(filepath)

                # Encode to base64
                success, buffer = cv2.imencode('.png', deskewed)
                if not success:
                    continue
                image_b64 = base64.b64encode(buffer).decode('utf-8')
                processed_images_base64.append(f"data:image/png;base64,{image_b64}")

        if not processed_images_base64:
            return jsonify({"message": "No valid objects found."}), 200

        return jsonify({
            "processed_images": processed_images_base64,
            "saved_files": saved_file_paths
        }), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
