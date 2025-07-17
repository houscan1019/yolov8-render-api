import os
import uuid
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO("weights.pt")

# Output directory for processed images
OUTPUT_DIR = "processed"
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
        if abs(angle) < 1 or abs(angle) > 18:
            return image  # Skip deskew

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except Exception as e:
        print(f"[ERROR] Deskewing failed: {e}")
        return image

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

                filename = f"{uuid.uuid4().hex}.png"
                filepath = os.path.join(OUTPUT_DIR, filename)
                cv2.imwrite(filepath, deskewed)

                public_url = f"https://yolov8-render-api.onrender.com/processed/{filename}"
                public_urls.append(public_url)
                print(f"[DEBUG] Saved {filename}")
                print(f"[DEBUG] Public URL: {public_url}")

        if not public_urls:
            return jsonify({"message": "No valid objects found."}), 200

        return jsonify({
            "processed_image_urls": public_urls
        }), 200

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
