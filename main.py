import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO
import uuid
import os

app = Flask(__name__)

# Create output folder if it doesn't exist
OUTPUT_DIR = "processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLOv8 segmentation model
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
        print(f"Detected skew angle: {skew_angle:.2f} degrees")
        (h, w) = image_np.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
        return cv2.warpAffine(image_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return image_np

def crop_using_polygon(image, polygon):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)

    # Apply the mask
    masked = cv2.bitwise_and(image, image, mask=mask)

    # Crop to the bounding rect of the polygon
    x, y, w, h = cv2.boundingRect(pts)
    return masked[y:y+h, x:x+w]

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
