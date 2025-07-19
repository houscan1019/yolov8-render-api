
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

model = YOLO("weights.pt")

def deskew_with_known_angle(image, angle):
    if abs(angle) < 0.2:
        print(f"[DEBUG] Angle too small ({angle:.2f}°), skipping deskew")
        return image
    angle = round(angle * 4) / 4.0
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    print(f"[INFO] Deskewed image with angle {angle:.2f}")
    return deskewed

def deskew_rectangular_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is None:
        print("[WARN] No lines detected for deskewing.")
        return image

    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta - np.pi / 2) * 180 / np.pi
        angles.append(angle)

    if len(angles) == 0:
        return image

    median_angle = np.median(angles)
    if abs(median_angle) < 0.2:
        print(f"[DEBUG] Hough angle too small ({median_angle:.2f}°), skipping")
        return image
    median_angle = round(median_angle * 4) / 4.0
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    print(f"[INFO] Deskewed using Hough angle {median_angle:.2f}")
    return deskewed

@app.route('/detect-and-process', methods=['POST'])
def detect_and_process():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, f"raw_{filename}")
    file.save(file_path)

    image = cv2.imread(file_path)
    results = model.predict(image, save=False, conf=0.4)

    processed_files = []
    for r in results:
        if r.masks is not None:
            for seg in r.masks.segments:
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                points = np.array([seg], dtype=np.int32)
                cv2.fillPoly(mask, points, 255)

                x, y, w, h = cv2.boundingRect(points)
                cropped = cv2.bitwise_and(image, image, mask=mask)[y:y+h, x:x+w]

                if len(seg) >= 4:
                    rect = cv2.minAreaRect(np.array(seg, dtype=np.float32))
                    angle = rect[-1]
                    if angle < -45:
                        angle += 90
                    cropped = deskew_with_known_angle(cropped, angle)
                else:
                    cropped = deskew_rectangular_image(cropped)

                out_filename = f"{uuid.uuid4().hex}.jpg"
                out_path = os.path.join(UPLOAD_FOLDER, out_filename)
                cv2.imwrite(out_path, cropped)
                processed_files.append(f"/processed/{out_filename}")

    return jsonify({'processed_files': processed_files})

@app.route('/processed/<path:filename>')
def serve_processed(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
