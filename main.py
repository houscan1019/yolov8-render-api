import os
import uuid
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory, url_for
from ultralytics import YOLO

app = Flask(__name__, static_url_path='/static', static_folder='static')
OUTPUT_DIR = os.path.join(app.static_folder, 'processed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO("weights.pt")

def decode_base64_image(image_base64):
    try:
        image_data = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"[ERROR] Decoding base64 failed: {e}")
        return None

def force_rectangular_mask(polygon):
    try:
        if len(polygon) == 4:
            return polygon.astype(np.int32)
        polygon_float = polygon.astype(np.float32)
        rect = cv2.minAreaRect(polygon_float)
        box = cv2.boxPoints(rect)
        return np.int32(box)
    except Exception as e:
        print(f"[ERROR] Rectangle conversion failed: {e}")
        x_coords = polygon[:, 0]
        y_coords = polygon[:, 1]
        x1, y1 = np.min(x_coords), np.min(y_coords)
        x2, y2 = np.max(x_coords), np.max(y_coords)
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)

def crop_using_rectangular_mask(image, rectangle):
    try:
        x_coords = rectangle[:, 0]
        y_coords = rectangle[:, 1]
        x1, y1 = np.min(x_coords), np.min(y_coords)
        x2, y2 = np.max(x_coords), np.max(y_coords)
        height, width = image.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(width, int(x2)), min(height, int(y2))
        if x2 <= x1 or y2 <= y1:
            return None, None
        cropped = image[y1:y2, x1:x2]
        return cropped, {
            'rectangle': rectangle,
            'angle': get_rectangle_angle(rectangle),
            'cropped_image': cropped
        }
    except Exception as e:
        print(f"[ERROR] Rectangular cropping failed: {e}")
        return None, None

def get_rectangle_angle(rectangle):
    try:
        edges = []
        for i in range(4):
            p1 = rectangle[i]
            p2 = rectangle[(i + 1) % 4]
            length = np.linalg.norm(p2 - p1)
            angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
            edges.append((length, angle))
        angle = max(edges, key=lambda x: x[0])[1]
        if angle > 45: angle -= 90
        elif angle < -45: angle += 90
        return angle
    except Exception as e:
        print(f"[ERROR] Angle calculation failed: {e}")
        return 0

def trim_border(image, border_pixels=15):
    try:
        h, w = image.shape[:2]
        if h > border_pixels * 2 and w > border_pixels * 2:
            return image[border_pixels:h-border_pixels, border_pixels:w-border_pixels]
        return image
    except Exception as e:
        print(f"[ERROR] Border trimming failed: {e}")
        return image

def detect_faces_and_orientation(image):
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        face_quality = 0
        if len(faces) > 0:
            total_area = sum(w * h for (x, y, w, h) in faces)
            avg_face_area = total_area / len(faces)
            face_quality = (avg_face_area / (image.shape[0] * image.shape[1])) * len(faces) * 100
        return len(faces), face_quality
    except Exception as e:
        print(f"[ERROR] Face detection failed: {e}")
        return 0, 0

def analyze_photo_composition(image):
    try:
        height = image.shape[0]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        upper = hsv[:height//3, :]
        lower = hsv[height//3:, :]
        sky_upper = cv2.inRange(upper, (100, 50, 50), (130, 255, 255))
        sky_lower = cv2.inRange(lower, (100, 50, 50), (130, 255, 255))
        green_lower = cv2.inRange(lower, (40, 50, 50), (80, 255, 255))
        green_upper = cv2.inRange(upper, (40, 50, 50), (80, 255, 255))
        score = (
            (np.sum(sky_upper > 0) / sky_upper.size) * 2 +
            (np.sum(green_lower > 0) / green_lower.size) * 2 +
            max(0, 0.1 - (np.sum(sky_lower > 0) / sky_lower.size)) * 5 +
            max(0, 0.1 - (np.sum(green_upper > 0) / green_upper.size)) * 3
        )
        return score
    except Exception as e:
        print(f"[ERROR] Composition analysis failed: {e}")
        return 0

def detect_horizon_line(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        score = 0
        if lines is not None:
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta)
                if abs(angle) < 10 or abs(angle - 180) < 10:
                    score += 2
        return score
    except Exception as e:
        print(f"[ERROR] Horizon detection failed: {e}")
        return 0

def analyze_aspect_ratio_preference(image):
    h, w = image.shape[:2]
    ratio = w / h
    if ratio > 1.2:
        return min(ratio, 2.0)
    elif ratio < 0.8:
        return ratio * 0.8
    return 0.9

def detect_best_photo_orientation(image):
    try:
        orientations = {
            0: image,
            90: cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
            180: cv2.rotate(image, cv2.ROTATE_180),
            270: cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        }
        best_angle = 0
        best_score = 0
        for angle, rotated in orientations.items():
            fc, fq = detect_faces_and_orientation(rotated)
            comp = analyze_photo_composition(rotated)
            hz = detect_horizon_line(rotated)
            ar = analyze_aspect_ratio_preference(rotated)
            score = fq * 3 + comp * 2 + hz * 1.5 + ar
            if score > best_score:
                best_score = score
                best_angle = angle
        return orientations[best_angle], best_angle
    except Exception as e:
        print(f"[ERROR] Orientation detection failed: {e}")
        return image, 0

def smart_rotate_image(image):
    try:
        rotated, angle = detect_best_photo_orientation(image)
        return rotated
    except Exception as e:
        print(f"[ERROR] Smart rotation failed: {e}")
        return image

def deskew_with_known_angle(image, angle):
    try:
        if abs(angle) < 0.5 or abs(angle) > 45:
            return image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception as e:
        print(f"[ERROR] Deskew with angle failed: {e}")
        return image

def deskew_rectangular_image(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, 3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 50)
        if lines is None:
            return image
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta)
            if angle > 135: angle -= 180
            elif angle > 45: angle -= 90
            if abs(angle) < 45:
                angles.append(angle)
        if not angles:
            return image
        median_angle = np.median(angles)
        if abs(median_angle) < 0.5 or abs(median_angle) > 15:
            return image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception as e:
        print(f"[ERROR] Rectangular deskewing failed: {e}")
        return image

@app.route('/processed/<filename>')
def serve_processed_file(filename):
    return send_from_directory(OUTPUT_DIR, filename)

@app.route("/detect-and-process", methods=["POST"])
def detect_and_process():
    try:
        data = request.get_json()
        image_base64 = data.get('image')
        if not image_base64:
            return jsonify({"error": "No image provided."}), 400
        img_np = decode_base64_image(image_base64)
        if img_np is None:
            return jsonify({"error": "Could not decode image from base64."}), 400
        results = model(img_np)
        public_urls = []
        for r in results:
            processed_objects = []
            if hasattr(r, 'masks') and r.masks is not None:
                for polygon in r.masks.xy:
                    if len(polygon) < 3: continue
                    rectangle = force_rectangular_mask(polygon)
                    processed_objects.append(('rectangle_mask', rectangle))
            elif hasattr(r, 'boxes') and r.boxes is not None:
                for box in r.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box
                    rectangle = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                    processed_objects.append(('rectangle_box', rectangle))
            for obj_type, rectangle in processed_objects:
                crop_result = crop_using_rectangular_mask(img_np, rectangle)
                if crop_result is None or crop_result[0] is None or crop_result[0].size == 0:
                    continue
                cropped, rect_info = crop_result
                trimmed = trim_border(cropped, border_pixels=10)
                if trimmed is None or trimmed.size == 0:
                    trimmed = cropped
                rotated = smart_rotate_image(trimmed)
                if rotated is None or rotated.size == 0:
                    rotated = trimmed
                angle = rect_info.get('angle', 0)
                deskewed = deskew_with_known_angle(rotated, angle) if angle else deskew_rectangular_image(rotated)
                if deskewed is None or deskewed.size == 0:
                    deskewed = rotated
                filename = f"{uuid.uuid4().hex}.png"
                filepath = os.path.join(OUTPUT_DIR, filename)
                cv2.imwrite(filepath, deskewed)
                public_urls.append(url_for('serve_processed_file', filename=filename, _external=True))
        return jsonify({"processed_image_urls": public_urls}) if public_urls else jsonify({"message": "No valid objects found."}), 200
    except Exception as e:
        print(f"[ERROR] Processing error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "output_dir": OUTPUT_DIR})

@app.route('/debug/files')
def debug_files():
    try:
        return jsonify({"files": os.listdir(OUTPUT_DIR), "output_dir": OUTPUT_DIR})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
