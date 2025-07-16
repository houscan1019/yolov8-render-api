import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

try:
    model = YOLO("weights.pt") # <<<<<<<<<<<<<< IMPORTANT: Using weights.pt as confirmed
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
            if x2 - x1 == 0:
                angle = 90.0 if y2 > y1 else -90.0
            else:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)

    if angles:
        filtered_angles = [a for a in angles if abs(a % 90) > 1 and abs(a % 90) < 89]

        if filtered_angles:
            skew_angle = np.median(filtered_angles)
            print(f"Detected skew angle: {skew_angle:.2f} degrees")

            (h, w) = image_np.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
            deskewed = cv2.warpAffine(image_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return deskewed
    return image_np

@app.route('/detect-and-process', methods=['POST'])
def detect_and_process():
    if model is None:
        return jsonify({"error": "YOLOv8 model not loaded. Check server logs."}), 500

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided. Please send a JSON with an 'image' key containing base64 data."}), 400

    base64_image = data['image']

    try:
        if "," in base64_image:
            base64_image = base64_image.split(",")[1]

        nparr = np.frombuffer(base64.b64decode(base64_image), np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_np is None:
            return jsonify({"error": "Could not decode image from base64. Ensure it's a valid image format (e.g., JPEG, PNG)."}), 400

        results = model(img_np)

        processed_images_base64 = []

        for r in results:
            boxes = r.boxes.xyxy.tolist()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_np.shape[1], x2)
                y2 = min(img_np.shape[0], y2)

                cropped_region = img_np[y1:y2, x1:x2]

                if cropped_region.size == 0:
                    print(f"Skipping empty or invalid cropped region: {x1,y1,x2,y2}")
                    continue

                deskewed_cropped_region = deskew_image(cropped_region)

                if deskewed_cropped_region is None or deskewed_cropped_region.size == 0:
                    print(f"Skipping empty or invalid deskewed region after processing.")
                    continue

                is_success, buffer = cv2.imencode('.png', deskewed_cropped_region)
                if not is_success:
                    print(f"Failed to encode image to PNG.")
                    continue

                processed_image_b64 = base64.b64encode(buffer).decode('utf-8')
                processed_images_base64.append(f"data:image/png;base64,{processed_image_b64}")

        if not processed_images_base64:
            return jsonify({"message": "No objects detected or no valid regions to process."}), 200

        return jsonify({"processed_images": processed_images_base64}), 200

    except Exception as e:
        print(f"Error during processing: {e}")
        return jsonify({"error": f"An error occurred during processing: {str(e)}", "trace": str(e.__traceback__.tb_next)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
