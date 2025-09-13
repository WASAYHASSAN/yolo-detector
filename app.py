# app.py
import os
import uuid
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "static/results"
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Paths to your model files (place these in repo root or update paths)
CFG_PATH = "yolov4.cfg"
WEIGHTS_PATH = "yolov4.weights"
NAMES_PATH = "coco.names"

# Load class names
with open(NAMES_PATH, "r") as f:
    CLASS_NAMES = [c.strip() for c in f.readlines()]

# Load YOLO network
net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGHTS_PATH)
# Use OpenCV's CPU backend (works on Render). If you have GPU support you can try net.setPreferableBackend etc.
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


def run_yolo(image_path):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    # Create blob and forward
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = float(scores[class_id])
            if confidence > CONFIDENCE_THRESHOLD:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)

    # Apply non-maxima suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    # Draw boxes
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            label = CLASS_NAMES[class_ids[i]] if class_ids[i] < len(CLASS_NAMES) else str(class_ids[i])
            conf = confidences[i]
            # Choose rectangle color based on class id
            color = [int(c) for c in list(np.random.RandomState(class_ids[i]).randint(0, 255, size=3))]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = f"{label}: {conf:.2f}"
            # put text background
            (tlw, tlh), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x, y - int(1.2*tlh)), (x + tlw, y), color, -1)
            cv2.putText(img, text, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # Save result
    out_filename = f"{uuid.uuid4().hex}.jpg"
    out_path = os.path.join(app.config['RESULT_FOLDER'], out_filename)
    cv2.imwrite(out_path, img)
    return out_filename

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == "":
            return redirect(request.url)
        # Save upload
        uid = uuid.uuid4().hex
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}_{file.filename}")
        file.save(upload_path)

        # Run detection
        result_name = run_yolo(upload_path)
        result_url = url_for('static', filename=f"results/{result_name}")
        return render_template("index.html", result_image=result_url)

    return render_template("index.html", result_image=None)

if __name__ == "__main__":
    # For local development
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
