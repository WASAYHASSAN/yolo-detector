# app.py
import os
import uuid
import threading
import logging
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import cv2
import numpy as np

# -------- Config --------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yolo-app")

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "static/results"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# Model files (must be in repo root)
CFG_PATH = "yolov4.cfg"
WEIGHTS_PATH = "yolov4.weights"
NAMES_PATH = "coco.names"

# -------- Flask app --------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET", "please-change-me")
app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    RESULT_FOLDER=RESULT_FOLDER,
    MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH,
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# -------- Load classes & model (fail fast if missing) --------
if not (os.path.exists(CFG_PATH) and os.path.exists(WEIGHTS_PATH) and os.path.exists(NAMES_PATH)):
    logger.error("One or more model files missing: %s %s %s", CFG_PATH, WEIGHTS_PATH, NAMES_PATH)
    # Continue — app will show helpful error if files are missing at runtime.

CLASS_NAMES = []
try:
    with open(NAMES_PATH, "r") as f:
        CLASS_NAMES = [line.strip() for line in f if line.strip()]
except Exception as e:
    logger.warning("Could not load coco.names: %s", e)

logger.info("Loading YOLO network (this may take a few seconds)...")
net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGHTS_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()

def get_output_layers(net):
    # Robustly handle different OpenCV return types
    outs = net.getUnconnectedOutLayers()
    try:
        idxs = outs.flatten().tolist()
    except Exception:
        try:
            idxs = [int(x[0]) for x in outs]
        except Exception:
            idxs = [int(x) for x in outs]
    return [layer_names[i - 1] for i in idxs]

OUTPUT_LAYERS = get_output_layers(net)
logger.info("YOLO output layers: %s", OUTPUT_LAYERS)

# Single lock to avoid concurrent net.forward calls (safer on shared workers)
model_lock = threading.Lock()

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def run_yolo(image_path: str) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("cv2.imread returned None (bad image or path)")
    h, w = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    with model_lock:
        net.setInput(blob)
        outputs = net.forward(OUTPUT_LAYERS)

    boxes, confidences, class_ids = [], [], []
    for out in outputs:
        for det in out:
            scores = det[5:]
            if scores.size == 0:
                continue
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence >= CONFIDENCE_THRESHOLD:
                cx = int(det[0] * w)
                cy = int(det[1] * h)
                bw = int(det[2] * w)
                bh = int(det[3] * h)
                x = int(cx - bw / 2)
                y = int(cy - bh / 2)
                boxes.append([x, y, bw, bh])
                confidences.append(confidence)
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    if len(idxs) > 0:
        try:
            iter_idxs = idxs.flatten()
        except Exception:
            iter_idxs = idxs
        for i in np.array(iter_idxs).flatten():
            i = int(i)
            x, y, bw, bh = boxes[i]
            label = CLASS_NAMES[class_ids[i]] if class_ids[i] < len(CLASS_NAMES) else str(class_ids[i])
            color = tuple(map(int, np.random.RandomState(class_ids[i]).randint(0, 255, size=3)))
            cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 2)
            text = f"{label}: {confidences[i]:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x, y - int(1.2 * th)), (x + tw, y), color, -1)
            cv2.putText(img, text, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    out_filename = f"result_{uuid.uuid4().hex}.jpg"
    out_path = os.path.join(app.config["RESULT_FOLDER"], out_filename)
    success = cv2.imwrite(out_path, img)
    if not success:
        raise IOError("Failed to write result image")
    return out_filename

# -------- Routes --------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            flash("No file part in the request", "error")
            return redirect(request.url)
        file = request.files["image"]
        if file.filename == "":
            flash("No file selected", "error")
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash("Invalid file type. Allowed: png, jpg, jpeg", "error")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        uid = uuid.uuid4().hex
        saved_name = f"{uid}_{filename}"
        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_name)
        try:
            file.save(upload_path)
        except Exception as e:
            logger.exception("Failed to save uploaded file: %s", e)
            flash("Failed to save uploaded file", "error")
            return redirect(request.url)

        # Run detection
        try:
            result_name = run_yolo(upload_path)
        except Exception as e:
            logger.exception("YOLO inference failed: %s", e)
            flash(f"Detection failed: {e}", "error")
            return redirect(request.url)

        result_url = url_for("static", filename=f"results/{result_name}")
        return render_template("index.html", result_image=result_url)

    return render_template("index.html", result_image=None)

@app.route("/healthz")
def healthz():
    return "ok", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting local dev server on port %d", port)
    app.run(host="0.0.0.0", port=port)
