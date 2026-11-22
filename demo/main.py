import ast
import traceback

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="Safety Helmet Detection ONNX API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "best.onnx"
DEFAULT_CLASS_NAMES = ["helmet", "no-helmet", "person"]  # used only if the model has no metadata
NMS_THRES = 0.45
ZOOM_OUT_BELOW = 0.5  # if the best score is under this, also try a zoomed-out copy

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
_shape = session.get_inputs()[0].shape
img_size = _shape[2] if isinstance(_shape[2], int) else 640


def load_class_names():
    """Read class names straight from the ONNX file (Ultralytics stores them there)."""
    try:
        meta = session.get_modelmeta().custom_metadata_map
        names = ast.literal_eval(meta["names"])
        return [names[i] for i in sorted(names)]
    except Exception:
        return DEFAULT_CLASS_NAMES


CLASS_NAMES = load_class_names()
print("Classes:", CLASS_NAMES)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))


def letterbox(img):
    """Resize to fit img_size x img_size, pad with gray, return blob and scale."""
    h0, w0 = img.shape[:2]
    r = min(img_size / h0, img_size / w0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    resized = cv2.resize(img, (new_w, new_h))

    padded = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized  # top-left padding, so coords / r map back directly

    blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    blob = blob.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return blob, r


def postprocess(output, h0, w0, r, conf_thres):
    p = np.squeeze(output, axis=0)
    preds = p.T if p.shape[0] < p.shape[1] else p  # -> (anchors, 4 + num_classes)
    boxes = preds[:, :4]
    scores = preds[:, 4:]

    # Ultralytics ONNX already outputs probabilities; only squash raw logits.
    if scores.min() < 0 or scores.max() > 1:
        scores = sigmoid(scores)

    class_ids = scores.argmax(axis=1)
    confs = scores.max(axis=1)

    keep = confs > conf_thres
    boxes, class_ids, confs = boxes[keep], class_ids[keep], confs[keep]

    results = []
    for c_id in np.unique(class_ids):
        m = class_ids == c_id
        b, c = boxes[m], confs[m]

        x1 = (b[:, 0] - b[:, 2] / 2) / r
        y1 = (b[:, 1] - b[:, 3] / 2) / r
        x2 = (b[:, 0] + b[:, 2] / 2) / r
        y2 = (b[:, 1] + b[:, 3] / 2) / r

        nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        idxs = cv2.dnn.NMSBoxes(nms_boxes, c.tolist(), conf_thres, NMS_THRES)
        for i in np.array(idxs).flatten():
            results.append({
                "class": CLASS_NAMES[int(c_id)],
                "confidence": round(float(c[i]), 4),
                "box": {
                    "x1": max(0.0, float(x1[i])),
                    "y1": max(0.0, float(y1[i])),
                    "x2": min(float(w0), float(x2[i])),
                    "y2": min(float(h0), float(y2[i])),
                },
            })
    return results


def run_model(img, conf_thres):
    h0, w0 = img.shape[:2]
    blob, r = letterbox(img)
    out = session.run(None, {input_name: blob})[0]
    return postprocess(out, h0, w0, r, conf_thres)


def run_zoomed_out(img, conf_thres, factor=2):
    """Place the image in the middle of a larger gray canvas so big close-up
    objects look smaller to the model, then map boxes back to the original image."""
    h, w = img.shape[:2]
    canvas = np.full((h * factor, w * factor, 3), 114, dtype=np.uint8)
    oy, ox = (h * factor - h) // 2, (w * factor - w) // 2
    canvas[oy:oy + h, ox:ox + w] = img

    out = []
    for d in run_model(canvas, conf_thres):
        b = d["box"]
        x1 = min(max(b["x1"] - ox, 0.0), float(w))
        y1 = min(max(b["y1"] - oy, 0.0), float(h))
        x2 = min(max(b["x2"] - ox, 0.0), float(w))
        y2 = min(max(b["y2"] - oy, 0.0), float(h))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        d["box"] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        out.append(d)
    return out


def best_conf(dets):
    return max((d["confidence"] for d in dets), default=0.0)


@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = 0.25, tta: bool = True):
    """Upload an image and get helmet / no-helmet / person detections.

    conf: minimum confidence to return (0.01 - 0.99)
    tta:  if True, retry on a zoomed-out copy when the first pass is not confident
    """
    try:
        conf = min(max(conf, 0.01), 0.99)
        image_bytes = await file.read()
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Uploaded file is not a valid image")
        h0, w0 = img.shape[:2]

        detections = run_model(img, conf)
        zoom_used = False

        if tta and best_conf(detections) < ZOOM_OUT_BELOW:
            alt = run_zoomed_out(img, conf)
            if best_conf(alt) > best_conf(detections):
                detections, zoom_used = alt, True

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        summary = {n: sum(d["class"] == n for d in detections) for n in CLASS_NAMES}

        return {
            "filename": file.filename,
            "width": w0,
            "height": h0,
            "zoom_out_used": zoom_used,
            "summary": summary,
            "detections": detections,
        }
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e), "detections": []})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "detections": []})


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True, "classes": CLASS_NAMES}
