"""
ChestXray14 Classification API
================================
• Automatically picks the best .pth checkpoint from MODEL_DIR (highest val_auc)
• Deterministic results: same image → same output (hash-based caching)
• Auto-reconstructs classifier head from checkpoint weights (handles any architecture)
• Endpoints:
    GET  /health          → server status + loaded model info
    POST /analyze         → upload image, get predictions + probabilities
    GET  /classes         → list of 14 disease classes
"""

import os
import io
import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as torchvision_models
from PIL import Image
import torchvision.transforms as transforms

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

from flask import Flask, request, jsonify
from flask_cors import CORS

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR     = Path(os.getenv("MODEL_DIR", "./models"))
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD     = float(os.getenv("THRESHOLD", "0.5"))
CHEST_SERVICE_API_KEY = os.getenv("CHEST_SERVICE_API_KEY", "").strip()
IMG_SIZE      = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
MAX_FILE_MB   = 20
MAX_CACHE     = 512

CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# HEAD RECONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_head(state_dict: dict, prefix: str) -> nn.Sequential:
    """
    Rebuild a Sequential classifier head exactly as saved.
    Works by walking indices 0..max present; gaps become Dropout(0.3).
    Handles: Linear, BatchNorm1d, Dropout (no weights saved for dropout).
    """
    cls_keys = {k: v for k, v in state_dict.items() if k.startswith(prefix + ".")}
    if not cls_keys:
        return None

    present = sorted(set(int(k.split(".")[len(prefix.split("."))]) for k in cls_keys))
    max_idx = present[-1]

    # Build type map for indices that have weights
    idx_map = {}
    for idx in present:
        w_key  = f"{prefix}.{idx}.weight"
        rm_key = f"{prefix}.{idx}.running_mean"
        if w_key in state_dict and state_dict[w_key].dim() == 2:
            in_f, out_f = state_dict[w_key].shape[1], state_dict[w_key].shape[0]
            idx_map[idx] = ("linear", in_f, out_f)
        elif rm_key in state_dict:
            idx_map[idx] = ("bn", state_dict[rm_key].shape[0])
        # num_batches_tracked-only keys are part of a BN already captured above

    # Walk full range; gaps = Dropout
    layers = []
    for idx in range(max_idx + 1):
        if idx in idx_map:
            kind = idx_map[idx][0]
            if kind == "linear":
                layers.append(nn.Linear(idx_map[idx][1], idx_map[idx][2]))
            elif kind == "bn":
                layers.append(nn.BatchNorm1d(idx_map[idx][1]))
        else:
            layers.append(nn.Dropout(0.3))

    log.info(f"  Reconstructed head ({prefix}): {[type(l).__name__ for l in layers]}")
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL BUILDING
# ─────────────────────────────────────────────────────────────────────────────

def build_model(model_name: str, num_classes: int = 14, state_dict: dict = None):
    """
    Build model skeleton, then reconstruct the classifier head from state_dict
    so the architecture matches exactly what was trained — regardless of how
    complex the head is.
    """
    if model_name.startswith("efficientnet"):
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm not installed — cannot load EfficientNet model")
        model = timm.create_model(model_name, pretrained=False)
        in_f  = model.classifier.in_features
        head  = reconstruct_head(state_dict, "classifier") if state_dict else None
        model.classifier = head if head else nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_f, num_classes))

    elif model_name == "resnet50":
        model = torchvision_models.resnet50(weights=None)
        in_f  = model.fc.in_features
        head  = reconstruct_head(state_dict, "fc") if state_dict else None
        model.fc = head if head else nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_f, num_classes))

    elif model_name == "densenet121":
        model = torchvision_models.densenet121(weights=None)
        in_f  = model.classifier.in_features
        head  = reconstruct_head(state_dict, "classifier") if state_dict else None
        model.classifier = head if head else nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_f, num_classes))

    else:
        raise ValueError(f"Unknown model architecture: {model_name}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def find_best_checkpoint(model_dir: Path):
    candidates = []
    for p in model_dir.glob("*.pth"):
        try:
            ck  = torch.load(p, map_location="cpu", weights_only=False)
            auc = float(ck.get("val_auc", -1))
            candidates.append((auc, p, ck))
            log.info(f"  Found checkpoint: {p.name}  val_auc={auc:.4f}")
        except Exception as e:
            log.warning(f"  Skipping {p.name}: {e}")

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_auc, best_path, best_ck = candidates[0]
    log.info(f"✅ Best checkpoint: {best_path.name}  val_auc={best_auc:.4f}")
    return best_path, best_ck


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP LOAD
# ─────────────────────────────────────────────────────────────────────────────

_model      = None
_model_meta = {}

def load_model():
    global _model, _model_meta

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_path, checkpoint = find_best_checkpoint(MODEL_DIR)

    if best_path is None:
        log.warning("⚠️  No checkpoint found. API returns 503 until a .pth is placed in models/")
        return

    cfg         = checkpoint.get("cfg", {})
    model_name  = cfg.get("model_name", "efficientnet_b0")
    num_classes = cfg.get("num_classes", 14)
    classes     = checkpoint.get("classes", CLASSES)
    sd          = checkpoint["model_state_dict"]

    model = build_model(model_name, num_classes, state_dict=sd)
    model.load_state_dict(sd)
    model.eval()
    model.to(DEVICE)

    _model = model
    _model_meta = {
        "checkpoint":    best_path.name,
        "model_name":    model_name,
        "num_classes":   num_classes,
        "val_auc":       checkpoint.get("val_auc"),
        "val_f1":        checkpoint.get("val_f1"),
        "trained_epoch": checkpoint.get("epoch"),
        "classes":       list(classes),
        "threshold":     THRESHOLD,
        "device":        str(DEVICE),
    }
    log.info(f"🧠 Model ready: {model_name} | AUC={_model_meta['val_auc']} | device={DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE  (deterministic + cached)
# ─────────────────────────────────────────────────────────────────────────────

_inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

_result_cache: dict = {}


def image_hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def run_inference(raw: bytes) -> dict:
    h = image_hash(raw)
    if h in _result_cache:
        log.info(f"Cache hit: {h[:12]}…")
        return _result_cache[h]

    image  = Image.open(io.BytesIO(raw)).convert("RGB")
    tensor = _inference_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.sigmoid(_model(tensor)).squeeze().cpu().numpy()

    classes   = _model_meta.get("classes", CLASSES)
    threshold = THRESHOLD

    predictions = sorted(
        [{"condition": c, "probability": round(float(p), 4), "detected": float(p) >= threshold}
         for c, p in zip(classes, probs)],
        key=lambda x: x["probability"], reverse=True
    )
    detected = [p for p in predictions if p["detected"]]

    result = {
        "image_hash":  h,
        "threshold":   threshold,
        "predictions": predictions,
        "detected":    detected,
        "no_finding":  len(detected) == 0,
        "model":       _model_meta.get("model_name"),
        "checkpoint":  _model_meta.get("checkpoint"),
    }

    if len(_result_cache) >= MAX_CACHE:
        del _result_cache[next(iter(_result_cache))]
    _result_cache[h] = result
    return result


# ─────────────────────────────────────────────────────────────────────────────
# FLASK
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

load_model()


@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    ready = _model is not None
    return jsonify({
        "status": "ok" if ready else "no_model",
        "model_ready": ready,
        "service": "chest-xray-classifier",
        "classes": _model_meta.get("classes", CLASSES) if _model_meta else CLASSES,
        "model": _model_meta or None,
        "endpoints": {
            "health": "GET /health",
            "classes": "GET /classes",
            "analyze": "POST /analyze (multipart field: image)",
        },
    })


@app.route("/classes", methods=["GET"])
def get_classes():
    return jsonify({"classes": _model_meta.get("classes", CLASSES)})


def _verify_service_key():
    if not CHEST_SERVICE_API_KEY:
        return None
    key = request.headers.get("X-Chest-Service-Key", "")
    if key != CHEST_SERVICE_API_KEY:
        return jsonify({"error": "Invalid or missing X-Chest-Service-Key"}), 401
    return None


@app.route("/analyze", methods=["POST"])
def analyze():
    auth_err = _verify_service_key()
    if auth_err:
        return auth_err
    if _model is None:
        return jsonify({"error": "No model loaded. Place a .pth checkpoint in models/"}), 503
    if "image" not in request.files:
        return jsonify({"error": "No 'image' field in request."}), 400

    raw = request.files["image"].read()
    if not raw:
        return jsonify({"error": "Empty file."}), 400
    if len(raw) > MAX_FILE_MB * 1024 * 1024:
        return jsonify({"error": f"File exceeds {MAX_FILE_MB} MB."}), 413

    try:
        return jsonify(run_inference(raw))
    except Exception as e:
        log.exception("Inference error")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
