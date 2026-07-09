"""Web demo cho Drivable Area Segmentation (KLTN)."""

import sys
import os
from pathlib import Path
from io import BytesIO
import base64
import csv

import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify

# Thêm thư mục gốc vào sys.path để import được model, utils
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import build_model
from utils import load_weights, preprocess_image, overlay_mask

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = ROOT / "demo" / "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# ===== Khám phá các model đã train =====
RUNS_DIR = ROOT / "runs"
BENCHMARK_CSV = RUNS_DIR / "benchmark.csv"

# Đọc benchmark để có thông tin metrics
BENCHMARK_DATA = {}  # backbone -> dict metrics
if BENCHMARK_CSV.exists():
    with open(BENCHMARK_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            BENCHMARK_DATA[row["backbone"]] = {
                "miou": float(row["miou"]),
                "iou_da": float(row["iou_da"]),
                "f1": float(row["f1"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "params": int(row["params"]),
                "gflops": float(row["gflops"]),
                "latency_ms": float(row["latency_ms"]),
                "fps": float(row["fps"]),
            }


def discover_models():
    """Quét thư mục runs/ để tìm các checkpoint hợp lệ."""
    models = []
    if not RUNS_DIR.exists():
        return models
    for folder in sorted(RUNS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        best_pt = folder / f"{folder.name}_best.pt"
        if not best_pt.exists():
            continue
        info = {
            "name": folder.name,
            "weight": str(best_pt),
            "best_miou": None,
        }
        # Gắn benchmark metrics nếu có
        if folder.name in BENCHMARK_DATA:
            info.update(BENCHMARK_DATA[folder.name])
        models.append(info)
    return models


AVAILABLE_MODELS = discover_models()
# Mặc định chọn model có IoU_DA cao nhất
DEFAULT_MODEL = max(AVAILABLE_MODELS, key=lambda m: m.get("iou_da", 0)) if AVAILABLE_MODELS else None


# ===== Cache model (lazy loading) =====
_model_cache = {}


def get_model(model_name):
    """Lấy model từ cache hoặc load mới."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = build_model(model_name, decoder_ch=128, pretrained=False).to(device)
        weight_path = ROOT / "runs" / model_name / f"{model_name}_best.pt"
        load_weights(model, str(weight_path), device)
        model.eval()
        _model_cache[model_name] = model
        return model
    except Exception as e:
        raise RuntimeError(f"Không thể load model '{model_name}': {e}")


def encode_image_to_b64(image_bgr):
    """Chuyển ảnh BGR OpenCV sang base64 để nhúng HTML."""
    _, buffer = cv2.imencode(".png", image_bgr)
    return base64.b64encode(buffer).decode("utf-8")


# ===== Routes =====

@app.route("/")
def index():
    """Trang chủ: danh sách model + form upload."""
    return render_template(
        "index.html",
        models=AVAILABLE_MODELS,
        default_model=DEFAULT_MODEL["name"] if DEFAULT_MODEL else None,
    )


@app.route("/api/models", methods=["GET"])
def list_models():
    """API trả về danh sách model dạng JSON."""
    return jsonify(AVAILABLE_MODELS)


def compute_seg_metrics(pred_mask, gt_mask):
    """Tính IoU, Precision, Recall, F1 giữa mask dự đoán và ground truth."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    eps = 1e-12
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {
        "iou_gt": round(float(iou), 4),
        "precision_gt": round(float(precision), 4),
        "recall_gt": round(float(recall), 4),
        "f1_gt": round(float(f1), 4),
    }


@app.route("/api/predict", methods=["POST"])
def predict():
    """API nhận ảnh + ground truth + tên model, trả về ảnh overlay và metrics."""
    if "image" not in request.files:
        return jsonify({"error": "Không có file ảnh"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Chưa chọn file"}), 400

    model_name = request.form.get("model", DEFAULT_MODEL["name"] if DEFAULT_MODEL else None)
    if not model_name:
        return jsonify({"error": "Không có model nào được chọn"}), 400

    alpha = float(request.form.get("alpha", 0.45))
    img_h = int(request.form.get("img_h", 384))
    img_w = int(request.form.get("img_w", 640))

    # Lưu file tạm
    filepath = app.config["UPLOAD_FOLDER"] / file.filename
    file.save(str(filepath))

    # Xử lý ground truth nếu có
    gt_file = request.files.get("ground_truth")
    gt_path = None
    if gt_file and gt_file.filename != "":
        gt_path = app.config["UPLOAD_FOLDER"] / f"gt_{gt_file.filename}"
        gt_file.save(str(gt_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Đọc kích thước gốc của ảnh đầu vào TRƯỚC KHI resize
        orig_img_for_size = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
        if orig_img_for_size is None:
            raise RuntimeError("Không thể đọc ảnh đầu vào")
        orig_img_h = orig_img_for_size.shape[0]
        orig_img_w = orig_img_for_size.shape[1]

        # Load model
        model = get_model(model_name)

        # Tiền xử lý (resize về img_h, img_w cho inference)
        tensor, image_bgr = preprocess_image(str(filepath), (img_h, img_w), device)

        # Inference
        with torch.no_grad():
            logits = model(tensor)
            pred = logits.argmax(1)[0].cpu().numpy().astype("uint8")

        # Resize mask về kích thước ảnh gốc (resized)
        mask = cv2.resize(pred, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Overlay
        result = overlay_mask(image_bgr, mask, alpha)

        # Encode ảnh
        orig_b64 = encode_image_to_b64(image_bgr)
        result_b64 = encode_image_to_b64(result)

        # Tính phần trăm drivable area
        drivable_pct = float(mask.sum() / mask.size * 100)

        # Khởi tạo response
        response = {
            "original": orig_b64,
            "result": result_b64,
            "drivable_pct": round(drivable_pct, 2),
            "model": model_name,
            "orig_size": {"w": orig_img_w, "h": orig_img_h},
        }

        # Xử lý ground truth — giữ nguyên ảnh gốc người dùng import
        gt_b64 = None
        gt_metrics = None
        gt_size_mismatch = False
        if gt_path and gt_path.exists():
            gt_img = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
            if gt_img is not None:
                # Kiểm tra kích thước ảnh GT so với ẢNH GỐC (trước resize)
                gt_h = gt_img.shape[0]
                gt_w = gt_img.shape[1]
                if gt_h != orig_img_h or gt_w != orig_img_w:
                    gt_size_mismatch = True

                # Resize GT về cùng kích thước với ảnh đã resize để hiển thị đồng bộ
                gt_img_resized = cv2.resize(gt_img, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
                # Nếu ảnh grayscale (1 kênh) thì chuyển sang BGR 3 kênh để encode
                if len(gt_img_resized.shape) == 2:
                    gt_display = cv2.cvtColor(gt_img_resized, cv2.COLOR_GRAY2BGR)
                elif gt_img_resized.shape[2] == 4:
                    gt_display = gt_img_resized[:, :, :3]
                else:
                    gt_display = gt_img_resized
                gt_b64 = encode_image_to_b64(gt_display)

                # Tính metrics so sánh prediction vs ground truth (dùng mask nhị phân)
                gt_gray = cv2.cvtColor(gt_display, cv2.COLOR_BGR2GRAY)
                gt_mask = (gt_gray > 0).astype(np.uint8)
                gt_metrics = compute_seg_metrics(mask, gt_mask)

                response["ground_truth"] = gt_b64
                response["gt_metrics"] = gt_metrics
                response["gt_size_mismatch"] = gt_size_mismatch

        # Xoá file tạm
        filepath.unlink(missing_ok=True)
        if gt_path and gt_path.exists():
            gt_path.unlink(missing_ok=True)

        return jsonify(response)

    except Exception as e:
        filepath.unlink(missing_ok=True)
        if gt_path and gt_path.exists():
            gt_path.unlink(missing_ok=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"✔ Phát hiện {len(AVAILABLE_MODELS)} model đã train:")
    for m in AVAILABLE_MODELS:
        iou = m.get("iou_da", "?")
        fps = m.get("fps", "?")
        print(f"   - {m['name']}: IoU_DA={iou}, FPS={fps}")
    if DEFAULT_MODEL:
        print(f"\n► Model mặc định: {DEFAULT_MODEL['name']} (IoU_DA={DEFAULT_MODEL.get('iou_da', '?'):})")

    app.run(host="0.0.0.0", port=5000, debug=True)
