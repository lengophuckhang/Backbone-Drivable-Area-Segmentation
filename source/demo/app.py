"""Web demo cho Drivable Area Segmentation (KLTN)."""

import sys
import os
import time
import uuid
from pathlib import Path
from io import BytesIO
import base64
import csv

import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify, Response, session

# Thêm thư mục gốc vào sys.path để import được model, utils
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import build_model
from utils import load_weights, preprocess_image, overlay_mask, MEAN, STD

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = ROOT / "demo" / "uploads"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB
app.secret_key = "kltn_drivable_seg_2026"

# Lưu trữ session video: session_id -> {"filepath": Path}
_video_sessions = {}

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


@app.route("/api/device", methods=["GET"])
def get_device():
    """API trả về thông tin thiết bị (CPU/GPU)."""
    has_cuda = torch.cuda.is_available()
    return jsonify({
        "device": "cuda" if has_cuda else "cpu",
        "cuda_available": has_cuda,
        "cuda_version": torch.version.cuda if has_cuda else None,
        "device_count": torch.cuda.device_count() if has_cuda else 0,
    })


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


# ===== Video Processing =====


def process_video_frame(frame_bgr, model, device, img_h, img_w, alpha):
    """Xử lý một frame video: resize → normalize → inference → overlay.

    Args:
        frame_bgr: numpy array (H,W,3) ảnh BGR từ OpenCV.
        model: torch model.
        device: torch device.
        img_h, img_w: kích thước đầu vào model.
        alpha: độ trong suốt overlay.

    Returns:
        result_bgr: ảnh BGR đã overlay mask.
        mask: mask nhị phân (0/1).
        drivable_pct: phần trăm drivable area.
        inference_ms: thời gian inference (ms).
    """
    t0 = time.perf_counter()

    # Resize frame về kích thước model
    frame_resized = cv2.resize(frame_bgr, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

    # Chuẩn hoá
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    frame_norm = (frame_rgb - MEAN) / STD
    tensor = torch.from_numpy(frame_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = model(tensor)
        pred = logits.argmax(1)[0].cpu().numpy().astype("uint8")

    # Overlay
    result = overlay_mask(frame_resized, pred, alpha)

    # Thống kê
    drivable_pct = float(pred.sum() / pred.size * 100)
    inference_ms = (time.perf_counter() - t0) * 1000

    return result, pred, drivable_pct, inference_ms


@app.route("/api/predict_video", methods=["POST"])
def predict_video():
    """API nhận video → upload → trả về URL stream."""
    if "video" not in request.files:
        return jsonify({"error": "Không có file video"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Chưa chọn file"}), 400

    model_name = request.form.get("model", DEFAULT_MODEL["name"] if DEFAULT_MODEL else None)
    if not model_name:
        return jsonify({"error": "Không có model nào được chọn"}), 400

    alpha = float(request.form.get("alpha", 0.45))
    img_h = int(request.form.get("img_h", 384))
    img_w = int(request.form.get("img_w", 640))

    # Tạo session ID và lưu video với tên unique
    session_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix if Path(file.filename).suffix else ".mp4"
    save_name = f"video_{session_id}{ext}"
    filepath = app.config["UPLOAD_FOLDER"] / save_name
    file.save(str(filepath))

    # Lưu thông tin session
    _video_sessions[session_id] = {
        "filepath": filepath,
        "model_name": model_name,
        "alpha": alpha,
        "img_h": img_h,
        "img_w": img_w,
    }

    stream_url = f"/api/video_stream/{session_id}"
    return jsonify({
        "stream_url": stream_url,
        "session_id": session_id,
        "filename": file.filename,
    })


@app.route("/api/video_stream/<session_id>")
def video_stream(session_id):
    """API MJPEG stream: đọc video từ session → inference → stream frame."""
    session_data = _video_sessions.get(session_id)
    if session_data is None:
        return jsonify({"error": "Session không hợp lệ hoặc đã hết hạn"}), 404

    filepath = session_data["filepath"]
    model_name = session_data["model_name"]
    alpha = session_data["alpha"]
    img_h = session_data["img_h"]
    img_w = session_data["img_w"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate():
        try:
            model = get_model(model_name)
            cap = cv2.VideoCapture(str(filepath))
            if not cap.isOpened():
                yield b'--frame\r\nContent-Type: text/plain\r\n\r\nError: Cannot open video\r\n'
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                total_frames = 0
            frame_idx = 0
            total_inference_ms = 0.0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # Process frame
                result, pred_mask, drivable_pct, inf_ms = process_video_frame(
                    frame, model, device, img_h, img_w, alpha
                )
                total_inference_ms += inf_ms
                avg_inference_ms = total_inference_ms / frame_idx
                current_fps = 1000.0 / avg_inference_ms if avg_inference_ms > 0 else 0

                # Vẽ thông tin lên frame
                info_lines = [
                    f"Frame: {frame_idx}/{total_frames if total_frames > 0 else '?'}",
                    f"{inf_ms:.0f} ms | {current_fps:.1f} FPS | {model_name}",
                ]
                for i, line in enumerate(info_lines):
                    y = 28 + i * 26
                    cv2.rectangle(result, (8, y - 18), (520, y + 4), (0, 0, 0), cv2.FILLED)
                    cv2.putText(result, line, (14, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Encode frame thành JPEG
                _, buffer = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            cap.release()

        except Exception as e:
            yield (b'--frame\r\n'
                   b'Content-Type: text/plain\r\n\r\nError: ' + str(e).encode() + b'\r\n')
        finally:
            filepath.unlink(missing_ok=True)
            _video_sessions.pop(session_id, None)

    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )


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
        # Load model
        model = get_model(model_name)

        # Đọc ảnh 1 lần duy nhất
        image_bgr = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError("Không thể đọc ảnh đầu vào")
        orig_img_h, orig_img_w = image_bgr.shape[:2]

        # Tiền xử lý (resize + normalize)
        frame_resized = cv2.resize(image_bgr, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frame_norm = (frame_rgb - MEAN) / STD
        tensor = torch.from_numpy(frame_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

        # Inference — chỉ đo thời gian forward
        t_infer = time.perf_counter()
        with torch.no_grad():
            logits = model(tensor)
            pred = logits.argmax(1)[0].cpu().numpy().astype("uint8")
        inference_ms = (time.perf_counter() - t_infer) * 1000

        # Overlay
        result = overlay_mask(frame_resized, pred, alpha)

        # Encode ảnh
        orig_b64 = encode_image_to_b64(frame_resized)
        result_b64 = encode_image_to_b64(result)

        # Tính phần trăm drivable area (pred là mask nhị phân 0/1)
        drivable_pct = float(pred.sum() / pred.size * 100)

        # Khởi tạo response
        response = {
            "original": orig_b64,
            "result": result_b64,
            "drivable_pct": round(drivable_pct, 2),
            "model": model_name,
            "orig_size": {"w": orig_img_w, "h": orig_img_h},
            "inference_ms": round(inference_ms, 1),
        }

        # Xử lý ground truth — giữ nguyên ảnh gốc người dùng import
        gt_b64 = None
        gt_metrics = None
        if gt_path and gt_path.exists():
            gt_img = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
            if gt_img is not None:

                # Resize GT về cùng kích thước frame_resized để hiển thị đồng bộ
                gt_h, gt_w = frame_resized.shape[:2]
                gt_img_resized = cv2.resize(gt_img, (gt_w, gt_h), interpolation=cv2.INTER_NEAREST)
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
                gt_metrics = compute_seg_metrics(pred, gt_mask)

                response["ground_truth"] = gt_b64
                response["gt_metrics"] = gt_metrics

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
