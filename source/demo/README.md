# 🚗 Drivable Area Segmentation — Web Demo

Trang web demo trực quan cho đồ án **Phân đoạn vùng làn đường (Drivable Area Segmentation)** — Khóa luận tốt nghiệp.

Cho phép tải lên ảnh gốc, chọn model đã huấn luyện, và xem kết quả phân đoạn vùng drivable area được tô màu xanh ngay trên giao diện.

---

## ✨ Tính năng

- **🧠 13 model đã train** — Chọn từ dropdown, từ nhẹ (`ghostnet_100`, `resnet18`) đến chính xác (`pvt_v2_b2`, `efficientnet_b3`)
- **🖼️ Upload ảnh JPG** — Ảnh đầu vào là làn đường cần phân đoạn
- **🏷️ Ground Truth (tuỳ chọn)** — Upload mask PNG/JPG để so sánh kết quả dự đoán với ground truth
- **🎨 Điều chỉnh độ trong suốt** — Thanh trượt alpha cho overlay
- **📊 Metrics model** — Hiển thị IoU_DA, F1, FPS, Params, GFLOPs từ benchmark
- **📐 Thống kê** — Tỉ lệ drivable area (%), thời gian inference
- **⚠️ Cảnh báo kích thước** — Tự động phát hiện nếu ảnh gốc và GT không cùng kích thước
- **📈 So sánh GT** — IoU, F1, Precision, Recall giữa prediction và ground truth

---

## 🖥️ Giao diện

```
┌─────────────────────────────────────────────────────┐
│  🧠 Model              🎨 Alpha        🚀 Phân đoạn │
│  [pvt_v2_b2        ▼]  [═══●══════]   [   Phân đoạn]│
│                                                     │
│  🖼️ Ảnh đầu vào        🏷️ Ground Truth (tuỳ chọn)   │
│  [📁 Chọn ảnh (JPG)]   [🏷️ Chọn mask ground truth]  │
├─────────────────────────────────────────────────────┤
│  IoU: 0.8863  │  F1: 0.9397  │  FPS: 40.1  │  ...  │
├─────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ Ảnh gốc   │  │ Kết quả   │  │ GT Ref    │         │
│  │ (Input)   │  │ (Output)  │  │ (tuỳ chọn)│         │
│  └──────────┘  └──────────┘  └──────────┘         │
│  📐 Drivable: 21.43%  🧠 Model: pvt_v2_b2  ⚡ 1.32s │
│  🎯 IoU vs GT: 0.89  📊 F1 vs GT: 0.94  ...        │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 Hướng dẫn chạy

### Yêu cầu

- Python 3.10+
- Các thư viện trong `requirements.txt` (từ thư mục gốc)
- Checkpoint các model đã train trong thư mục `runs/`

### Cài đặt

```bash
# Di chuyển đến thư mục source
cd source

# Cài đặt dependencies
pip install -r requirements.txt
pip install flask
```

### Chạy web demo

```bash
cd demo
python app.py
```

Mở trình duyệt và truy cập: **http://127.0.0.1:5000**

> ⚡ Nếu có GPU, model sẽ tự động chạy trên CUDA để inference nhanh hơn.

---

## 📂 Cấu trúc thư mục

```
source/
├── demo/                     # Web demo
│   ├── app.py                # Flask backend
│   ├── templates/
│   │   └── index.html        # Giao diện người dùng
│   ├── static/               # Assets tĩnh
│   ├── uploads/              # Ảnh upload tạm thời
│   └── README.md             # Hướng dẫn này
├── runs/                     # Checkpoint các model đã train
│   ├── resnet18/
│   ├── pvt_v2_b2/
│   ├── efficientnet_b3/
│   └── ...                   # 13 models
├── model.py                  # Định nghĩa model DASegmenter
├── utils.py                  # Hàm tiện ích
├── inference.py              # CLI inference
└── ...
```

---

## 🧠 Các model khả dụng

| Model | IoU_DA ↑ | F1 ↑ | FPS ↑ | Params | GFLOPs |
|-------|----------|------|-------|--------|--------|
| **pvt_v2_b2** | **0.8863** | **0.9397** | 40.1 | 25.7M | 8.15 |
| efficientnet_b3 | 0.8843 | 0.9386 | 62.2 | 10.9M | 9.92 |
| convnext_tiny | 0.8842 | 0.9385 | 70.0 | 28.7M | 7.07 |
| efficientnet_b0 | 0.8814 | 0.9370 | 97.2 | 4.4M | 7.11 |
| mobilenetv3_large_100 | 0.8789 | 0.9356 | **129.7** | 3.9M | 6.39 |
| resnet18 | 0.8719 | 0.9316 | **179.4** | 12.0M | 14.36 |
| ghostnet_100 | 0.8713 | 0.9312 | 127.8 | **1.7M** | **5.81** |

> **Mẹo**: Chọn `resnet18` / `mobilenetv3_large_100` nếu cần tốc độ cao. Chọn `pvt_v2_b2` / `efficientnet_b3` nếu cần độ chính xác cao nhất.

---

## 🔗 API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/` | Trang chủ |
| GET | `/api/models` | Danh sách model (JSON) |
| POST | `/api/predict` | Dự đoán (multipart: image + ground_truth tuỳ chọn) |

### Ví dụ API predict

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -F "image=@anh.jpg" \
  -F "ground_truth=@mask.png" \
  -F "model=pvt_v2_b2" \
  -F "alpha=0.45"
```

---
