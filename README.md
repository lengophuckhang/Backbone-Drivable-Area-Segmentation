# BDD100K Drivable Area Segmentation

## 1. Cài đặt

```bash
pip install -r requirements.txt
```

Các thư viện chính: `torch`, `torchvision`, `timm`, `torchinfo`, `opencv-python`, `numpy`, `tqdm`, `ultralytics`.

## 2. Dataset

Cấu trúc phù hợp với thư mục đang dùng:

```text
BDD100K/
├── 100k/
│   ├── train/
│   ├── val/
│   └── test/
└── bdd_seg_gt/
    ├── train/
    └── val/
```

Ảnh và mask cần cùng tên stem:

```text
BDD100K/100k/train/0000f77c-62c2a288.jpg
BDD100K/bdd_seg_gt/train/0000f77c-62c2a288.png
```

Mask được đưa về nhị phân:

```text
0: background
1: drivable area, gồm các pixel mask > 0
```

Cũng hỗ trợ cấu trúc:

```text
BDD100K/
├── images/train
├── images/val
├── drivable_area_annotations/train
└── drivable_area_annotations/val
```

Với cây thư mục trong ảnh, dùng `--data_root BDD100K`. Không dùng `--data_root data/bdd100k` nếu không có thư mục đó.

## 3. File chính

```text
dataset.py       # Đọc BDD100K DA, resize, augment nhẹ, chuẩn hóa ảnh
model.py         # Backbone timm + FPN decoder + segmentation head; có torchinfo ở main
utils.py         # Loss, metric, train/eval loop, checkpoint, inference helper, profiling helper
train.py         # Train single-task DA
val.py           # Đánh giá checkpoint trên val set
inference.py     # Suy luận ảnh/thư mục ảnh và lưu overlay
benchmark.py     # Đánh giá nhiều backbone và ghi CSV
requirements.txt
README.md
```

## 4. Train

Mặc định dùng `resnet18`:

```bash
python train.py \
  --data_root BDD100K \
  --out_dir runs/resnet18 \
  --backbone resnet18 \
  --epochs 50 \
  --batch_size 8 \
  --img_h 384 \
  --img_w 640
```

Dùng pretrained ImageNet của `timm`:

```bash
python train.py \
  --data_root BDD100K \
  --out_dir runs/resnet18 \
  --backbone resnet18 \
  --pretrained
```

Checkpoint và log:

```text
runs/resnet18/
├── resnet18_last.pt
├── resnet18_best.pt
└── log.csv
```

## 5. Validation

```bash
python val.py \
  --data_root BDD100K \
  --weight runs/resnet18/resnet18_best.pt \
  --backbone resnet18
```

Metric in ra gồm:

```text
loss
pixel_acc
iou_bg
iou_da
miou
precision
recall
f1
```

## 6. Inference

Suy luận một thư mục ảnh:

```bash
python inference.py \
  --source BDD100K/100k/test \
  --save_dir outputs/resnet18_test \
  --weight runs/resnet18/resnet18_best.pt \
  --backbone resnet18
```

Kết quả là ảnh overlay vùng drivable area màu xanh trong `save_dir`.

## 7. Xem thông tin mô hình bằng torchinfo

```bash
python model.py --backbone resnet18 --img_h 384 --img_w 640
```

Ví dụ backbone khác:

```bash
python model.py --backbone mobilenetv3_large_100 --img_h 384 --img_w 640
python model.py --backbone convnext_tiny --img_h 384 --img_w 640
```

`torchinfo` hiển thị layer, input/output shape, số tham số và Mult-Adds. Latency/FPS phụ thuộc phần cứng nên được đo trong `benchmark.py`.

## 8. Benchmark nhiều backbone

Train từng backbone trước:

```text
runs/
├── resnet18/resnet18_best.pt
├── mobilenetv3_large_100/mobilenetv3_large_100_best.pt
└── convnext_tiny/convnext_tiny_best.pt
```

Sau đó chạy:

```bash
python benchmark.py \
  --data_root BDD100K \
  --weight_dir runs \
  --save_csv runs/benchmark.csv \
  --backbones resnet18 mobilenetv3_large_100 convnext_tiny
```

CSV gồm các cột chính:

```text
backbone, loss, pixel_acc, iou_bg, iou_da, miou, precision, recall, f1,
params, trainable_params, macs, gflops, latency_ms, fps
```

## 9. Đổi backbone

Chỉ cần thay `--backbone` và đổi `--out_dir` tương ứng.

MobileNetV3:

```bash
python train.py \
  --data_root BDD100K \
  --out_dir runs/mobilenetv3_large_100 \
  --backbone mobilenetv3_large_100 \
  --epochs 100 \
  --batch_size 8
```

ConvNeXt-Tiny:

```bash
python train.py \
  --data_root BDD100K \
  --out_dir runs/convnext_tiny \
  --backbone convnext_tiny \
  --epochs 100 \
  --batch_size 4
```

Khi so sánh backbone, nên giữ giống nhau: `img_h`, `img_w`, optimizer, weight decay, train/val split và thiết bị đo tốc độ.
