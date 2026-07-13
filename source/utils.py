from pathlib import Path
import csv
import time
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchinfo import summary


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def set_seed(seed=42):
    """
    Chức năng: cố định seed cho thí nghiệm có khả năng lặp lại tốt hơn.
    Input: seed là số nguyên.
    Output: không trả về giá trị.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SegLoss(nn.Module):
    """Loss gọn cho segmentation nhị phân: CrossEntropy + Dice foreground."""

    def __init__(self, dice_weight=1.0):
        """
        Chức năng: khởi tạo trọng số Dice loss.
        Input: dice_weight là hệ số nhân cho Dice loss.
        Output: module loss.
        """
        super().__init__()
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        """
        Chức năng: tính tổng loss từ logits và mask nhãn.
        Input: logits [B,2,H,W], target [B,H,W] với nhãn 0/1.
        Output: scalar loss để backpropagation.
        """
        ce = F.cross_entropy(logits, target)
        prob = torch.softmax(logits, dim=1)[:, 1]
        fg = (target == 1).float()
        inter = (prob * fg).sum()
        dice = 1.0 - (2.0 * inter + 1.0) / (prob.sum() + fg.sum() + 1.0)
        return ce + self.dice_weight * dice


class ConfusionMeter:
    """Bộ cộng dồn confusion matrix cho segmentation 2 lớp."""

    def __init__(self, num_classes=2):
        """
        Chức năng: tạo ma trận nhầm lẫn kích thước CxC.
        Input: num_classes số lớp phân đoạn.
        Output: đối tượng metric có thể update và compute.
        """
        self.num_classes = num_classes
        self.mat = np.zeros((num_classes, num_classes), dtype=np.float64)

    def update(self, pred, target):
        """
        Chức năng: cập nhật confusion matrix từ batch dự đoán.
        Input: pred và target tensor/ndarray [B,H,W] cùng kích thước.
        Output: không trả về giá trị, cập nhật self.mat.
        """
        pred = pred.detach().cpu().numpy() if torch.is_tensor(pred) else pred
        target = target.detach().cpu().numpy() if torch.is_tensor(target) else target
        assert pred.shape == target.shape
        mask = (target >= 0) & (target < self.num_classes)
        hist = np.bincount(
            self.num_classes * target[mask].astype(int) + pred[mask].astype(int),
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        self.mat += hist

    def compute(self):
        """
        Chức năng: tính pixel accuracy, IoU từng lớp, mIoU, precision, recall, F1.
        Input: không có, dùng confusion matrix đã cộng dồn.
        Output: dict chứa các chỉ số đánh giá.
        """
        eps = 1e-12
        diag = np.diag(self.mat)
        union = self.mat.sum(1) + self.mat.sum(0) - diag
        iou = diag / (union + eps)
        tp, fp, fn = self.mat[1, 1], self.mat[0, 1], self.mat[1, 0]
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        return {
            "pixel_acc": diag.sum() / (self.mat.sum() + eps),
            "iou_bg": iou[0],
            "iou_da": iou[1],
            "miou": float(iou.mean()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


def poly_lr(optimizer, base_lr, epoch, max_epochs, power=1.0):
    """
    Chức năng: cập nhật learning rate theo lịch poly.
    Input: optimizer, base_lr, epoch hiện tại, max_epochs, power.
    Output: learning rate mới.
    """
    lr = base_lr * (1 - epoch / max_epochs) ** power
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, amp=False):
    """
    Chức năng: huấn luyện mô hình trong một epoch.
    Input: model, dataloader, loss, optimizer, device, scaler AMP, cờ amp.
    Output: loss trung bình của epoch.
    """
    model.train()
    total_loss = 0.0
    for images, masks, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            loss = criterion(model(images), masks)

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler is None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp=False):
    """
    Chức năng: đánh giá mô hình trên validation/test set.
    Input: model, dataloader, criterion hoặc None, device, cờ amp.
    Output: dict metric gồm loss, mIoU, IoU DA, F1, precision, recall, pixel_acc.
    """
    model.eval()
    meter = ConfusionMeter(num_classes=2)
    total_loss = 0.0
    for images, masks, _ in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images)
            loss = criterion(logits, masks) if criterion is not None else torch.tensor(0.0, device=device)
        pred = logits.argmax(1)
        meter.update(pred, masks)
        total_loss += loss.item() * images.size(0)
    metrics = meter.compute()
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def save_checkpoint(path, model, optimizer, epoch, best_miou):
    """
    Chức năng: lưu checkpoint huấn luyện.
    Input: path đường dẫn lưu, model, optimizer, epoch, best_miou.
    Output: file .pt trên ổ đĩa.
    """
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_miou": best_miou}, path)


def load_weights(model, weight, device):
    """
    Chức năng: nạp trọng số cho mô hình từ checkpoint hoặc state_dict.
    Input: model, weight đường dẫn .pt/.pth, device.
    Output: model đã nạp trọng số.
    """
    ckpt = torch.load(weight, map_location=device)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    return model


def count_params(model):
    """
    Chức năng: đếm số tham số trainable và tổng tham số.
    Input: model PyTorch.
    Output: tuple (total_params, trainable_params).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def profile_inference(model, img_size=(384, 640), device="cuda", warmup=20, iters=100):
    """
    Chức năng: đo Params, MACs, latency và FPS thống nhất cho một mô hình.
    Input: model, img_size=(H,W), device, số vòng warmup và số vòng đo.
    Output: dict chứa params, MACs, GFLOPs xấp xỉ, latency_ms, FPS.
    """
    model.eval().to(device)
    h, w = img_size
    x = torch.randn(1, 3, h, w, device=device)
    info = summary(model, input_size=(1, 3, h, w), device=device, verbose=0)
    for _ in range(warmup):
        _ = model(x)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    latency = (time.perf_counter() - t0) * 1000.0 / iters
    total, trainable = count_params(model)
    macs = int(info.total_mult_adds)
    return {"params": total, "trainable_params": trainable, "macs": macs, "gflops": macs / 1e9, "latency_ms": latency, "fps": 1000.0 / latency}


def list_images(source):
    """
    Chức năng: liệt kê ảnh từ một file hoặc một thư mục.
    Input: source là đường dẫn ảnh hoặc thư mục ảnh.
    Output: list Path ảnh.
    """
    p = Path(source)
    return [p] if p.is_file() else sorted([x for x in p.glob("*.*") if x.suffix.lower() in IMG_EXTS])


def preprocess_image(path, img_size, device):
    """
    Chức năng: đọc và chuẩn hóa một ảnh cho inference.
    Input: path ảnh, img_size=(H,W), device.
    Output: tensor [1,3,H,W] và ảnh BGR gốc.
    """
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert image_bgr is not None
    h, w = img_size
    image = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = (image - MEAN) / STD
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    return image, image_bgr


def overlay_mask(image_bgr, mask, alpha=0.45):
    """
    Chức năng: phủ mask DA màu xanh lên ảnh gốc.
    Input: image_bgr ndarray [H,W,3], mask ndarray [H,W] nhãn 0/1, alpha độ trong suốt.
    Output: ảnh BGR đã overlay.
    """
    color = np.zeros_like(image_bgr)
    color[mask == 1] = (0, 255, 0)
    return np.where(color > 0, (image_bgr * (1 - alpha) + color * alpha).astype(np.uint8), image_bgr)


def write_csv(path, rows, fieldnames):
    """
    Chức năng: ghi danh sách dict thành file CSV.
    Input: path file CSV, rows danh sách bản ghi, fieldnames tên cột.
    Output: file CSV trên ổ đĩa.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
