from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _first_existing(paths):
    """
    Chức năng: lấy đường dẫn đầu tiên đang tồn tại trong danh sách ứng viên.
    Input: paths là list Path.
    Output: Path hợp lệ đầu tiên hoặc None.
    """
    for p in paths:
        if p.exists():
            return p
    return None


class BDD100KDA(Dataset):
    """Dataset single-task cho Drivable Area Segmentation trên BDD100K."""

    def __init__(self, root="BDD100K", split="train", img_size=(384, 640), augment=False):
        """
        Chức năng: tạo danh sách ảnh và mask DA theo split.
        Input: root thư mục BDD100K, split train/val, img_size=(H,W), augment bật/tắt tăng cường.
        Output: đối tượng Dataset trả về image tensor, mask tensor, đường dẫn ảnh.
        """
        self.root = Path(root)
        self.split = split
        self.img_h, self.img_w = img_size
        self.augment = augment

        self.image_dir = _first_existing([
            self.root / "images" / split,      # dạng: BDD100K/images/train
            self.root / "100k" / split,        # dạng đang dùng: BDD100K/100k/train
            self.root / split,                 # khi truyền --data_root BDD100K/100k
        ])
        self.mask_dir = _first_existing([
            self.root / "drivable_area_annotations" / split,  # dạng source tham khảo
            self.root / "bdd_seg_gt" / split,                 # dạng đang dùng trong hình
            self.root.parent / "bdd_seg_gt" / split,          # khi truyền --data_root BDD100K/100k
        ])
        assert self.image_dir is not None and self.mask_dir is not None
        self.images = sorted(self.image_dir.glob("*.jpg")) + sorted(self.image_dir.glob("*.png"))
        self.masks = [self.mask_dir / f"{p.stem}.png" for p in self.images]
        assert len(self.images) > 0

    def __len__(self):
        """
        Chức năng: trả về số mẫu trong dataset.
        Input: không có.
        Output: số lượng ảnh.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Chức năng: đọc một ảnh và mask DA, resize, chuẩn hóa, chuyển sang tensor.
        Input: idx là chỉ số mẫu.
        Output: image FloatTensor [3,H,W], mask LongTensor [H,W], path ảnh.
        """
        image = cv2.imread(str(self.images[idx]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self.masks[idx]), cv2.IMREAD_GRAYSCALE)
        assert image is not None and mask is not None

        if self.augment:
            image, mask = self._augment(image, mask)

        image = cv2.resize(image, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = (image - MEAN) / STD
        mask = (mask > 0).astype(np.int64)  # 0: background, 1: drivable area

        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).long()
        return image, mask, str(self.images[idx])

    def _augment(self, image, mask):
        """
        Chức năng: tăng cường dữ liệu tối giản cho ảnh và mask.
        Input: image BGR ndarray, mask grayscale ndarray.
        Output: image và mask đã biến đổi đồng bộ.
        """
        if np.random.rand() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])

        if np.random.rand() < 0.5:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + np.random.uniform(-10, 10)) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] * np.random.uniform(0.8, 1.2), 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] * np.random.uniform(0.8, 1.2), 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return image, mask
