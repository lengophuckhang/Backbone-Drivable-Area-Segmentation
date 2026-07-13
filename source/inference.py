from argparse import ArgumentParser
from pathlib import Path
import cv2
import torch
import torch.nn.functional as F

from model import build_model
from utils import list_images, preprocess_image, overlay_mask, load_weights


def parse_args():
    """
    Chức năng: khai báo tham số inference cho ảnh hoặc thư mục ảnh.
    Input: tham số CLI từ terminal.
    Output: Namespace chứa checkpoint, source và kích thước ảnh.
    """
    p = ArgumentParser()
    p.add_argument("--source", default="samples")
    p.add_argument("--save_dir", default="outputs")
    p.add_argument("--weight", default="runs/resnet18/resnet18_best.pt")
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--decoder_ch", type=int, default=128)
    p.add_argument("--img_h", type=int, default=384)
    p.add_argument("--img_w", type=int, default=640)
    p.add_argument("--alpha", type=float, default=0.45)
    return p.parse_args()


@torch.no_grad()
def main():
    """
    Chức năng: chạy suy luận DA và lưu ảnh overlay.
    Input: source ảnh/thư mục ảnh và checkpoint.
    Output: ảnh kết quả trong save_dir.
    """
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args.backbone, decoder_ch=args.decoder_ch, pretrained=False).to(device)
    load_weights(model, args.weight, device).eval()

    for path in list_images(args.source):
        tensor, image_bgr = preprocess_image(path, (args.img_h, args.img_w), device)
        logits = model(tensor)
        mask = logits.argmax(1)[0].cpu().numpy().astype("uint8")
        mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        result = overlay_mask(image_bgr, mask, args.alpha)
        cv2.imwrite(str(save_dir / Path(path).name), result)
        print(save_dir / Path(path).name)


if __name__ == "__main__":
    main()
