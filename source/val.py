from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader

from dataset import BDD100KDA
from model import build_model
from utils import SegLoss, evaluate, load_weights


def parse_args():
    """
    Chức năng: khai báo tham số validation, không dùng file cấu hình.
    Input: tham số CLI từ terminal.
    Output: Namespace chứa đường dẫn và hyperparameter đánh giá.
    """
    p = ArgumentParser()
    p.add_argument("--data_root", default="BDD100K")
    p.add_argument("--weight", default="runs/resnet18/resnet18_best.pt")
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--decoder_ch", type=int, default=128)
    p.add_argument("--img_h", type=int, default=384)
    p.add_argument("--img_w", type=int, default=640)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def main():
    """
    Chức năng: nạp checkpoint và đánh giá trên tập val BDD100K DA.
    Input: tham số từ parse_args().
    Output: in các metric chính ra màn hình.
    """
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.amp and device.type == "cuda"

    val_set = BDD100KDA(args.data_root, "val", (args.img_h, args.img_w), augment=False)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.backbone, decoder_ch=args.decoder_ch, pretrained=False).to(device)
    load_weights(model, args.weight, device)
    metrics = evaluate(model, val_loader, SegLoss(), device, amp)

    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
