from argparse import ArgumentParser
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from dataset import BDD100KDA
from model import build_model
from utils import SegLoss, evaluate, load_weights, profile_inference, write_csv


def parse_args():
    """
    Chức năng: khai báo tham số benchmark nhiều backbone đã train.
    Input: tham số CLI từ terminal.
    Output: Namespace chứa danh sách backbone, dataset, weight_dir và output CSV.
    """
    p = ArgumentParser()
    p.add_argument("--data_root", default="BDD100K")
    p.add_argument("--weight_dir", default="runs")
    p.add_argument("--save_csv", default="runs/benchmark.csv")
    p.add_argument("--backbones", nargs="+", default=["resnet18", "mobilenetv3_large_100", "convnext_tiny"])
    p.add_argument("--decoder_ch", type=int, default=128)
    p.add_argument("--img_h", type=int, default=384)
    p.add_argument("--img_w", type=int, default=640)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    return p.parse_args()


def main():
    """
    Chức năng: đánh giá độ chính xác và tốc độ của nhiều backbone trong cùng kiến trúc model.
    Input: checkpoint dạng weight_dir/backbone/backbone_best.pt cho từng backbone.
    Output: file CSV gồm mIoU, IoU DA, F1, Params, GFLOPs, latency, FPS.
    """
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_set = BDD100KDA(args.data_root, "val", (args.img_h, args.img_w), augment=False)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    rows = []
    for backbone in args.backbones:
        model = build_model(backbone, decoder_ch=args.decoder_ch, pretrained=False).to(device)
        load_weights(model, Path(args.weight_dir) / backbone / f"{backbone}_best.pt", device)
        metrics = evaluate(model, val_loader, SegLoss(), device, amp=False)
        prof = profile_inference(model, (args.img_h, args.img_w), device, args.warmup, args.iters)
        row = {"backbone": backbone, **metrics, **prof}
        rows.append(row)
        print(f"{backbone}: mIoU={metrics['miou']:.4f}, IoU_DA={metrics['iou_da']:.4f}, F1={metrics['f1']:.4f}, FPS={prof['fps']:.2f}")

    fields = ["backbone", "loss", "pixel_acc", "iou_bg", "iou_da", "miou", "precision", "recall", "f1", "params", "trainable_params", "macs", "gflops", "latency_ms", "fps"]
    write_csv(args.save_csv, rows, fields)


if __name__ == "__main__":
    main()
