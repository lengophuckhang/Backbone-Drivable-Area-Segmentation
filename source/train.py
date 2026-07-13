from argparse import ArgumentParser
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from dataset import BDD100KDA
from model import build_model
from utils import SegLoss, set_seed, poly_lr, train_one_epoch, evaluate, save_checkpoint, write_csv


def parse_args():
    """
    Chức năng: khai báo tham số dòng lệnh, không dùng file cấu hình.
    Input: tham số CLI từ terminal.
    Output: Namespace chứa hyperparameter và đường dẫn.
    """
    p = ArgumentParser()
    p.add_argument("--data_root", default="BDD100K")
    p.add_argument("--out_dir", default="runs/resnet18")
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--decoder_ch", type=int, default=128)
    p.add_argument("--img_h", type=int, default=384)
    p.add_argument("--img_w", type=int, default=640)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--dice_weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def main():
    """
    Chức năng: train single-task DA, lưu last/best checkpoint và log.csv.
    Input: tham số từ parse_args().
    Output: checkpoint và log huấn luyện trong out_dir.
    """
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = BDD100KDA(args.data_root, "train", (args.img_h, args.img_w), augment=True)
    val_set = BDD100KDA(args.data_root, "val", (args.img_h, args.img_w), augment=False)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.backbone, decoder_ch=args.decoder_ch, pretrained=args.pretrained).to(device)
    criterion = SegLoss(args.dice_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    amp = args.amp and device.type == "cuda"

    start_epoch = 1
    best_miou = 0.0
    rows = []
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt["best_miou"]
        print(f"Resume từ epoch {ckpt['epoch']}, best_miou={best_miou:.4f}")
        log_path = out_dir / "log.csv"
        if log_path.exists():
            import csv
            with open(log_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
    
    for epoch in range(start_epoch, args.epochs + 1):
        lr = poly_lr(optimizer, args.lr, epoch - 1, args.epochs)
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, amp)
        metrics = evaluate(model, val_loader, criterion, device, amp)
        best_miou = max(best_miou, metrics["miou"])

        save_checkpoint(out_dir / f"{args.backbone}_last.pt", model, optimizer, epoch, best_miou)
        if metrics["miou"] == best_miou:
            save_checkpoint(out_dir / f"{args.backbone}_best.pt", model, optimizer, epoch, best_miou)

        row = {"epoch": epoch, "lr": lr, "train_loss": train_loss, **metrics}
        rows.append(row)
        write_csv(out_dir / "log.csv", rows, row.keys())
        print(f"epoch={epoch:03d} lr={lr:.6f} train_loss={train_loss:.4f} val_mIoU={metrics['miou']:.4f} IoU_DA={metrics['iou_da']:.4f} F1={metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
