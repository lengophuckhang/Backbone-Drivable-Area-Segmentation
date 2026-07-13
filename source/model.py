from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchinfo import summary


class ConvBNAct(nn.Module):
    """Khối Conv-BatchNorm-ReLU dùng trong decoder."""

    def __init__(self, in_ch, out_ch, k=3):
        """
        Chức năng: tạo phép biến đổi tích chập cơ bản.
        Input: in_ch số kênh vào, out_ch số kênh ra, k kích thước kernel.
        Output: module biến đổi feature map [B,in_ch,H,W] -> [B,out_ch,H,W].
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Chức năng: lan truyền qua Conv-BN-ReLU.
        Input: x tensor [B,C,H,W].
        Output: tensor đặc trưng đã biến đổi.
        """
        return self.block(x)


class FPNDecoder(nn.Module):
    """Decoder FPN cố định để hợp nhất feature đa mức."""

    def __init__(self, in_channels, decoder_ch=128):
        """
        Chức năng: chiếu đặc trưng đa mức về cùng số kênh và hợp nhất top-down.
        Input: in_channels danh sách số kênh từ backbone, decoder_ch số kênh decoder.
        Output: module decoder trả về feature map mức cao đã hợp nhất.
        """
        super().__init__()
        self.proj = nn.ModuleList([ConvBNAct(c, decoder_ch, k=1) for c in in_channels])
        self.smooth = nn.ModuleList([ConvBNAct(decoder_ch, decoder_ch) for _ in in_channels])

    def forward(self, feats):
        """
        Chức năng: hợp nhất feature pyramid từ sâu lên nông.
        Input: feats là list tensor từ backbone theo thứ tự nông -> sâu.
        Output: tensor decoder tại độ phân giải của feature nông nhất.
        """
        x = self.proj[-1](feats[-1])
        for i in range(len(feats) - 2, -1, -1):
            x = F.interpolate(x, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
            x = self.smooth[i](x + self.proj[i](feats[i]))
        return x


class DASegmenter(nn.Module):
    """Mô hình single-task Drivable Area Segmentation với backbone thay được."""

    def __init__(self, backbone="resnet18", num_classes=2, decoder_ch=128, pretrained=False):
        """
        Chức năng: tạo backbone timm và segmentation head cố định.
        Input: backbone tên model timm, num_classes số lớp, decoder_ch số kênh decoder, pretrained dùng ImageNet hay không.
        Output: mô hình nhận ảnh [B,3,H,W] và trả logits [B,num_classes,H,W].
        """
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True)
        channels = self.backbone.feature_info.channels()
        assert len(channels) >= 3
        channels = channels[-4:]
        self.num_feats = len(channels)
        self.decoder = FPNDecoder(channels, decoder_ch)
        self.head = nn.Sequential(
            ConvBNAct(decoder_ch, decoder_ch),
            nn.Conv2d(decoder_ch, num_classes, kernel_size=1),
        )

    def forward(self, x):
        """
        Chức năng: trích xuất đặc trưng, giải mã và nội suy về kích thước ảnh vào.
        Input: x tensor ảnh [B,3,H,W].
        Output: logits phân đoạn [B,2,H,W].
        """
        size = x.shape[-2:]
        feats = self.backbone(x)[-self.num_feats:]
        out = self.head(self.decoder(feats))
        return F.interpolate(out, size=size, mode="bilinear", align_corners=False)


def build_model(backbone="resnet18", num_classes=2, decoder_ch=128, pretrained=False):
    """
    Chức năng: dựng mô hình dùng chung cho train/val/inference/benchmark.
    Input: tên backbone, số lớp, số kênh decoder, cờ pretrained.
    Output: đối tượng DASegmenter.
    """
    return DASegmenter(backbone, num_classes, decoder_ch, pretrained)


def parse_args():
    """
    Chức năng: khai báo tham số xem thông tin mô hình bằng torchinfo.
    Input: tham số CLI từ terminal.
    Output: Namespace chứa backbone, kích thước input và số kênh decoder.
    """
    p = ArgumentParser()
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--decoder_ch", type=int, default=128)
    p.add_argument("--img_h", type=int, default=384)
    p.add_argument("--img_w", type=int, default=640)
    p.add_argument("--pretrained", action="store_true")
    return p.parse_args()


def main():
    """
    Chức năng: in kiến trúc, số tham số và Mult-Adds bằng torchinfo.
    Input: tham số từ parse_args().
    Output: bảng summary in ra terminal.
    """
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.backbone, decoder_ch=args.decoder_ch, pretrained=args.pretrained).to(device)
    summary(
        model,
        input_size=(1, 3, args.img_h, args.img_w),
        device=device,
        col_names=("input_size", "output_size", "num_params", "mult_adds"),
        depth=4,
    )


if __name__ == "__main__":
    main()
