from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Building blocks ---


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """Attention gate on skip connections — suppresses line features, preserves text."""

    def __init__(self, f_g: int, f_l: int, f_int: int) -> None:
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(f_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(f_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = F.relu(g1 + x1, inplace=True)
        psi = self.psi(psi)
        return x * psi


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.attn = AttentionGate(f_g=out_ch, f_l=skip_ch, f_int=out_ch // 2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if spatial dims don't match exactly
        if x.shape != skip.shape:
            x = F.pad(x, [0, skip.shape[3] - x.shape[3], 0, skip.shape[2] - x.shape[2]])
        skip = self.attn(g=x, x=skip)
        return self.conv(torch.cat([x, skip], dim=1))


# --- U-Net with Attention ---


class LineRemoverNN(nn.Module):
    """
    Attention U-Net for line removal.
    Channels: [32, 64, 128, 256] encoder + 512 bottleneck (~7M params).
    Scale up to [64, 128, 256, 512] for better quality if VRAM allows.
    """

    def __init__(self, channels: list[int] = [32, 64, 128, 256]) -> None:
        super().__init__()
        ch = channels

        # Encoder
        self.enc1 = ConvBlock(1, ch[0])
        self.enc2 = ConvBlock(ch[0], ch[1])
        self.enc3 = ConvBlock(ch[1], ch[2])
        self.enc4 = ConvBlock(ch[2], ch[3])
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(ch[3], ch[3] * 2)

        # Decoder
        self.dec4 = UpBlock(ch[3] * 2, ch[3], ch[3])
        self.dec3 = UpBlock(ch[3], ch[2], ch[2])
        self.dec2 = UpBlock(ch[2], ch[1], ch[1])
        self.dec1 = UpBlock(ch[1], ch[0], ch[0])

        self.out = nn.Conv2d(ch[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        logits = self.out(d1)
        return (x - logits, logits)
