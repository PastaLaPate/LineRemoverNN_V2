# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- NEW: Residual Block ---
class ResidualBlock(nn.Module):
    """Standard Residual Block for better gradient flow."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.conv2(self.bn2(self.relu(self.conv1(self.bn1(x))))))
        out += identity
        out = self.relu2(out)
        return out


# Renaming the old Conv part for clarity when updating the main model
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        return x


# --- Update the main U-Net Blocks ---
# We wrap the structure to use Residual Blocks instead of simple Conv blocks
class DownBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x)


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # Concatenation of skip connection and upsampled feature map
        self.conv_mid = ConvBlock(in_c + out_c, in_c)
        # Upsampling (using ConvTranspose2d is common, but we'll use simple pooling reverse)
        self.up = nn.ConvTranspose2d(out_c, in_c, kernel_size=2, stride=2)
        self.conv_out = ConvBlock(in_c, out_c)

    def forward(self, x1, x2):
        # x1 is the output from the deeper layer (upsampled)
        # x2 is the skip connection (from the encoder)
        x1 = self.up(x1)
        # Pad and crop to ensure dimensions match the skip connection
        diff_h = x2.size(2) - x1.size(2)
        diff_w = x2.size(3) - x1.size(3)
        x1 = F.pad(
            x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2]
        )

        x = torch.cat([x2, x1], dim=1)
        x = self.conv_mid(x)
        return self.conv_out(x)


# Rebuilding the main model structure using Residual/ConvBlocks
class LineRemoverNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=64):
        super().__init__()

        # Encoder Path
        self.enc1 = DownBlock(in_channels, base_channels)
        self.enc2 = DownBlock(base_channels, base_channels * 2)
        self.enc3 = DownBlock(base_channels * 2, base_channels * 4)
        self.enc4 = DownBlock(base_channels * 4, base_channels * 8)

        # Bottleneck (Using Residual Block for depth)
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 8, base_channels * 8),
            ResidualBlock(base_channels * 8, base_channels * 8),
        )

        # Decoder Path
        self.up1 = UpBlock(base_channels * 8, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2)
        self.up3 = UpBlock(base_channels * 2, base_channels)
        self.up4 = UpBlock(base_channels, base_channels)

        # Output layer
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # Skip connection 1
        e2 = self.enc2(e1)  # Skip connection 2
        e3 = self.enc3(e2)  # Skip connection 3
        e4 = self.enc4(e3)  # Skip connection 4

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder
        d1 = self.up1(b, e3)
        d2 = self.up2(d1, e2)
        d3 = self.up3(d2, e1)
        d4 = self.up4(
            d3, x
        )  # Note: The input x here should ideally be the original input for the final skip connection matching

        # Final output
        output = self.outc(d4)
        return output
