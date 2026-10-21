"""PyTorch architectures for the HipMRI segmentation comparison.

The baseline is a conventional slice-wise 2D U-Net.  The proposed model is a
3D residual U-Net with deep-supervision heads, matching the hard-difficulty
architecture described in COMP3710 Section 2.2.  This module intentionally
depends only on PyTorch.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    """Return a GroupNorm group count that divides ``channels``."""
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class DoubleConv2D(nn.Module):
    """Two 3x3 convolutions used by the standard 2D U-Net baseline."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = _group_count(out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet2D(nn.Module):
    """Standard 2D U-Net baseline for independent axial slices.

    Reference: Ronneberger, Fischer, and Brox, MICCAI 2015.
    """

    def __init__(
        self, in_channels: int = 1, num_classes: int = 6, base_channels: int = 32
    ) -> None:
        super().__init__()
        widths = [base_channels * (2**i) for i in range(5)]
        self.encoders = nn.ModuleList(
            [
                DoubleConv2D(in_channels, widths[0]),
                DoubleConv2D(widths[0], widths[1]),
                DoubleConv2D(widths[1], widths[2]),
                DoubleConv2D(widths[2], widths[3]),
            ]
        )
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv2D(widths[3], widths[4])
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(widths[4], widths[3], 2, stride=2),
                nn.ConvTranspose2d(widths[3], widths[2], 2, stride=2),
                nn.ConvTranspose2d(widths[2], widths[1], 2, stride=2),
                nn.ConvTranspose2d(widths[1], widths[0], 2, stride=2),
            ]
        )
        self.decoders = nn.ModuleList(
            [
                DoubleConv2D(widths[3] * 2, widths[3]),
                DoubleConv2D(widths[2] * 2, widths[2]),
                DoubleConv2D(widths[1] * 2, widths[1]),
                DoubleConv2D(widths[0] * 2, widths[0]),
            ]
        )
        self.head = nn.Conv2d(widths[0], num_classes, 1)

    # forward: encoder-decoder with skip connections, using bilinear upsampling to match skip shapes
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat((skip, x), dim=1))

        return self.head(x)
    