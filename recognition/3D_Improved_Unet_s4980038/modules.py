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



class ResidualBlock3D(nn.Module):
    """3D residual convolutional block.

    Inspired by the residual context modules from
    Isensee et al. (2018).

    Uses two 3D convolutions, GroupNorm, LeakyReLU,
    optional dropout, and a residual shortcut.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        groups = _group_count(out_channels)

        self.body = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
        )

        self.projection = (
            nn.Conv3d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    # residual forward: apply body, add projection of input, apply activation
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(x) + self.projection(x))


class ImprovedUNet3D(nn.Module):
    """Adapted 3D Improved U-Net for HipMRI segmentation.

    Based on the volumetric U-Net architecture of
    Cicek et al. (2016) and inspired by the improved
    architecture of Isensee et al. (2018).

    Incorporates 3D residual blocks, encoder-decoder
    skip connections, and deep supervision.

    Unlike the original architecture, this implementation
    uses GroupNorm, transposed convolutions, and two
    auxiliary supervision heads with weighted losses.

    During training, returns main and auxiliary logits.
    During evaluation, returns only the main logits.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        base_channels: int = 16,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        widths = [base_channels * (2**i) for i in range(5)]
        self.encoders = nn.ModuleList(
            [
                ResidualBlock3D(in_channels, widths[0]),
                ResidualBlock3D(widths[0], widths[1]),
                ResidualBlock3D(widths[1], widths[2], dropout / 2),
                ResidualBlock3D(widths[2], widths[3], dropout),
            ]
        )
        self.downsamples = nn.ModuleList(
            [nn.Conv3d(widths[i], widths[i], 2, stride=2) for i in range(4)]
        )
        self.bottleneck = ResidualBlock3D(widths[3], widths[4], dropout)
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose3d(widths[4], widths[3], 2, stride=2),
                nn.ConvTranspose3d(widths[3], widths[2], 2, stride=2),
                nn.ConvTranspose3d(widths[2], widths[1], 2, stride=2),
                nn.ConvTranspose3d(widths[1], widths[0], 2, stride=2),
            ]
        )
        self.decoders = nn.ModuleList(
            [
                ResidualBlock3D(widths[3] * 2, widths[3], dropout),
                ResidualBlock3D(widths[2] * 2, widths[2], dropout / 2),
                ResidualBlock3D(widths[1] * 2, widths[1]),
                ResidualBlock3D(widths[0] * 2, widths[0]),
            ]
        )
        self.head = nn.Conv3d(widths[0], num_classes, 1)
        self.aux_heads = nn.ModuleList(
            [nn.Conv3d(widths[2], num_classes, 1), nn.Conv3d(widths[1], num_classes, 1)]
        )

    # 3d encoder-decoder with skip connections, using trilinear upsampling to match skip shapes
    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | dict[str, torch.Tensor | list[torch.Tensor]]:
        
        input_size = x.shape[-3:]
        skips: list[torch.Tensor] = []

        # Encoder path: apply residual blocks and downsample, storing skip connections
        for encoder, downsample in zip(self.encoders, self.downsamples):
            x = encoder(x)
            skips.append(x)
            x = downsample(x)
        x = self.bottleneck(x)

        decoder_features: list[torch.Tensor] = []

        # Decoder path: apply upconvs and decoders, concatenating skip connections
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.shape[-3:] != skip.shape[-3:]:
                x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            x = decoder(torch.cat((skip, x), dim=1))
            decoder_features.append(x)

        main_logits = self.head(x)
        if not self.training:
            return main_logits

        # Auxiliary logits: apply auxiliary heads to intermediate decoder features and upsample to input size
        auxiliary = [
            F.interpolate(head(feature), size=input_size, mode="trilinear", align_corners=False)
            for head, feature in zip(self.aux_heads, decoder_features[1:3])
        ]

        return {"out": main_logits, "aux": auxiliary}


class DiceCrossEntropyLoss(nn.Module):
    """Combined cross-entropy and foreground soft Dice loss.

    The multi-class soft Dice component is inspired by
    Isensee et al. (2018), while this implementation combines
    it with cross-entropy using configurable weighting.
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        dice_weight: float = 0.6,
        smooth: float = 1e-5,
    ) -> None:
        
        super().__init__()
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the combined loss for multi-class segmentation."""

        ce = F.cross_entropy(logits, target.long(), weight=self.class_weights)
        probabilities = logits.softmax(dim=1)
        one_hot = F.one_hot(target.long(), logits.shape[1]).movedim(-1, 1).float()

        dims = tuple(range(2, logits.ndim))
        intersection = (probabilities * one_hot).sum(dim=dims)
        denominator = probabilities.sum(dim=dims) + one_hot.sum(dim=dims)

        # Compute the soft Dice coefficient for each class, excluding the background (class 0)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        foreground_dice_loss = 1.0 - dice[:, 1:].mean()

        return (1.0 - self.dice_weight) * ce + self.dice_weight * foreground_dice_loss


def primary_logits(output: torch.Tensor | dict[str, object]) -> torch.Tensor:
    """Return main logits from either architecture's forward result."""

    if isinstance(output, dict):
        logits = output["out"]
        if not isinstance(logits, torch.Tensor):
            raise TypeError("Model output 'out' must be a tensor")
        return logits
    return output


def supervised_loss(
    output: torch.Tensor | dict[str, object],
    target: torch.Tensor,
    criterion: nn.Module,
    auxiliary_weights: tuple[float, ...] = (0.4, 0.2),
) -> torch.Tensor:
    """Apply the main loss and weighted deep-supervision losses when present."""

    loss = criterion(primary_logits(output), target)

    if isinstance(output, dict):
        auxiliary = output.get("aux", [])
        if not isinstance(auxiliary, list):
            raise TypeError("Model output 'aux' must be a list")
        for weight, logits in zip(auxiliary_weights, auxiliary):
            loss = loss + weight * criterion(logits, target)

    return loss


def build_model(
    name: str,
    num_classes: int = 6,
    base_channels: int | None = None,
) -> nn.Module:
    """Construct a supported model by command-line name."""

    if name == "unet2d":
        return UNet2D(num_classes=num_classes, base_channels=base_channels or 32)
    if name == "improved_unet3d":
        return ImprovedUNet3D(num_classes=num_classes, base_channels=base_channels or 16)
    
    raise ValueError(f"Unknown model: {name}")