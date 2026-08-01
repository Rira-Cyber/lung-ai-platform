from __future__ import annotations

import torch
import torch.nn as nn

from backend.models.blocks import (
    DoubleConv,
    DownBlock,
    UpBlock,
)


class UNet3D(nn.Module):
    """
    3D U-Net for binary segmentation.

    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: tuple[int, ...] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()

        # -------------------------
        # Encoder
        # -------------------------

        self.encoder1 = DownBlock(
            in_channels,
            features[0],
        )

        self.encoder2 = DownBlock(
            features[0],
            features[1],
        )

        self.encoder3 = DownBlock(
            features[1],
            features[2],
        )

        self.encoder4 = DownBlock(
            features[2],
            features[3],
        )

        # -------------------------
        # Bottleneck
        # -------------------------

        self.bottleneck = DoubleConv(
            features[3],
            features[3] * 2,
        )

        # -------------------------
        # Decoder
        # -------------------------
        self.decoder4 = UpBlock(
            in_channels=features[3] * 2,
            skip_channels=features[3],
            out_channels=features[3],
        )

        self.decoder3 = UpBlock(
            in_channels=features[3],
            skip_channels=features[2],
            out_channels=features[2],
        )

        self.decoder2 = UpBlock(
            in_channels=features[2],
            skip_channels=features[1],
            out_channels=features[1],
        )

        self.decoder1 = UpBlock(
            in_channels=features[1],
            skip_channels=features[0],
            out_channels=features[0],
        )

        # -------------------------
        # Output
        # -------------------------

        self.head = nn.Conv3d(
            in_channels=features[0],
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x, skip1 = self.encoder1(x)

        x, skip2 = self.encoder2(x)

        x, skip3 = self.encoder3(x)

        x, skip4 = self.encoder4(x)

        x = self.bottleneck(x)

        x = self.decoder4(x, skip4)

        x = self.decoder3(x, skip3)

        x = self.decoder2(x, skip2)

        x = self.decoder1(x, skip1)

        x = self.head(x)

        return x
