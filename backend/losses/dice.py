from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation
    """

    def __init__(
        self,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()

        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        prediction = torch.sigmoid(logits)

        prediction = prediction.reshape(-1)

        target = target.reshape(-1)

        intersection = (prediction * target).sum()

        dice = (2.0 * intersection + self.smooth) / (
            prediction.sum() + target.sum() + self.smooth
        )

        loss = 1.0 - dice

        return loss
