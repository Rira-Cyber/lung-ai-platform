from __future__ import annotations

import torch
import torch.nn as nn


class DiceMetric(nn.Module):
    """
    Dice score for binary segmentation
    """

    def __init__(
        self,
        threshold: float = 0.5,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()

        self.threshold = threshold
        self.smooth = smooth

    @torch.no_grad()
    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        prediction = torch.sigmoid(logits)

        prediction = (prediction > self.threshold).float()

        prediction = prediction.reshape(-1)
        target = target.reshape(-1)

        intersection = (prediction * target).sum()

        dice = (2.0 * intersection + self.smooth) / (
            prediction.sum() + target.sum() + self.smooth
        )

        return dice
