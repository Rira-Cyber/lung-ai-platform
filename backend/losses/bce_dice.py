from __future__ import annotations

import torch
import torch.nn as nn

from backend.losses.dice import DiceLoss


class BCEDiceLoss(nn.Module):
    """
    Combined BCE-with-logits and Dice loss for binary segmentation.

    BCE provides voxel-level supervision, including on fully
    negative patches, while Dice focuses on segmentation overlap.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        if bce_weight < 0:
            raise ValueError("bce_weight cannot be negative.")

        if dice_weight < 0:
            raise ValueError("dice_weight cannot be negative.")

        if bce_weight == 0 and dice_weight == 0:
            raise ValueError("At least one loss weight must be greater than zero.")

        self.bce_weight = float(bce_weight)

        self.dice_weight = float(dice_weight)

        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.dice = DiceLoss(smooth=smooth)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if logits.shape != target.shape:
            raise ValueError("logits and target must have identical shapes.")

        target = target.float()

        bce_loss = self.bce(
            logits,
            target,
        )

        dice_loss = self.dice(
            logits,
            target,
        )

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
