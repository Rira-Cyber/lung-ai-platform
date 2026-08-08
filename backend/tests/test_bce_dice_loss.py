from __future__ import annotations

import pytest
import torch

from backend.losses.bce_dice import (
    BCEDiceLoss,
)


def test_loss_returns_scalar() -> None:
    criterion = BCEDiceLoss()

    logits = torch.zeros(
        (2, 1, 8, 8, 8),
        dtype=torch.float32,
    )

    target = torch.zeros_like(logits)

    loss = criterion(
        logits,
        target,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_loss_supports_backward() -> None:
    criterion = BCEDiceLoss()

    logits = torch.zeros(
        (1, 1, 4, 4, 4),
        dtype=torch.float32,
        requires_grad=True,
    )

    target = torch.zeros_like(logits)

    loss = criterion(
        logits,
        target,
    )

    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_negative_patch_penalizes_false_positive_prediction() -> None:
    criterion = BCEDiceLoss()

    target = torch.zeros(
        (1, 1, 4, 4, 4),
        dtype=torch.float32,
    )

    good_logits = torch.full(
        target.shape,
        -10.0,
    )

    bad_logits = torch.full(
        target.shape,
        10.0,
    )

    good_loss = criterion(
        good_logits,
        target,
    )

    bad_loss = criterion(
        bad_logits,
        target,
    )

    assert good_loss < bad_loss


def test_positive_patch_penalizes_false_negative_prediction() -> None:
    criterion = BCEDiceLoss()

    target = torch.ones(
        (1, 1, 4, 4, 4),
        dtype=torch.float32,
    )

    good_logits = torch.full(
        target.shape,
        10.0,
    )

    bad_logits = torch.full(
        target.shape,
        -10.0,
    )

    good_loss = criterion(
        good_logits,
        target,
    )

    bad_loss = criterion(
        bad_logits,
        target,
    )

    assert good_loss < bad_loss


def test_combined_loss_is_finite_for_mixed_target() -> None:
    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
    )

    logits = torch.randn(
        (2, 1, 4, 4, 4),
        dtype=torch.float32,
    )

    target = torch.zeros_like(logits)

    target[
        0,
        :,
        1:3,
        1:3,
        1:3,
    ] = 1.0

    loss = criterion(
        logits,
        target,
    )

    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    ("bce_weight", "dice_weight"),
    [
        (-0.1, 1.0),
        (1.0, -0.1),
        (0.0, 0.0),
    ],
)
def test_invalid_weights_raise_error(
    bce_weight: float,
    dice_weight: float,
) -> None:
    with pytest.raises(ValueError):
        BCEDiceLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
        )


def test_shape_mismatch_raises_error() -> None:
    criterion = BCEDiceLoss()

    logits = torch.zeros((1, 1, 4, 4, 4))

    target = torch.zeros((1, 1, 2, 2, 2))

    with pytest.raises(
        ValueError,
        match="identical shapes",
    ):
        criterion(
            logits,
            target,
        )
