from __future__ import annotations

from pathlib import Path

import pytest
import torch

from backend.training.checkpoint import CheckpointManager


def create_checkpoint_payload() -> dict:
    return {
        "epoch": 5,
        "metrics": {
            "train_loss": 0.1234,
            "val_loss": 0.1111,
        },
        "model_state_dict": {
            "weight": torch.tensor(
                [1.0, 2.0],
            ),
        },
        "optimizer_state_dict": {
            "state": {},
            "param_groups": [],
        },
        "scheduler_state_dict": {
            "last_epoch": 5,
        },
        "early_stopping_state": {
            "best_score": 0.1111,
            "bad_epochs": 1,
            "should_stop": False,
        },
        "best_metric": "val_dice",
        "best_mode": "max",
        "best_score": 0.85,
        "history": {
            "train_loss": [0.2, 0.15, 0.1234],
            "val_loss": [0.18, 0.13, 0.1111],
        },
    }


def test_save_and_load_latest_checkpoint(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(
        save_dir=tmp_path,
    )

    payload = create_checkpoint_payload()

    manager.save(
        checkpoint=payload,
    )

    assert manager.exists()
    assert manager.latest_path().exists()

    loaded = manager.load()

    assert loaded["epoch"] == 5
    assert loaded["best_score"] == 0.85
    assert loaded["metrics"]["train_loss"] == pytest.approx(0.1234)

    assert torch.equal(
        loaded["model_state_dict"]["weight"],
        payload["model_state_dict"]["weight"],
    )


def test_save_best_checkpoint(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(
        save_dir=tmp_path,
    )

    payload = create_checkpoint_payload()

    manager.save(
        checkpoint=payload,
        is_best=True,
    )

    assert manager.exists()
    assert manager.exists(best=True)
    assert manager.best_path().exists()

    loaded_best = manager.load(
        best=True,
    )

    assert loaded_best["epoch"] == 5
    assert loaded_best["best_score"] == 0.85


def test_latest_checkpoint_is_always_updated(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(
        save_dir=tmp_path,
    )

    first_payload = create_checkpoint_payload()

    manager.save(
        checkpoint=first_payload,
        is_best=True,
    )

    second_payload = {
        **first_payload,
        "epoch": 6,
        "best_score": 0.80,
    }

    manager.save(
        checkpoint=second_payload,
        is_best=False,
    )

    latest = manager.load()
    best = manager.load(best=True)

    assert latest["epoch"] == 6
    assert best["epoch"] == 5


def test_load_explicit_checkpoint_path(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(
        save_dir=tmp_path,
    )

    payload = create_checkpoint_payload()

    custom_path = tmp_path / "custom_checkpoint.pt"

    torch.save(
        payload,
        custom_path,
    )

    loaded = manager.load(
        checkpoint_path=custom_path,
    )

    assert loaded["epoch"] == 5


def test_load_missing_checkpoint_raises_error(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(
        save_dir=tmp_path,
    )

    with pytest.raises(
        FileNotFoundError,
        match="Checkpoint not found",
    ):
        manager.load()


def test_invalid_checkpoint_payload_raises_error(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(
        save_dir=tmp_path,
    )

    invalid_path = tmp_path / "invalid.pt"

    torch.save(
        ["not", "a", "dictionary"],
        invalid_path,
    )

    with pytest.raises(
        ValueError,
        match="must be a dictionary",
    ):
        manager.load(
            checkpoint_path=invalid_path,
        )
