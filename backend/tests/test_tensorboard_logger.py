from __future__ import annotations

from pathlib import Path

import pytest

from backend.loggers.tensorboard_logger import (
    TensorBoardLogger,
)


def test_tensorboard_logger_creates_event_file(
    tmp_path: Path,
) -> None:
    logger = TensorBoardLogger(
        log_dir=tmp_path,
        flush_secs=1,
    )

    logger.log_epoch(
        epoch=1,
        epochs=5,
        metrics={
            "train_loss": 0.5,
            "train_dice": 0.6,
            "val_loss": 0.4,
            "val_dice": 0.7,
            "learning_rate": 0.001,
        },
    )

    logger.close()

    event_files = list(tmp_path.glob("events.out.tfevents.*"))

    assert event_files


def test_tensorboard_logger_ignores_unsupported_values(
    tmp_path: Path,
) -> None:
    logger = TensorBoardLogger(
        log_dir=tmp_path,
    )

    logger.log_epoch(
        epoch=1,
        epochs=1,
        metrics={
            "train_loss": 0.5,
            "message": "ignored",
            "flag": True,
            "missing": None,
        },
    )

    logger.close()

    assert logger.closed is True


def test_tensorboard_logger_close_is_idempotent(
    tmp_path: Path,
) -> None:
    logger = TensorBoardLogger(
        log_dir=tmp_path,
    )

    logger.close()
    logger.close()

    assert logger.closed is True


def test_logging_after_close_raises_error(
    tmp_path: Path,
) -> None:
    logger = TensorBoardLogger(
        log_dir=tmp_path,
    )

    logger.close()

    with pytest.raises(
        RuntimeError,
        match="after TensorBoardLogger is closed",
    ):
        logger.log_epoch(
            epoch=1,
            epochs=1,
            metrics={
                "train_loss": 0.5,
            },
        )


@pytest.mark.parametrize(
    ("epoch", "epochs", "message"),
    [
        (0, 10, "epoch must be greater"),
        (1, 0, "epochs must be greater"),
        (11, 10, "cannot be greater"),
    ],
)
def test_invalid_epoch_values_raise_error(
    tmp_path: Path,
    epoch: int,
    epochs: int,
    message: str,
) -> None:
    logger = TensorBoardLogger(
        log_dir=tmp_path,
    )

    try:
        with pytest.raises(
            ValueError,
            match=message,
        ):
            logger.log_epoch(
                epoch=epoch,
                epochs=epochs,
                metrics={},
            )
    finally:
        logger.close()


def test_invalid_flush_secs_raises_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="flush_secs",
    ):
        TensorBoardLogger(
            log_dir=tmp_path,
            flush_secs=0,
        )
