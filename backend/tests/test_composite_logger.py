from __future__ import annotations

from typing import Any

import pytest

from backend.loggers.base_logger import BaseLogger
from backend.loggers.composite_logger import (
    CompositeLogger,
)


class RecordingLogger(BaseLogger):
    def __init__(self) -> None:
        self.records: list[dict] = []
        self.closed = False

    def log_epoch(
        self,
        epoch: int,
        epochs: int,
        metrics: dict[str, Any],
    ) -> None:
        self.records.append(
            {
                "epoch": epoch,
                "epochs": epochs,
                "metrics": dict(metrics),
            }
        )

    def close(self) -> None:
        self.closed = True


class FailingCloseLogger(BaseLogger):
    def log_epoch(
        self,
        epoch: int,
        epochs: int,
        metrics: dict[str, Any],
    ) -> None:
        pass

    def close(self) -> None:
        raise RuntimeError("Close failed.")


def test_composite_logger_distributes_metrics() -> None:
    first = RecordingLogger()
    second = RecordingLogger()

    logger = CompositeLogger(
        [
            first,
            second,
        ]
    )

    metrics = {
        "train_loss": 0.5,
        "val_dice": 0.8,
    }

    logger.log_epoch(
        epoch=1,
        epochs=10,
        metrics=metrics,
    )

    assert first.records == second.records

    assert first.records[0] == {
        "epoch": 1,
        "epochs": 10,
        "metrics": metrics,
    }


def test_composite_logger_closes_all_loggers() -> None:
    first = RecordingLogger()
    second = RecordingLogger()

    logger = CompositeLogger(
        [
            first,
            second,
        ]
    )

    logger.close()

    assert first.closed is True
    assert second.closed is True


def test_empty_logger_collection_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="at least one logger",
    ):
        CompositeLogger([])


def test_invalid_logger_type_raises_error() -> None:
    with pytest.raises(
        TypeError,
        match="inherit from BaseLogger",
    ):
        CompositeLogger(
            [
                RecordingLogger(),
                object(),
            ]
        )


def test_close_attempts_all_loggers_before_raising() -> None:
    successful = RecordingLogger()

    logger = CompositeLogger(
        [
            FailingCloseLogger(),
            successful,
        ]
    )

    with pytest.raises(
        RuntimeError,
        match="Failed to close",
    ):
        logger.close()

    assert successful.closed is True
