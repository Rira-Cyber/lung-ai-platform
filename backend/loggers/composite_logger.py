from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from backend.loggers.base_logger import BaseLogger


class CompositeLogger(BaseLogger):
    """
    Distribute logging calls to multiple logger implementations.

    CompositeLogger contains no formatting or persistence logic.
    Each child logger remains responsible for its own output.
    """

    def __init__(
        self,
        loggers: Iterable[BaseLogger],
    ) -> None:
        self.loggers = list(loggers)

        if not self.loggers:
            raise ValueError("CompositeLogger requires at least one logger.")

        if any(not isinstance(logger, BaseLogger) for logger in self.loggers):
            raise TypeError("All loggers must inherit from BaseLogger.")

    def log_epoch(
        self,
        epoch: int,
        epochs: int,
        metrics: dict[str, Any],
    ) -> None:
        for logger in self.loggers:
            logger.log_epoch(
                epoch=epoch,
                epochs=epochs,
                metrics=metrics,
            )

    def close(self) -> None:
        errors: list[Exception] = []

        for logger in self.loggers:
            try:
                logger.close()
            except Exception as error:
                errors.append(error)

        if errors:
            raise RuntimeError(f"Failed to close {len(errors)} logger(s).") from errors[
                0
            ]
