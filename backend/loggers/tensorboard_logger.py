from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter

from backend.loggers.base_logger import BaseLogger


class TensorBoardLogger(BaseLogger):
    """
    Log scalar epoch metrics to TensorBoard.

    Each metric is written using its original name, for example:

    - train_loss
    - train_dice
    - val_loss
    - val_dice
    - learning_rate
    """

    def __init__(
        self,
        log_dir: str | Path = "logs/tensorboard",
        flush_secs: int = 30,
    ) -> None:
        if flush_secs <= 0:
            raise ValueError("flush_secs must be greater than zero.")

        self.log_dir = Path(log_dir)

        self.log_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.writer = SummaryWriter(
            log_dir=str(self.log_dir),
            flush_secs=flush_secs,
        )

        self.closed = False

    def log_epoch(
        self,
        epoch: int,
        epochs: int,
        metrics: dict[str, Any],
    ) -> None:
        if self.closed:
            raise RuntimeError("Cannot log metrics after TensorBoardLogger is closed.")

        if epoch <= 0:
            raise ValueError("epoch must be greater than zero.")

        if epochs <= 0:
            raise ValueError("epochs must be greater than zero.")

        if epoch > epochs:
            raise ValueError("epoch cannot be greater than total epochs.")

        for name, value in metrics.items():
            if value is None:
                continue

            if isinstance(
                value,
                bool,
            ):
                continue

            if isinstance(
                value,
                int | float,
            ):
                self.writer.add_scalar(
                    tag=name,
                    scalar_value=float(value),
                    global_step=epoch,
                )

        self.writer.flush()

    def close(self) -> None:
        if self.closed:
            return

        self.writer.flush()
        self.writer.close()

        self.closed = True
