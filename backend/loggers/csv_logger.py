from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from backend.loggers.base_logger import BaseLogger


class CSVLogger(BaseLogger):
    def __init__(
        self,
        save_dir: str = "logs",
        filename: str = "training.csv",
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.file_path = self.save_dir / filename

        self.file = open(
            self.file_path,
            mode="w",
            newline="",
        )

        self.writer = None

    def log_epoch(
        self,
        epoch: int,
        epochs: int,
        metrics: dict[str, Any],
    ) -> None:
        row = {
            "epoch": epoch,
            **metrics,
        }

        if self.writer is None:
            self.writer = csv.DictWriter(
                self.file,
                fieldnames=row.keys(),
            )

            self.writer.writeheader()

        self.writer.writerow(row)

        self.file.flush()

    def close(self) -> None:
        if not self.file.closed:
            self.file.close()
