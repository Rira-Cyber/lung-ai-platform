from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


class CheckpointManager:
    """
    Persist and load training checkpoint payloads.

    This class is responsible only for checkpoint file I/O.
    It does not know about models, optimizers, schedulers, or
    other training components.
    """

    def __init__(
        self,
        save_dir: str | Path = "checkpoints",
    ) -> None:
        self.save_dir = Path(save_dir)

        self.save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.latest_checkpoint = self.save_dir / "latest.pt"

        self.best_checkpoint = self.save_dir / "best.pt"

    def save(
        self,
        checkpoint: Mapping[str, Any],
        *,
        is_best: bool = False,
    ) -> None:
        """
        Save the latest checkpoint and optionally the best checkpoint.
        """

        self._atomic_save(
            checkpoint=checkpoint,
            destination=self.latest_checkpoint,
        )

        if is_best:
            self._atomic_save(
                checkpoint=checkpoint,
                destination=self.best_checkpoint,
            )

    def load(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        best: bool = False,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """
        Load and return a checkpoint payload.
        """

        resolved_path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else (self.best_checkpoint if best else self.latest_checkpoint)
        )

        if not resolved_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resolved_path}")

        checkpoint = torch.load(
            resolved_path,
            map_location=map_location,
            weights_only=False,
        )

        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint payload must be a dictionary.")

        return checkpoint

    def exists(
        self,
        *,
        best: bool = False,
    ) -> bool:
        checkpoint_path = self.best_checkpoint if best else self.latest_checkpoint

        return checkpoint_path.exists()

    def latest_path(self) -> Path:
        return self.latest_checkpoint

    def best_path(self) -> Path:
        return self.best_checkpoint

    @staticmethod
    def _atomic_save(
        checkpoint: Mapping[str, Any],
        destination: Path,
    ) -> None:
        """
        Write to a temporary file before replacing the target.
        """

        temporary_path = destination.with_suffix(destination.suffix + ".tmp")

        torch.save(
            dict(checkpoint),
            temporary_path,
        )

        temporary_path.replace(destination)
