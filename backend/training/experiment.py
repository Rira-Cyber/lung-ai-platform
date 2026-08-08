from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from backend.configs.config import Config


class ExperimentPaths:
    """
    Resolve and create isolated filesystem paths for one training run.
    """

    def __init__(
        self,
        root_dir: str | Path,
        experiment_name: str,
    ) -> None:
        if not experiment_name.strip():
            raise ValueError("experiment_name cannot be empty.")

        self.root_dir = Path(root_dir)
        self.experiment_name = experiment_name.strip()

        self.experiment_dir = self.root_dir / self.experiment_name

        self.checkpoint_dir = self.experiment_dir / "checkpoints"

        self.log_dir = self.experiment_dir / "logs"

        self.tensorboard_dir = self.log_dir / "tensorboard"

        self.config_path = self.experiment_dir / "config.json"

        self.manifest_path = self.experiment_dir / "patient_manifest.json"

        self.summary_path = self.experiment_dir / "summary.md"

    def create(self) -> None:
        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.tensorboard_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save_config(
        self,
        config: Config,
    ) -> None:
        self.create()

        with self.config_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                asdict(config),
                file,
                indent=4,
            )
