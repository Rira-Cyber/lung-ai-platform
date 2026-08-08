from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Config:
    """
    Application configuration for dataset, model, training,
    checkpointing, and logging.
    """

    # Dataset
    dataset_path: str
    patch_size: tuple[int, int, int]
    positive_ratio: float
    patches_per_patient: int

    # Experiment
    experiment_root: str
    experiment_name: str

    # Dataset split
    val_ratio: float
    test_ratio: float
    split_seed: int
    patient_manifest_path: str | None

    # DataLoader
    batch_size: int
    num_workers: int
    pin_memory: bool
    shuffle: bool

    # Model
    in_channels: int
    out_channels: int
    features: tuple[int, int, int, int]

    # Training
    learning_rate: float
    epochs: int
    device: str
    max_grad_norm: float | None

    # Scheduler
    scheduler_enabled: bool
    scheduler_type: Literal[
        "reduce_on_plateau",
        "step",
    ]
    scheduler_monitor: str
    scheduler_factor: float
    scheduler_patience: int
    scheduler_min_lr: float
    scheduler_step_size: int
    scheduler_gamma: float

    # Early stopping
    early_stopping_enabled: bool
    early_stopping_monitor: str
    early_stopping_mode: Literal[
        "min",
        "max",
    ]
    early_stopping_patience: int
    early_stopping_min_delta: float

    # Best model selection
    best_metric: str
    best_mode: Literal[
        "min",
        "max",
    ]

    # Checkpoint
    checkpoint_dir: str

    # Logging
    log_dir: str
    csv_logging_enabled: bool
    tensorboard_enabled: bool
    tensorboard_log_dir: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.val_ratio < 1.0:
            raise ValueError("val_ratio must be in the range [0.0, 1.0).")

        if not 0.0 <= self.test_ratio < 1.0:
            raise ValueError("test_ratio must be in the range [0.0, 1.0).")

        if self.val_ratio + self.test_ratio >= 1.0:
            raise ValueError("val_ratio + test_ratio must be less than 1.0.")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")

        if self.epochs <= 0:
            raise ValueError("epochs must be greater than zero.")

        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be greater than zero or None.")

        if self.scheduler_factor <= 0:
            raise ValueError("scheduler_factor must be greater than zero.")

        if self.scheduler_patience < 0:
            raise ValueError("scheduler_patience cannot be negative.")

        if self.scheduler_min_lr < 0:
            raise ValueError("scheduler_min_lr cannot be negative.")

        if self.scheduler_step_size <= 0:
            raise ValueError("scheduler_step_size must be greater than zero.")

        if self.scheduler_gamma <= 0:
            raise ValueError("scheduler_gamma must be greater than zero.")

        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be greater than zero.")

        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta cannot be negative.")
