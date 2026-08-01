from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Application configuration for dataset, model, and training settings."""

    # Dataset
    dataset_path: str
    patch_size: tuple[int, int, int]
    positive_ratio: float
    patches_per_patient: int

    # Dataset split
    val_ratio: float
    split_seed: int

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
