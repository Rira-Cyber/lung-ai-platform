from dataclasses import dataclass


@dataclass
class Config:

    # Dataset

    dataset_path: str

    patch_size: tuple[int, int, int]

    positive_ratio: float

    patches_per_patient: int

    # Dataloader

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