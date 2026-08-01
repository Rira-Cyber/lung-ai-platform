from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from backend.datasets.dataset_splitter import (
    PatientSplit,
    PatientSplitter,
)
from backend.preprocessing.patch_sampler import PatchSampler


DatasetFactory = Callable[..., Any]
MetadataGeneratorFactory = Callable[..., Any]


@dataclass(frozen=True)
class TrainValidationLoaders:
    """
    Container for training and validation DataLoaders.

    The patient split and validation metadata path are exposed
    for reproducibility, debugging, and experiment tracking.
    """

    train_loader: DataLoader
    val_loader: DataLoader
    patient_split: PatientSplit
    validation_metadata_path: Path


def create_train_validation_loaders(
    dataset_path: str | Path,
    batch_size: int,
    num_workers: int,
    patch_size: tuple[int, int, int],
    positive_ratio: float,
    patches_per_patient: int,
    val_ratio: float,
    split_seed: int,
    pin_memory: bool = False,
    shuffle: bool = True,
    validation_metadata_path: str | Path | None = None,
    overwrite_validation_metadata: bool = False,
    *,
    training_dataset_factory: DatasetFactory | None = None,
    validation_dataset_factory: DatasetFactory | None = None,
    metadata_generator_factory: MetadataGeneratorFactory | None = None,
) -> TrainValidationLoaders:
    """
    Create leakage-safe training and deterministic validation loaders.

    Training patches are sampled dynamically.
    Validation patches are loaded from fixed metadata.

    Dataset and metadata implementations can be injected for testing.
    Production implementations are resolved lazily by default.
    """

    _validate_loader_arguments(
        batch_size=batch_size,
        num_workers=num_workers,
        patches_per_patient=patches_per_patient,
    )

    resolved_dataset_path = Path(dataset_path)

    resolved_metadata_path = (
        Path(validation_metadata_path)
        if validation_metadata_path is not None
        else (
            resolved_dataset_path.parent.parent
            / "processed"
            / "validation_metadata.json"
        )
    )

    resolved_training_dataset_factory = (
        training_dataset_factory or _get_training_dataset_factory()
    )

    resolved_validation_dataset_factory = (
        validation_dataset_factory or _get_validation_dataset_factory()
    )

    resolved_metadata_generator_factory = (
        metadata_generator_factory or _get_metadata_generator_factory()
    )

    discovery_sampler = PatchSampler(
        patch_size=patch_size,
        positive_ratio=positive_ratio,
        random_seed=split_seed,
    )

    discovery_dataset = resolved_training_dataset_factory(
        dataset_path=resolved_dataset_path,
        sampler=discovery_sampler,
        patches_per_patient=1,
    )

    patient_split = PatientSplitter(
        val_ratio=val_ratio,
        seed=split_seed,
    ).split(discovery_dataset.patient_ids)

    train_sampler = PatchSampler(
        patch_size=patch_size,
        positive_ratio=positive_ratio,
    )

    train_dataset = resolved_training_dataset_factory(
        dataset_path=resolved_dataset_path,
        sampler=train_sampler,
        patches_per_patient=(patches_per_patient),
        patient_ids=(patient_split.train_ids),
    )

    metadata_generator = resolved_metadata_generator_factory(
        dataset_path=resolved_dataset_path,
        patch_size=patch_size,
        positive_ratio=positive_ratio,
        patches_per_patient=(patches_per_patient),
        seed=split_seed,
    )

    validation_metadata = metadata_generator.generate_or_load(
        patient_ids=(patient_split.val_ids),
        metadata_path=(resolved_metadata_path),
        overwrite=(overwrite_validation_metadata),
    )

    validation_dataset = resolved_validation_dataset_factory(
        dataset_path=resolved_dataset_path,
        metadata=validation_metadata,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(split_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=(loader_generator if shuffle else None),
    )

    val_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return TrainValidationLoaders(
        train_loader=train_loader,
        val_loader=val_loader,
        patient_split=patient_split,
        validation_metadata_path=(resolved_metadata_path),
    )


def _get_training_dataset_factory() -> DatasetFactory:
    """
    Resolve the production training Dataset lazily.
    """

    from backend.datasets.lidc_dataset import (
        LIDCDataset,
    )

    return LIDCDataset


def _get_validation_dataset_factory() -> DatasetFactory:
    """
    Resolve the production validation Dataset lazily.
    """

    from backend.datasets.validation_dataset import (
        ValidationDataset,
    )

    return ValidationDataset


def _get_metadata_generator_factory() -> MetadataGeneratorFactory:
    """
    Resolve the production metadata generator lazily.
    """

    from backend.datasets.validation_metadata import (
        ValidationMetadataGenerator,
    )

    return ValidationMetadataGenerator


def _validate_loader_arguments(
    batch_size: int,
    num_workers: int,
    patches_per_patient: int,
) -> None:
    """
    Validate DataLoader factory arguments.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    if patches_per_patient <= 0:
        raise ValueError("patches_per_patient must be greater than zero.")
