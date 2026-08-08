from __future__ import annotations

import numpy as np
import torch

from backend.training.dataloader import (
    create_train_validation_loaders,
)


class FakeLIDCDataset:
    """
    Lightweight replacement for LIDCDataset.

    It exposes patient IDs during discovery and returns synthetic
    training patches after the patient split is applied.
    """

    AVAILABLE_PATIENT_IDS = [
        "LIDC-IDRI-0001",
        "LIDC-IDRI-0002",
        "LIDC-IDRI-0003",
        "LIDC-IDRI-0004",
        "LIDC-IDRI-0005",
    ]

    def __init__(
        self,
        dataset_path,
        sampler,
        transforms=None,
        patches_per_patient: int = 20,
        patient_ids=None,
    ) -> None:
        self.dataset_path = dataset_path
        self.sampler = sampler
        self.transforms = transforms
        self.patches_per_patient = patches_per_patient

        self.patient_ids = (
            list(patient_ids)
            if patient_ids is not None
            else self.AVAILABLE_PATIENT_IDS.copy()
        )

    def __len__(self) -> int:
        return len(self.patient_ids) * self.patches_per_patient

    def __getitem__(
        self,
        index: int,
    ) -> dict:
        patient_index = index // self.patches_per_patient

        patient_id = self.patient_ids[patient_index]

        image = torch.zeros(
            (1, 8, 8, 8),
            dtype=torch.float32,
        )

        mask = torch.zeros(
            (1, 8, 8, 8),
            dtype=torch.float32,
        )

        return {
            "image": image,
            "mask": mask,
            "patient_id": patient_id,
            "center": np.array(
                [4, 4, 4],
                dtype=np.int32,
            ),
            "patch_bbox": (
                np.array(
                    [0, 0, 0],
                    dtype=np.int32,
                ),
                np.array(
                    [8, 8, 8],
                    dtype=np.int32,
                ),
            ),
            "is_positive": False,
        }


class FakeValidationDataset:
    """
    Lightweight replacement for ValidationDataset.
    """

    def __init__(
        self,
        dataset_path,
        metadata,
        transforms=None,
        processor_factory=None,
    ) -> None:
        self.dataset_path = dataset_path
        self.metadata = metadata
        self.transforms = transforms
        self.processor_factory = processor_factory

    def __len__(self) -> int:
        return len(self.metadata.patches)

    def __getitem__(
        self,
        index: int,
    ) -> dict:
        patch = self.metadata.patches[index]

        return {
            "image": torch.zeros(
                (1, 8, 8, 8),
                dtype=torch.float32,
            ),
            "mask": torch.zeros(
                (1, 8, 8, 8),
                dtype=torch.float32,
            ),
            "patient_id": (patch.patient_id),
            "patch_index": (patch.patch_index),
            "center": np.asarray(
                patch.center,
                dtype=np.int32,
            ),
            "patch_bbox": (
                np.array(
                    [0, 0, 0],
                    dtype=np.int32,
                ),
                np.array(
                    [8, 8, 8],
                    dtype=np.int32,
                ),
            ),
            "is_positive": (patch.is_positive),
        }


class FakeMetadataGenerator:
    def __init__(
        self,
        dataset_path,
        patch_size,
        positive_ratio,
        patches_per_patient,
        seed,
    ) -> None:
        self.dataset_path = dataset_path
        self.patch_size = patch_size
        self.positive_ratio = positive_ratio
        self.patches_per_patient = patches_per_patient
        self.seed = seed

    def generate_or_load(
        self,
        patient_ids,
        metadata_path,
        *,
        overwrite=False,
    ):
        from backend.datasets.validation_metadata import (
            VALIDATION_METADATA_VERSION,
            ValidationMetadata,
            ValidationPatch,
        )

        patches = tuple(
            ValidationPatch(
                patient_id=patient_id,
                patch_index=patch_index,
                center=(4, 4, 4),
                is_positive=False,
            )
            for patient_id in patient_ids
            for patch_index in range(self.patches_per_patient)
        )

        return ValidationMetadata(
            version=(VALIDATION_METADATA_VERSION),
            patch_size=tuple(self.patch_size),
            seed=self.seed,
            patches=patches,
        )


def test_train_validation_loader_factory(
    tmp_path,
) -> None:
    loaders = create_train_validation_loaders(
        dataset_path=tmp_path,
        batch_size=2,
        num_workers=0,
        patch_size=(8, 8, 8),
        positive_ratio=0.5,
        patches_per_patient=2,
        val_ratio=0.4,
        test_ratio=0.2,
        split_seed=42,
        pin_memory=False,
        shuffle=True,
        validation_metadata_path=(
            tmp_path / "validation_metadata.json"
        ),
        training_dataset_factory=FakeLIDCDataset,
        validation_dataset_factory=FakeValidationDataset,
        metadata_generator_factory=FakeMetadataGenerator,
    )

    train_ids = set(
        loaders.patient_split.train_ids
    )

    val_ids = set(
        loaders.patient_split.val_ids
    )

    test_ids = set(
        loaders.patient_split.test_ids
    )

    all_ids = set(
        FakeLIDCDataset.AVAILABLE_PATIENT_IDS
    )

    assert train_ids.isdisjoint(
        val_ids
    )

    assert train_ids.isdisjoint(
        test_ids
    )

    assert val_ids.isdisjoint(
        test_ids
    )

    assert (
        train_ids
        | val_ids
        | test_ids
    ) == all_ids

    assert len(loaders.train_loader.dataset) == (len(train_ids) * 2)

    assert len(loaders.val_loader.dataset) == (len(val_ids) * 2)

    train_batch = next(iter(loaders.train_loader))

    val_batch = next(iter(loaders.val_loader))

    assert train_batch["image"].shape == (
        2,
        1,
        8,
        8,
        8,
    )

    assert train_batch["mask"].shape == (
        2,
        1,
        8,
        8,
        8,
    )

    assert val_batch["image"].shape == (
        2,
        1,
        8,
        8,
        8,
    )

    assert val_batch["mask"].shape == (
        2,
        1,
        8,
        8,
        8,
    )

    assert loaders.validation_metadata_path == tmp_path / "validation_metadata.json"
