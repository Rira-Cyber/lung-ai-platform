from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from backend.datasets.validation_dataset import ValidationDataset
from backend.datasets.validation_metadata import (
    VALIDATION_METADATA_VERSION,
    ValidationMetadata,
    ValidationPatch,
)


class FakeLIDCProcessor:
    """
    Lightweight processor used only for the smoke test.

    It prevents access to the real LIDC-IDRI dataset and avoids
    expensive medical preprocessing.
    """

    def __init__(
        self,
        dataset_path,
    ) -> None:
        self.dataset_path = dataset_path
        self.patient_id: str | None = None

    def load_patient(
        self,
        patient_id: str,
    ) -> None:
        self.patient_id = patient_id

    def hu_volume(
        self,
    ) -> np.ndarray:
        return np.ones(
            (16, 16, 16),
            dtype=np.float32,
        )

    def nodule_mask(
        self,
    ) -> np.ndarray:
        mask = np.zeros(
            (16, 16, 16),
            dtype=np.uint8,
        )

        mask[
            7:9,
            7:9,
            7:9,
        ] = 1

        return mask


def test_validation_pipeline_smoke(
    tmp_path,
) -> None:
    """
    Verify validation metadata, dataset, patch extraction,
    tensor conversion, and DataLoader batching.
    """

    metadata = ValidationMetadata(
        version=VALIDATION_METADATA_VERSION,
        patch_size=(8, 8, 8),
        seed=42,
        patches=(
            ValidationPatch(
                patient_id="LIDC-IDRI-0001",
                patch_index=0,
                center=(8, 8, 8),
                is_positive=True,
            ),
            ValidationPatch(
                patient_id="LIDC-IDRI-0002",
                patch_index=0,
                center=(4, 4, 4),
                is_positive=False,
            ),
        ),
    )

    metadata_path = tmp_path / "validation_metadata.json"

    metadata.save(metadata_path)

    loaded_metadata = ValidationMetadata.load(metadata_path)

    dataset = ValidationDataset(
        dataset_path=tmp_path,
        metadata=loaded_metadata,
        processor_factory=FakeLIDCProcessor,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    batch = next(iter(loader))

    assert batch["image"].shape == (
        2,
        1,
        8,
        8,
        8,
    )

    assert batch["mask"].shape == (
        2,
        1,
        8,
        8,
        8,
    )

    assert batch["image"].dtype == (torch.float32)

    assert batch["mask"].dtype == (torch.float32)

    assert batch["patient_id"] == [
        "LIDC-IDRI-0001",
        "LIDC-IDRI-0002",
    ]

    assert batch["patch_index"].tolist() == [
        0,
        0,
    ]
