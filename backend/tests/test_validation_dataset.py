from __future__ import annotations

import numpy as np
import pytest
import torch

from backend.datasets.validation_dataset import ValidationDataset
from backend.datasets.validation_metadata import (
    VALIDATION_METADATA_VERSION,
    ValidationMetadata,
    ValidationPatch,
)


class FakeLIDCProcessor:
    """
    Lightweight processor for ValidationDataset unit tests.

    No DICOM, pylidc, or real medical preprocessing is used.
    """

    def __init__(
        self,
        dataset_path,
    ) -> None:
        self.dataset_path = dataset_path
        self.loaded_patient_id: str | None = None

    def load_patient(
        self,
        patient_id: str,
    ) -> None:
        self.loaded_patient_id = patient_id

    def hu_volume(
        self,
    ) -> np.ndarray:
        return np.arange(
            16 * 16 * 16,
            dtype=np.float32,
        ).reshape(
            16,
            16,
            16,
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


def create_metadata() -> ValidationMetadata:
    return ValidationMetadata(
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
                patch_index=1,
                center=(4, 4, 4),
                is_positive=False,
            ),
        ),
    )


def create_dataset(
    tmp_path,
    transforms=None,
) -> ValidationDataset:
    return ValidationDataset(
        dataset_path=tmp_path,
        metadata=create_metadata(),
        transforms=transforms,
        processor_factory=FakeLIDCProcessor,
    )


def test_dataset_length(
    tmp_path,
) -> None:
    dataset = create_dataset(tmp_path)

    assert len(dataset) == 2


def test_dataset_returns_expected_keys(
    tmp_path,
) -> None:
    dataset = create_dataset(tmp_path)

    sample = dataset[0]

    assert set(sample) == {
        "image",
        "mask",
        "patient_id",
        "patch_index",
        "center",
        "patch_bbox",
        "is_positive",
    }


def test_dataset_returns_expected_tensor_shapes(
    tmp_path,
) -> None:
    dataset = create_dataset(tmp_path)

    sample = dataset[0]

    assert sample["image"].shape == (
        1,
        8,
        8,
        8,
    )

    assert sample["mask"].shape == (
        1,
        8,
        8,
        8,
    )


def test_dataset_returns_float32_tensors(
    tmp_path,
) -> None:
    dataset = create_dataset(tmp_path)

    sample = dataset[0]

    assert sample["image"].dtype == (torch.float32)

    assert sample["mask"].dtype == (torch.float32)


def test_dataset_preserves_metadata(
    tmp_path,
) -> None:
    dataset = create_dataset(tmp_path)

    sample = dataset[0]

    assert sample["patient_id"] == ("LIDC-IDRI-0001")

    assert sample["patch_index"] == 0

    assert sample["is_positive"] is True

    assert tuple(sample["center"]) == (
        8,
        8,
        8,
    )


def test_dataset_is_deterministic(
    tmp_path,
) -> None:
    dataset = create_dataset(tmp_path)

    first = dataset[0]
    second = dataset[0]

    assert torch.equal(
        first["image"],
        second["image"],
    )

    assert torch.equal(
        first["mask"],
        second["mask"],
    )

    assert tuple(first["center"]) == tuple(second["center"])


def test_dataset_loads_correct_patient(
    tmp_path,
) -> None:
    dataset = create_dataset(tmp_path)

    dataset[1]

    assert dataset.processor.loaded_patient_id == "LIDC-IDRI-0002"


def test_dataset_rejects_negative_index(
    tmp_path,
) -> None:
    dataset = create_dataset(tmp_path)

    with pytest.raises(
        IndexError,
        match="out of range",
    ):
        dataset[-1]


def test_dataset_rejects_index_equal_to_length(
    tmp_path,
) -> None:
    dataset = create_dataset(tmp_path)

    with pytest.raises(
        IndexError,
        match="out of range",
    ):
        dataset[len(dataset)]


def test_dataset_applies_transforms(
    tmp_path,
) -> None:
    def transform(
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        return (
            image + 10.0,
            mask * 2,
        )

    dataset = create_dataset(
        tmp_path,
        transforms=transform,
    )

    transformed = dataset[0]

    plain_dataset = create_dataset(tmp_path)

    plain = plain_dataset[0]

    assert torch.equal(
        transformed["image"],
        plain["image"] + 10.0,
    )

    assert torch.equal(
        transformed["mask"],
        plain["mask"] * 2,
    )


def test_positive_patch_contains_nodule_voxels(
    tmp_path,
) -> None:
    dataset = create_dataset(tmp_path)

    sample = dataset[0]

    assert sample["mask"].sum() > 0


def test_negative_metadata_is_preserved(
    tmp_path,
) -> None:
    dataset = create_dataset(tmp_path)

    sample = dataset[1]

    assert sample["is_positive"] is False


def test_patch_bbox_has_start_and_stop(
    tmp_path,
) -> None:
    dataset = create_dataset(tmp_path)

    sample = dataset[0]

    start, stop = sample["patch_bbox"]

    assert tuple(start) == (
        4,
        4,
        4,
    )

    assert tuple(stop) == (
        12,
        12,
        12,
    )
