from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.datasets.lidc_dataset import (
    LIDCDataset,
)
from backend.preprocessing.store import (
    PatientArtifact,
    PatientArtifactMetadata,
)


PATIENT_ID = "LIDC-IDRI-0001"


class FakePatientPreprocessor:
    def __init__(self) -> None:
        self.calls = 0

    def get(
        self,
        patient_id: str,
    ) -> PatientArtifact:
        self.calls += 1

        image = np.zeros(
            (8, 8, 8),
            dtype=np.float32,
        )

        lung_mask = np.ones(
            (8, 8, 8),
            dtype=np.uint8,
        )

        nodule_mask = np.zeros(
            (8, 8, 8),
            dtype=np.uint8,
        )

        nodule_mask[
            3:5,
            3:5,
            3:5,
        ] = 1

        metadata = PatientArtifactMetadata(
            artifact_version="1",
            patient_id=patient_id,
            source_fingerprint="source",
            preprocessing_fingerprint="preprocessing",
            image_shape=image.shape,
            image_dtype=str(image.dtype),
            lung_mask_shape=lung_mask.shape,
            lung_mask_dtype=str(lung_mask.dtype),
            nodule_mask_shape=nodule_mask.shape,
            nodule_mask_dtype=str(nodule_mask.dtype),
        )

        return PatientArtifact(
            patient_id=patient_id,
            image=image,
            lung_mask=lung_mask,
            nodule_mask=nodule_mask,
            metadata=metadata,
        )


class FakeSampler:
    def sample(
        self,
        *,
        image,
        nodule_mask,
        lung_mask,
        force_negative,
    ):
        return {
            "image": image[:4, :4, :4],
            "mask": nodule_mask[:4, :4, :4],
            "center": (2, 2, 2),
            "patch_bbox": (
                (0, 4),
                (0, 4),
                (0, 4),
            ),
            "is_positive": not force_negative,
        }


def create_dataset(
    tmp_path: Path,
    preprocessor: FakePatientPreprocessor,
) -> LIDCDataset:
    dataset_path = tmp_path / "raw"

    patient_path = dataset_path / PATIENT_ID

    patient_path.mkdir(parents=True)

    return LIDCDataset(
        dataset_path=dataset_path,
        sampler=FakeSampler(),
        patches_per_patient=2,
        patient_ids=[PATIENT_ID],
        patient_preprocessor=preprocessor,
    )


def test_dataset_uses_patient_preprocessor(
    tmp_path: Path,
) -> None:
    preprocessor = FakePatientPreprocessor()

    dataset = create_dataset(
        tmp_path,
        preprocessor,
    )

    initialization_calls = preprocessor.calls

    sample = dataset[0]

    assert preprocessor.calls == initialization_calls + 1

    assert sample["patient_id"] == PATIENT_ID

    assert sample["image"].shape == (
        1,
        4,
        4,
        4,
    )

    assert sample["mask"].shape == (
        1,
        4,
        4,
        4,
    )


def test_dataset_length_is_preserved(
    tmp_path: Path,
) -> None:
    preprocessor = FakePatientPreprocessor()

    dataset = create_dataset(
        tmp_path,
        preprocessor,
    )

    assert len(dataset) == 2


def test_dataset_preserves_patch_metadata(
    tmp_path: Path,
) -> None:
    preprocessor = FakePatientPreprocessor()

    dataset = create_dataset(
        tmp_path,
        preprocessor,
    )

    sample = dataset[0]

    assert sample["center"] == (2, 2, 2)

    assert sample["patch_bbox"] == (
        (0, 4),
        (0, 4),
        (0, 4),
    )

    assert sample["is_positive"] is True
