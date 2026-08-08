from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.preprocessing.store import (
    PatientArtifact,
    PatientArtifactMetadata,
    calculate_preprocessing_fingerprint,
    calculate_source_fingerprint,
)


def create_metadata() -> PatientArtifactMetadata:
    return PatientArtifactMetadata(
        artifact_version="1",
        patient_id="LIDC-IDRI-0001",
        source_fingerprint="source",
        preprocessing_fingerprint="preprocessing",
        image_shape=(4, 4, 4),
        image_dtype="float32",
        lung_mask_shape=(4, 4, 4),
        lung_mask_dtype="bool",
        nodule_mask_shape=(4, 4, 4),
        nodule_mask_dtype="bool",
    )


def test_artifact_accepts_matching_arrays() -> None:
    image = np.zeros(
        (4, 4, 4),
        dtype=np.float32,
    )

    lung_mask = np.zeros(
        (4, 4, 4),
        dtype=bool,
    )

    nodule_mask = np.zeros(
        (4, 4, 4),
        dtype=bool,
    )

    artifact = PatientArtifact(
        patient_id="LIDC-IDRI-0001",
        image=image,
        lung_mask=lung_mask,
        nodule_mask=nodule_mask,
        metadata=create_metadata(),
    )

    assert artifact.image.shape == (4, 4, 4)


def test_artifact_rejects_shape_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="lung_mask",
    ):
        PatientArtifact(
            patient_id="LIDC-IDRI-0001",
            image=np.zeros((4, 4, 4)),
            lung_mask=np.zeros((2, 2, 2)),
            nodule_mask=np.zeros((4, 4, 4)),
            metadata=create_metadata(),
        )


def test_metadata_round_trip() -> None:
    metadata = create_metadata()

    restored = PatientArtifactMetadata.from_dict(metadata.to_dict())

    assert restored == metadata


def test_preprocessing_fingerprint_is_deterministic() -> None:
    first = calculate_preprocessing_fingerprint(
        {
            "consensus_level": 0.5,
            "hu_min": -1000,
        }
    )

    second = calculate_preprocessing_fingerprint(
        {
            "hu_min": -1000,
            "consensus_level": 0.5,
        }
    )

    assert first == second


def test_preprocessing_fingerprint_changes_with_config() -> None:
    first = calculate_preprocessing_fingerprint({"consensus_level": 0.5})

    second = calculate_preprocessing_fingerprint({"consensus_level": 0.6})

    assert first != second


def test_source_fingerprint_is_deterministic(
    tmp_path: Path,
) -> None:
    patient_path = tmp_path / "patient"
    patient_path.mkdir()

    (patient_path / "slice1.dcm").write_bytes(b"dicom-one")

    first = calculate_source_fingerprint(patient_path)

    second = calculate_source_fingerprint(patient_path)

    assert first == second
