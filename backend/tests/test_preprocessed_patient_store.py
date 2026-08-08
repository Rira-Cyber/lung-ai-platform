from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.preprocessing.store import (
    PatientArtifact,
    PatientArtifactMetadata,
    PreprocessedPatientStore,
)


PATIENT_ID = "LIDC-IDRI-0001"


def create_artifact() -> PatientArtifact:
    image = np.zeros(
        (4, 4, 4),
        dtype=np.float32,
    )

    lung_mask = np.ones(
        (4, 4, 4),
        dtype=bool,
    )

    nodule_mask = np.zeros(
        (4, 4, 4),
        dtype=bool,
    )

    metadata = PatientArtifactMetadata(
        artifact_version="1",
        patient_id=PATIENT_ID,
        source_fingerprint="source-v1",
        preprocessing_fingerprint="preprocessing-v1",
        image_shape=image.shape,
        image_dtype=str(image.dtype),
        lung_mask_shape=lung_mask.shape,
        lung_mask_dtype=str(lung_mask.dtype),
        nodule_mask_shape=nodule_mask.shape,
        nodule_mask_dtype=str(nodule_mask.dtype),
    )

    return PatientArtifact(
        patient_id=PATIENT_ID,
        image=image,
        lung_mask=lung_mask,
        nodule_mask=nodule_mask,
        metadata=metadata,
    )


def test_save_and_load_round_trip(
    tmp_path: Path,
) -> None:
    store = PreprocessedPatientStore(tmp_path)

    artifact = create_artifact()

    store.save(artifact)

    loaded = store.load(PATIENT_ID)

    assert loaded.patient_id == PATIENT_ID

    np.testing.assert_array_equal(
        loaded.image,
        artifact.image,
    )

    np.testing.assert_array_equal(
        loaded.lung_mask,
        artifact.lung_mask,
    )

    np.testing.assert_array_equal(
        loaded.nodule_mask,
        artifact.nodule_mask,
    )


def test_exists_requires_complete_artifact(
    tmp_path: Path,
) -> None:
    store = PreprocessedPatientStore(tmp_path)

    patient_dir = store.patient_dir(PATIENT_ID)

    patient_dir.mkdir(parents=True)

    assert not store.exists(PATIENT_ID)


def test_valid_artifact_is_reusable(
    tmp_path: Path,
) -> None:
    store = PreprocessedPatientStore(tmp_path)

    store.save(create_artifact())

    assert store.is_valid(
        PATIENT_ID,
        source_fingerprint="source-v1",
        preprocessing_fingerprint="preprocessing-v1",
        artifact_version="1",
    )


def test_source_change_invalidates_artifact(
    tmp_path: Path,
) -> None:
    store = PreprocessedPatientStore(tmp_path)

    store.save(create_artifact())

    assert not store.is_valid(
        PATIENT_ID,
        source_fingerprint="source-v2",
        preprocessing_fingerprint="preprocessing-v1",
        artifact_version="1",
    )


def test_preprocessing_change_invalidates_artifact(
    tmp_path: Path,
) -> None:
    store = PreprocessedPatientStore(tmp_path)

    store.save(create_artifact())

    assert not store.is_valid(
        PATIENT_ID,
        source_fingerprint="source-v1",
        preprocessing_fingerprint="preprocessing-v2",
        artifact_version="1",
    )


def test_invalidate_removes_only_patient(
    tmp_path: Path,
) -> None:
    store = PreprocessedPatientStore(tmp_path)

    store.save(create_artifact())

    store.invalidate(PATIENT_ID)

    assert not store.exists(PATIENT_ID)


def test_load_missing_artifact_raises(
    tmp_path: Path,
) -> None:
    store = PreprocessedPatientStore(tmp_path)

    with pytest.raises(FileNotFoundError):
        store.load(PATIENT_ID)


def test_invalid_patient_id_is_rejected(
    tmp_path: Path,
) -> None:
    store = PreprocessedPatientStore(tmp_path)

    with pytest.raises(ValueError):
        store.patient_dir("../patient")
