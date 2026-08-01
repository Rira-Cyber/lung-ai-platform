from __future__ import annotations

import json

import pytest

from backend.datasets.validation_metadata import (
    VALIDATION_METADATA_VERSION,
    ValidationMetadata,
    ValidationPatch,
)


def create_sample_metadata() -> ValidationMetadata:
    return ValidationMetadata(
        version=VALIDATION_METADATA_VERSION,
        patch_size=(64, 64, 64),
        seed=42,
        patches=(
            ValidationPatch(
                patient_id="LIDC-IDRI-0001",
                patch_index=0,
                center=(10, 20, 30),
                is_positive=True,
            ),
            ValidationPatch(
                patient_id="LIDC-IDRI-0002",
                patch_index=0,
                center=(40, 50, 60),
                is_positive=False,
            ),
        ),
    )


def test_validation_patch_creation() -> None:
    patch = ValidationPatch(
        patient_id="LIDC-IDRI-0001",
        patch_index=0,
        center=(10, 20, 30),
        is_positive=True,
    )

    assert patch.patient_id == "LIDC-IDRI-0001"
    assert patch.patch_index == 0
    assert patch.center == (10, 20, 30)
    assert patch.is_positive is True


def test_validation_patch_from_dict() -> None:
    patch = ValidationPatch.from_dict(
        {
            "patient_id": "LIDC-IDRI-0001",
            "patch_index": 2,
            "center": [10, 20, 30],
            "is_positive": False,
        }
    )

    assert patch.patient_id == "LIDC-IDRI-0001"
    assert patch.patch_index == 2
    assert patch.center == (10, 20, 30)
    assert patch.is_positive is False


def test_validation_patch_rejects_empty_patient_id() -> None:
    with pytest.raises(
        ValueError,
        match="patient_id cannot be empty",
    ):
        ValidationPatch(
            patient_id="",
            patch_index=0,
            center=(10, 20, 30),
            is_positive=True,
        )


def test_validation_patch_rejects_negative_index() -> None:
    with pytest.raises(
        ValueError,
        match="patch_index cannot be negative",
    ):
        ValidationPatch(
            patient_id="LIDC-IDRI-0001",
            patch_index=-1,
            center=(10, 20, 30),
            is_positive=True,
        )


def test_validation_patch_rejects_invalid_center() -> None:
    with pytest.raises(
        ValueError,
        match="center must contain exactly three coordinates",
    ):
        ValidationPatch(
            patient_id="LIDC-IDRI-0001",
            patch_index=0,
            center=(10, 20),
            is_positive=True,
        )


def test_validation_patch_from_dict_rejects_missing_fields() -> None:
    with pytest.raises(
        ValueError,
        match="missing fields",
    ):
        ValidationPatch.from_dict(
            {
                "patient_id": "LIDC-IDRI-0001",
                "patch_index": 0,
            }
        )


def test_metadata_save_and_load_round_trip(
    tmp_path,
) -> None:
    metadata = create_sample_metadata()

    metadata_path = tmp_path / "validation_metadata.json"

    metadata.save(metadata_path)

    loaded_metadata = ValidationMetadata.load(metadata_path)

    assert loaded_metadata == metadata


def test_metadata_json_is_valid(
    tmp_path,
) -> None:
    metadata = create_sample_metadata()

    metadata_path = tmp_path / "validation_metadata.json"

    metadata.save(metadata_path)

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    assert payload["version"] == (VALIDATION_METADATA_VERSION)

    assert payload["patch_size"] == [
        64,
        64,
        64,
    ]

    assert payload["seed"] == 42

    assert len(payload["patches"]) == 2


def test_metadata_patient_ids_are_unique_and_sorted() -> None:
    metadata = create_sample_metadata()

    assert metadata.patient_ids == (
        "LIDC-IDRI-0001",
        "LIDC-IDRI-0002",
    )


def test_metadata_rejects_unsupported_version() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported validation metadata version",
    ):
        ValidationMetadata(
            version=999,
            patch_size=(64, 64, 64),
            seed=42,
            patches=(
                ValidationPatch(
                    patient_id="LIDC-IDRI-0001",
                    patch_index=0,
                    center=(10, 20, 30),
                    is_positive=True,
                ),
            ),
        )


def test_metadata_rejects_invalid_patch_size_length() -> None:
    with pytest.raises(
        ValueError,
        match="patch_size must contain exactly three values",
    ):
        ValidationMetadata(
            version=VALIDATION_METADATA_VERSION,
            patch_size=(64, 64),
            seed=42,
            patches=(
                ValidationPatch(
                    patient_id="LIDC-IDRI-0001",
                    patch_index=0,
                    center=(10, 20, 30),
                    is_positive=True,
                ),
            ),
        )


def test_metadata_rejects_non_positive_patch_size() -> None:
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        ValidationMetadata(
            version=VALIDATION_METADATA_VERSION,
            patch_size=(64, 0, 64),
            seed=42,
            patches=(
                ValidationPatch(
                    patient_id="LIDC-IDRI-0001",
                    patch_index=0,
                    center=(10, 20, 30),
                    is_positive=True,
                ),
            ),
        )


def test_metadata_rejects_empty_patches() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        ValidationMetadata(
            version=VALIDATION_METADATA_VERSION,
            patch_size=(64, 64, 64),
            seed=42,
            patches=(),
        )


def test_metadata_rejects_duplicate_patch_identity() -> None:
    duplicate_patch = ValidationPatch(
        patient_id="LIDC-IDRI-0001",
        patch_index=0,
        center=(10, 20, 30),
        is_positive=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate patch identities",
    ):
        ValidationMetadata(
            version=VALIDATION_METADATA_VERSION,
            patch_size=(64, 64, 64),
            seed=42,
            patches=(
                duplicate_patch,
                duplicate_patch,
            ),
        )


def test_metadata_load_rejects_missing_file(
    tmp_path,
) -> None:
    missing_path = tmp_path / "missing.json"

    with pytest.raises(
        FileNotFoundError,
        match="Validation metadata file not found",
    ):
        ValidationMetadata.load(missing_path)


def test_metadata_load_rejects_missing_fields(
    tmp_path,
) -> None:
    metadata_path = tmp_path / "invalid_metadata.json"

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "version": (VALIDATION_METADATA_VERSION),
                "patch_size": [
                    64,
                    64,
                    64,
                ],
            },
            file,
        )

    with pytest.raises(
        ValueError,
        match="missing fields",
    ):
        ValidationMetadata.load(metadata_path)
