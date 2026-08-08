from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.datasets.patient_manifest import (
    load_patient_manifest,
)


def test_load_patient_manifest(
    tmp_path: Path,
) -> None:
    manifest_path = (
        tmp_path
        / "patients.json"
    )

    manifest_path.write_text(
        json.dumps(
            {
                "patient_ids": [
                    "LIDC-IDRI-0001",
                    "LIDC-IDRI-0002",
                ]
            }
        ),
        encoding="utf-8",
    )

    patient_ids = (
        load_patient_manifest(
            manifest_path
        )
    )

    assert patient_ids == (
        "LIDC-IDRI-0001",
        "LIDC-IDRI-0002",
    )


def test_missing_manifest_raises_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="not found",
    ):
        load_patient_manifest(
            tmp_path
            / "missing.json"
        )


def test_empty_manifest_raises_error(
    tmp_path: Path,
) -> None:
    manifest_path = (
        tmp_path
        / "patients.json"
    )

    manifest_path.write_text(
        json.dumps(
            {
                "patient_ids": []
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        load_patient_manifest(
            manifest_path
        )


def test_duplicate_patient_ids_raise_error(
    tmp_path: Path,
) -> None:
    manifest_path = (
        tmp_path
        / "patients.json"
    )

    manifest_path.write_text(
        json.dumps(
            {
                "patient_ids": [
                    "LIDC-IDRI-0001",
                    "LIDC-IDRI-0001",
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        load_patient_manifest(
            manifest_path
        )


def test_invalid_json_raises_error(
    tmp_path: Path,
) -> None:
    manifest_path = (
        tmp_path
        / "patients.json"
    )

    manifest_path.write_text(
        "{broken-json",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="invalid JSON",
    ):
        load_patient_manifest(
            manifest_path
        )