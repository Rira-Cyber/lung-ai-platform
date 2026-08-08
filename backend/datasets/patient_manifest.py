from __future__ import annotations

import json
from pathlib import Path


def load_patient_manifest(
    manifest_path: str | Path,
) -> tuple[str, ...]:
    """
    Load a reproducible patient subset manifest.

    Expected JSON structure:

    {
        "patient_ids": [
            "LIDC-IDRI-0001",
            "LIDC-IDRI-0002"
        ]
    }
    """

    path = Path(
        manifest_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Patient manifest not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Patient manifest is not a file: {path}"
        )

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(
                file
            )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Patient manifest contains invalid JSON: {path}"
        ) from error

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "Patient manifest must contain a JSON object."
        )

    patient_ids = payload.get(
        "patient_ids"
    )

    if not isinstance(
        patient_ids,
        list,
    ):
        raise ValueError(
            "Patient manifest must contain "
            "'patient_ids' as a list."
        )

    if not patient_ids:
        raise ValueError(
            "Patient manifest cannot be empty."
        )

    if not all(
        isinstance(
            patient_id,
            str,
        )
        and patient_id.strip()
        for patient_id in patient_ids
    ):
        raise ValueError(
            "Every patient ID must be a non-empty string."
        )

    normalized_patient_ids = [
        patient_id.strip()
        for patient_id in patient_ids
    ]

    if len(
        normalized_patient_ids
    ) != len(
        set(normalized_patient_ids)
    ):
        raise ValueError(
            "Patient manifest contains duplicate patient IDs."
        )

    return tuple(
        normalized_patient_ids
    )