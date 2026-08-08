from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def calculate_source_fingerprint(
    patient_path: str | Path,
) -> str:
    """
    Calculate a deterministic fingerprint for one patient's
    raw source files.

    The fingerprint reflects relative paths, file sizes,
    and modification timestamps without reading entire DICOM
    files into memory.
    """

    path = Path(patient_path)

    if not path.exists():
        raise FileNotFoundError(f"Patient source path not found: {path}")

    if not path.is_dir():
        raise ValueError(f"Patient source path is not a directory: {path}")

    hasher = hashlib.sha256()

    files = sorted(file for file in path.rglob("*") if file.is_file())

    if not files:
        raise ValueError(f"Patient source path contains no files: {path}")

    for file in files:
        stat = file.stat()

        relative_path = file.relative_to(path).as_posix()

        hasher.update(relative_path.encode("utf-8"))

        hasher.update(str(stat.st_size).encode("utf-8"))

        hasher.update(str(stat.st_mtime_ns).encode("utf-8"))

    return hasher.hexdigest()


def calculate_preprocessing_fingerprint(
    parameters: dict[str, Any],
) -> str:
    """
    Calculate a deterministic fingerprint for preprocessing
    parameters that affect generated patient artifacts.
    """

    serialized = json.dumps(
        parameters,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )

    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
