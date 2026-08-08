from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from backend.preprocessing.store.artifact import (
    PatientArtifact,
    PatientArtifactMetadata,
)


class PreprocessedPatientStore:
    """
    Filesystem-backed cache for preprocessed patient artifacts.

    Each patient is stored independently so cache validation and
    invalidation can happen at patient level.
    """

    ARRAYS_FILENAME = "arrays.npz"
    METADATA_FILENAME = "metadata.json"

    def __init__(
        self,
        root_dir: str | Path,
    ) -> None:
        self.root_dir = Path(root_dir)

    def patient_dir(
        self,
        patient_id: str,
    ) -> Path:
        self._validate_patient_id(patient_id)

        return self.root_dir / patient_id

    def exists(
        self,
        patient_id: str,
    ) -> bool:
        patient_dir = self.patient_dir(patient_id)

        return (patient_dir / self.ARRAYS_FILENAME).is_file() and (
            patient_dir / self.METADATA_FILENAME
        ).is_file()

    def is_valid(
        self,
        patient_id: str,
        *,
        source_fingerprint: str,
        preprocessing_fingerprint: str,
        artifact_version: str,
    ) -> bool:
        """
        Check whether an existing cached artifact can be reused.

        Corrupt, incomplete, or incompatible cache entries are treated
        as invalid rather than crashing the caller.
        """

        if not self.exists(patient_id):
            return False

        try:
            metadata = self.load_metadata(patient_id)
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ):
            return False

        return (
            metadata.patient_id == patient_id
            and metadata.source_fingerprint == source_fingerprint
            and metadata.preprocessing_fingerprint == preprocessing_fingerprint
            and metadata.artifact_version == artifact_version
        )

    def load_metadata(
        self,
        patient_id: str,
    ) -> PatientArtifactMetadata:
        metadata_path = self.patient_dir(patient_id) / self.METADATA_FILENAME

        if not metadata_path.is_file():
            raise FileNotFoundError(f"Patient metadata not found: {metadata_path}")

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(file)

        if not isinstance(payload, dict):
            raise ValueError("Patient metadata must contain a JSON object.")

        return PatientArtifactMetadata.from_dict(payload)

    def load(
        self,
        patient_id: str,
    ) -> PatientArtifact:
        """
        Load and validate one complete patient artifact.
        """

        patient_dir = self.patient_dir(patient_id)

        arrays_path = patient_dir / self.ARRAYS_FILENAME

        if not arrays_path.is_file():
            raise FileNotFoundError(f"Patient arrays not found: {arrays_path}")

        metadata = self.load_metadata(patient_id)

        try:
            with np.load(
                arrays_path,
                allow_pickle=False,
            ) as arrays:
                image = np.array(
                    arrays["image"],
                    copy=True,
                )

                lung_mask = np.array(
                    arrays["lung_mask"],
                    copy=True,
                )

                nodule_mask = np.array(
                    arrays["nodule_mask"],
                    copy=True,
                )
        except (
            OSError,
            ValueError,
            KeyError,
        ) as error:
            raise ValueError(
                f"Failed to load patient artifact: {patient_id}"
            ) from error

        self._validate_array_metadata(
            metadata=metadata,
            image=image,
            lung_mask=lung_mask,
            nodule_mask=nodule_mask,
        )

        return PatientArtifact(
            patient_id=patient_id,
            image=image,
            lung_mask=lung_mask,
            nodule_mask=nodule_mask,
            metadata=metadata,
        )

    def save(
        self,
        artifact: PatientArtifact,
    ) -> None:
        """
        Persist a patient artifact.

        Data is first written to a temporary sibling directory and then
        moved into place so partially written artifacts are not exposed
        as valid cache entries.
        """

        self.root_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        target_dir = self.patient_dir(artifact.patient_id)

        temporary_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{artifact.patient_id}-",
                dir=self.root_dir,
            )
        )

        try:
            arrays_path = temporary_dir / self.ARRAYS_FILENAME

            metadata_path = temporary_dir / self.METADATA_FILENAME

            np.savez_compressed(
                arrays_path,
                image=artifact.image,
                lung_mask=artifact.lung_mask,
                nodule_mask=artifact.nodule_mask,
            )

            with metadata_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    artifact.metadata.to_dict(),
                    file,
                    indent=4,
                )

            if target_dir.exists():
                shutil.rmtree(target_dir)

            temporary_dir.replace(target_dir)

        except Exception:
            if temporary_dir.exists():
                shutil.rmtree(
                    temporary_dir,
                    ignore_errors=True,
                )

            raise

    def invalidate(
        self,
        patient_id: str,
    ) -> None:
        """
        Remove one patient's cached artifact.
        """

        patient_dir = self.patient_dir(patient_id)

        if patient_dir.exists():
            shutil.rmtree(patient_dir)

    @staticmethod
    def _validate_patient_id(
        patient_id: str,
    ) -> None:
        if not patient_id.strip():
            raise ValueError("patient_id cannot be empty.")

        if "/" in patient_id or "\\" in patient_id or patient_id in {".", ".."}:
            raise ValueError("patient_id contains invalid path characters.")

    @staticmethod
    def _validate_array_metadata(
        *,
        metadata: PatientArtifactMetadata,
        image: np.ndarray,
        lung_mask: np.ndarray,
        nodule_mask: np.ndarray,
    ) -> None:
        checks = (
            (
                "image",
                image,
                metadata.image_shape,
                metadata.image_dtype,
            ),
            (
                "lung_mask",
                lung_mask,
                metadata.lung_mask_shape,
                metadata.lung_mask_dtype,
            ),
            (
                "nodule_mask",
                nodule_mask,
                metadata.nodule_mask_shape,
                metadata.nodule_mask_dtype,
            ),
        )

        for (
            name,
            array,
            expected_shape,
            expected_dtype,
        ) in checks:
            if array.shape != expected_shape:
                raise ValueError(f"{name} shape does not match metadata.")

            if str(array.dtype) != expected_dtype:
                raise ValueError(f"{name} dtype does not match metadata.")
