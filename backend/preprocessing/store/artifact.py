from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PatientArtifactMetadata:
    """
    Metadata required to validate and reproduce
    a preprocessed patient artifact.
    """

    artifact_version: str
    patient_id: str
    source_fingerprint: str
    preprocessing_fingerprint: str
    image_shape: tuple[int, ...]
    image_dtype: str
    lung_mask_shape: tuple[int, ...]
    lung_mask_dtype: str
    nodule_mask_shape: tuple[int, ...]
    nodule_mask_dtype: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_version": self.artifact_version,
            "patient_id": self.patient_id,
            "source_fingerprint": self.source_fingerprint,
            "preprocessing_fingerprint": self.preprocessing_fingerprint,
            "image_shape": list(self.image_shape),
            "image_dtype": self.image_dtype,
            "lung_mask_shape": list(self.lung_mask_shape),
            "lung_mask_dtype": self.lung_mask_dtype,
            "nodule_mask_shape": list(self.nodule_mask_shape),
            "nodule_mask_dtype": self.nodule_mask_dtype,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> PatientArtifactMetadata:
        return cls(
            artifact_version=str(payload["artifact_version"]),
            patient_id=str(payload["patient_id"]),
            source_fingerprint=str(payload["source_fingerprint"]),
            preprocessing_fingerprint=str(payload["preprocessing_fingerprint"]),
            image_shape=tuple(payload["image_shape"]),
            image_dtype=str(payload["image_dtype"]),
            lung_mask_shape=tuple(payload["lung_mask_shape"]),
            lung_mask_dtype=str(payload["lung_mask_dtype"]),
            nodule_mask_shape=tuple(payload["nodule_mask_shape"]),
            nodule_mask_dtype=str(payload["nodule_mask_dtype"]),
        )


@dataclass(frozen=True)
class PatientArtifact:
    """
    Preprocessed medical data for one patient.
    """

    patient_id: str
    image: NDArray[np.generic]
    lung_mask: NDArray[np.generic]
    nodule_mask: NDArray[np.generic]
    metadata: PatientArtifactMetadata

    def __post_init__(self) -> None:
        if not self.patient_id:
            raise ValueError("patient_id cannot be empty.")

        if self.patient_id != self.metadata.patient_id:
            raise ValueError("Artifact patient_id must match metadata patient_id.")

        if self.image.shape != self.lung_mask.shape:
            raise ValueError("image and lung_mask must have identical shapes.")

        if self.image.shape != self.nodule_mask.shape:
            raise ValueError("image and nodule_mask must have identical shapes.")
