from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from backend.medical.lidc_processor import LIDCProcessor
from backend.preprocessing.store import (
    PatientArtifact,
    PatientArtifactMetadata,
    PreprocessedPatientStore,
    calculate_preprocessing_fingerprint,
    calculate_source_fingerprint,
)


class PatientPreprocessor:
    """
    Orchestrate incremental preprocessing for LIDC patients.

    The preprocessor decides whether a cached patient artifact
    can be reused or must be rebuilt from the raw medical data.
    """

    ARTIFACT_VERSION = "1"

    def __init__(
        self,
        dataset_path: str | Path,
        store: PreprocessedPatientStore,
        consensus_level: float = 0.5,
        processor: LIDCProcessor | None = None,
    ) -> None:
        if not 0.0 < consensus_level <= 1.0:
            raise ValueError("consensus_level must be in the range (0.0, 1.0].")

        self.dataset_path = Path(dataset_path)

        self.store = store

        self.consensus_level = float(consensus_level)

        self.processor = (
            processor
            if processor is not None
            else LIDCProcessor(
                dataset_path=self.dataset_path,
                consensus_level=self.consensus_level,
            )
        )

    def get(
        self,
        patient_id: str,
    ) -> PatientArtifact:
        """
        Return a valid patient artifact.

        A valid cache entry is reused. Missing, stale, or corrupt
        artifacts are rebuilt automatically from raw LIDC data.
        """

        source_fingerprint = self._resolve_source_fingerprint(patient_id)

        preprocessing_fingerprint = self.preprocessing_fingerprint()

        if self.store.is_valid(
            patient_id,
            source_fingerprint=source_fingerprint,
            preprocessing_fingerprint=(preprocessing_fingerprint),
            artifact_version=self.ARTIFACT_VERSION,
        ):
            try:
                return self.store.load(patient_id)
            except (
                OSError,
                ValueError,
                KeyError,
            ):
                self.store.invalidate(patient_id)

        return self._build(
            patient_id=patient_id,
            source_fingerprint=source_fingerprint,
            preprocessing_fingerprint=(preprocessing_fingerprint),
        )

    def rebuild(
        self,
        patient_id: str,
    ) -> PatientArtifact:
        """
        Force rebuilding one patient artifact.
        """

        source_fingerprint = self._resolve_source_fingerprint(patient_id)

        return self._build(
            patient_id=patient_id,
            source_fingerprint=source_fingerprint,
            preprocessing_fingerprint=(self.preprocessing_fingerprint()),
        )

    def preprocessing_parameters(
        self,
    ) -> dict[str, Any]:
        """
        Return all preprocessing parameters that affect
        the generated artifact.

        Any future preprocessing parameter that changes image
        or mask generation must be added here.
        """

        return {
            "artifact_version": self.ARTIFACT_VERSION,
            "consensus_level": self.consensus_level,
            "lung_segmentation": {
                "body_threshold": -600,
                "lung_threshold": -320,
            },
        }

    def preprocessing_fingerprint(
        self,
    ) -> str:
        return calculate_preprocessing_fingerprint(self.preprocessing_parameters())

    def _resolve_source_fingerprint(
        self,
        patient_id: str,
    ) -> str:
        """
        Load the patient just far enough to resolve its actual
        DICOM directory and fingerprint the raw source.
        """

        scan = self.processor.load_patient(patient_id)

        dicom_path = Path(scan.get_path_to_dicom_files())

        return calculate_source_fingerprint(dicom_path)

    def _build(
        self,
        *,
        patient_id: str,
        source_fingerprint: str,
        preprocessing_fingerprint: str,
    ) -> PatientArtifact:
        """
        Build, persist, and return one patient artifact.
        """

        self.processor.load_patient(patient_id)

        image = np.asarray(
            self.processor.hu_volume(),
            dtype=np.float32,
        )

        lung_mask = np.asarray(
            self.processor.lung_mask(),
            dtype=np.uint8,
        )

        nodule_mask = np.asarray(
            self.processor.nodule_mask(),
            dtype=np.uint8,
        )

        self._validate_arrays(
            image=image,
            lung_mask=lung_mask,
            nodule_mask=nodule_mask,
        )

        metadata = PatientArtifactMetadata(
            artifact_version=self.ARTIFACT_VERSION,
            patient_id=patient_id,
            source_fingerprint=source_fingerprint,
            preprocessing_fingerprint=(preprocessing_fingerprint),
            image_shape=image.shape,
            image_dtype=str(image.dtype),
            lung_mask_shape=lung_mask.shape,
            lung_mask_dtype=str(lung_mask.dtype),
            nodule_mask_shape=nodule_mask.shape,
            nodule_mask_dtype=str(nodule_mask.dtype),
        )

        artifact = PatientArtifact(
            patient_id=patient_id,
            image=image,
            lung_mask=lung_mask,
            nodule_mask=nodule_mask,
            metadata=metadata,
        )

        self.store.save(artifact)

        return artifact

    @staticmethod
    def _validate_arrays(
        *,
        image: np.ndarray,
        lung_mask: np.ndarray,
        nodule_mask: np.ndarray,
    ) -> None:
        if image.ndim != 3:
            raise ValueError("Patient image must be a 3D volume.")

        if image.shape != lung_mask.shape:
            raise ValueError("Image and lung mask must have identical shapes.")

        if image.shape != nodule_mask.shape:
            raise ValueError("Image and nodule mask must have identical shapes.")

        if not np.isfinite(image).all():
            raise ValueError("Patient image contains non-finite values.")

    def is_cached(
        self,
        patient_id: str,
    ) -> bool:
        """
        Return whether the patient currently has a valid,
        reusable preprocessing artifact.
        """

        source_fingerprint = self._resolve_source_fingerprint(patient_id)

        return self.store.is_valid(
            patient_id,
            source_fingerprint=source_fingerprint,
            preprocessing_fingerprint=(self.preprocessing_fingerprint()),
            artifact_version=self.ARTIFACT_VERSION,
        )
