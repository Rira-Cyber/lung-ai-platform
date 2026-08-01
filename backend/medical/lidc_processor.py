from pathlib import Path

import numpy as np
import pylidc as pl

from .ct_volume import CTVolume
from .lung_segmenter import LungSegmenter


class LIDCProcessor:
    """
    Central medical processor for a single LIDC patient.

    Responsibilities
    ----------------
    - Load CT volume
    - Load LIDC annotations
    - Build nodule mask
    - Build lung mask
    - Cache all computed medical data
    """

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)

        self.patient_id = None

        self.scan = None
        self.ct = None

        self.annotations = []

        # ---------- Cache ----------

        self._hu_volume = None

        self._nodule_mask = None

        self._lung_mask = None

        self._lung_segmenter = LungSegmenter(
            body_threshold=-600,
            lung_threshold=-320,
        )

    def load_patient(
        self,
        patient_id: str,
    ):
        """
        Load one patient and reset cached data.
        """

        # اگر همین بیمار قبلاً لود شده، هیچ کاری نکن
        if self.patient_id == patient_id and self.ct is not None:
            return self.scan

        self.patient_id = patient_id

        self.scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).first()

        if self.scan is None:
            raise ValueError(f"No scan found for {patient_id}")

        dicom_folder = self.scan.get_path_to_dicom_files()

        self.ct = CTVolume(dicom_folder)

        self.annotations = []

        self.clear_cache()

        return self.scan

    def load_annotations(self):
        """
        Load all clustered annotations.
        """

        if self.scan is None:
            raise RuntimeError("Patient has not been loaded.")

        self.annotations = []

        clusters = self.scan.cluster_annotations()

        for cluster in clusters:
            self.annotations.extend(cluster)

        return self.annotations

    def hu_volume(self):
        """
        Return CT volume in Hounsfield Units.

        Cached after first computation.
        """

        if self.ct is None:
            raise RuntimeError("CT volume not loaded.")

        if self._hu_volume is None:
            self._hu_volume = self.ct.to_hu()

        return self._hu_volume

    def nodule_mask(self):
        """
        Return the full-volume binary nodule mask.

        The mask is computed only once and cached.
        """

        if self.ct is None:
            raise RuntimeError("CT volume not loaded.")

        if not self.annotations:
            self.load_annotations()

        if self._nodule_mask is not None:
            return self._nodule_mask

        full_mask = np.zeros(self.ct.volume.shape, dtype=np.uint8)

        for annotation in self.annotations:
            bbox = annotation.bbox()

            # pylidc -> (Y, X, Z)
            y = bbox[0]
            x = bbox[1]
            z = bbox[2]

            mask = annotation.boolean_mask()

            # (Y,X,Z) -> (Z,Y,X)
            mask = np.transpose(mask, (2, 0, 1))

            full_mask[z, y, x] = np.logical_or(full_mask[z, y, x], mask)

        self._nodule_mask = full_mask

        return self._nodule_mask

    def lung_mask(self):
        """
        Return the binary lung mask.

        Computed only once and cached.
        """

        if self._lung_mask is not None:
            return self._lung_mask

        self._lung_mask = self._lung_segmenter.segment_volume(self.hu_volume())

        return self._lung_mask

    def __repr__(self):
        return (
            "LIDCProcessor\n"
            "-------------------------\n"
            f"Patient ID : {self.patient_id}\n"
            f"CT Shape   : {self.ct.shape() if self.ct else None}\n"
            f"Annotations: {len(self.annotations)}\n"
            f"HU Cached  : {self._hu_volume is not None}\n"
            f"Lung Cached: {self._lung_mask is not None}\n"
            f"Nodule Cached: {self._nodule_mask is not None}"
        )

    def clear_cache(self):
        """
        Clear all cached medical data.
        """

        self._hu_volume = None
        self._nodule_mask = None
        self._lung_mask = None
