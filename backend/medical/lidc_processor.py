from pathlib import Path

import numpy as np
import pylidc as pl
from pylidc.utils import consensus

from .ct_volume import CTVolume
from .lung_segmenter import LungSegmenter


class LIDCProcessor:
    """
    Central medical processor for a single LIDC patient.

    Responsibilities
    ----------------
    - Load CT volume
    - Load clustered LIDC annotations
    - Build consensus nodule mask
    - Build lung mask
    - Cache all computed medical data
    """

    def __init__(
        self,
        dataset_path,
        consensus_level: float = 0.5,
    ):
        if not 0.0 < consensus_level <= 1.0:
            raise ValueError("consensus_level must be in the range (0.0, 1.0].")

        self.dataset_path = Path(dataset_path)

        self.consensus_level = consensus_level

        self.patient_id = None

        self.scan = None
        self.ct = None

        self.annotation_clusters = []

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

        if self.patient_id == patient_id and self.ct is not None:
            return self.scan

        self.patient_id = patient_id

        self.scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).first()

        if self.scan is None:
            raise ValueError(f"No scan found for {patient_id}")

        dicom_folder = self.scan.get_path_to_dicom_files()

        self.ct = CTVolume(dicom_folder)

        self.annotation_clusters = []

        self.clear_cache()

        return self.scan

    def load_annotations(
        self,
    ):
        """
        Load annotation clusters.

        Each cluster represents annotations referring to the same
        physical nodule.
        """

        if self.scan is None:
            raise RuntimeError("Patient has not been loaded.")

        self.annotation_clusters = self.scan.cluster_annotations()

        return self.annotation_clusters

    def hu_volume(
        self,
    ):
        """
        Return CT volume in Hounsfield Units.

        Cached after first computation.
        """

        if self.ct is None:
            raise RuntimeError("CT volume not loaded.")

        if self._hu_volume is None:
            self._hu_volume = self.ct.to_hu()

        return self._hu_volume

    def nodule_mask(
        self,
    ):
        """
        Return the full-volume binary consensus nodule mask.

        Annotation masks belonging to the same physical nodule are
        consolidated using pylidc consensus at ``consensus_level``.

        Consensus masks from separate physical nodules are combined
        into one full-volume binary mask.

        Consensus bounding boxes that partially extend outside the CT
        volume are clipped safely before being merged.
        """

        if self.ct is None:
            raise RuntimeError("CT volume not loaded.")

        if self._nodule_mask is not None:
            return self._nodule_mask

        if not self.annotation_clusters:
            self.load_annotations()

        full_mask = np.zeros(
            self.ct.volume.shape,
            dtype=np.uint8,
        )

        for cluster in self.annotation_clusters:
            if not cluster:
                continue

            consensus_mask, bbox = consensus(
                cluster,
                clevel=self.consensus_level,
                ret_masks=False,
            )

            consensus_mask = np.transpose(
                consensus_mask,
                (2, 0, 1),
            )

            y_slice = bbox[0]
            x_slice = bbox[1]
            z_slice = bbox[2]

            target_slices, mask_slices = self._clip_consensus_region(
                volume_shape=full_mask.shape,
                z_slice=z_slice,
                y_slice=y_slice,
                x_slice=x_slice,
            )

            clipped_consensus_mask = consensus_mask[mask_slices]

            if clipped_consensus_mask.size == 0:
                continue

            target_view = full_mask[target_slices]

            if target_view.shape != clipped_consensus_mask.shape:
                raise ValueError(
                    "Consensus mask shape does not match the clipped "
                    "CT target region: "
                    f"target={target_view.shape}, "
                    f"mask={clipped_consensus_mask.shape}"
                )

            full_mask[target_slices] = np.logical_or(
                target_view,
                clipped_consensus_mask,
            )

        self._nodule_mask = full_mask.astype(np.uint8)

        return self._nodule_mask

    @staticmethod
    def _clip_consensus_region(
        *,
        volume_shape: tuple[int, int, int],
        z_slice: slice,
        y_slice: slice,
        x_slice: slice,
    ) -> tuple[
        tuple[slice, slice, slice],
        tuple[slice, slice, slice],
    ]:
        """
        Clip a consensus bounding box to the CT volume.

        Returns both:
        - slices for the destination CT volume
        - corresponding slices for the consensus mask
        """

        source_slices = (
            z_slice,
            y_slice,
            x_slice,
        )

        target_slices = []
        mask_slices = []

        for source_slice, dimension_size in zip(
            source_slices,
            volume_shape,
            strict=True,
        ):
            start = 0 if source_slice.start is None else source_slice.start

            stop = dimension_size if source_slice.stop is None else source_slice.stop

            clipped_start = max(
                start,
                0,
            )

            clipped_stop = min(
                stop,
                dimension_size,
            )

            if clipped_start >= clipped_stop:
                return (
                    (
                        slice(0, 0),
                        slice(0, 0),
                        slice(0, 0),
                    ),
                    (
                        slice(0, 0),
                        slice(0, 0),
                        slice(0, 0),
                    ),
                )

            mask_start = clipped_start - start

            mask_stop = mask_start + clipped_stop - clipped_start

            target_slices.append(
                slice(
                    clipped_start,
                    clipped_stop,
                )
            )

            mask_slices.append(
                slice(
                    mask_start,
                    mask_stop,
                )
            )

        return (
            tuple(target_slices),
            tuple(mask_slices),
        )

    def lung_mask(
        self,
    ):
        """
        Return the binary lung mask.

        Computed only once and cached.
        """

        if self._lung_mask is not None:
            return self._lung_mask

        self._lung_mask = self._lung_segmenter.segment_volume(self.hu_volume())

        return self._lung_mask

    def __repr__(
        self,
    ):
        return (
            "LIDCProcessor\n"
            "-------------------------\n"
            f"Patient ID : {self.patient_id}\n"
            f"CT Shape   : "
            f"{self.ct.shape() if self.ct else None}\n"
            f"Annotation clusters: "
            f"{len(self.annotation_clusters)}\n"
            f"Consensus level: "
            f"{self.consensus_level}\n"
            f"HU Cached  : "
            f"{self._hu_volume is not None}\n"
            f"Lung Cached: "
            f"{self._lung_mask is not None}\n"
            f"Nodule Cached: "
            f"{self._nodule_mask is not None}"
        )

    def clear_cache(
        self,
    ):
        """
        Clear all cached medical data.
        """

        self._hu_volume = None
        self._nodule_mask = None
        self._lung_mask = None
