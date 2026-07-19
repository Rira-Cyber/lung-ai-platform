import numpy as np

from .threshold import Threshold
from .morphology import Morphology
from .connected import ConnectedComponents


class LungSegmenter:
    """
    Classical lung segmentation for CT images.
    """

    def __init__(
        self,
        body_threshold: int = -600,
        lung_threshold: int = -320,
    ):

        self.body_threshold = body_threshold
        self.lung_threshold = lung_threshold

    # ---------------------------------------------------
    # Public API
    # ---------------------------------------------------

    def segment_slice(self, image: np.ndarray) -> np.ndarray:
        """
        Segment lungs from a single CT slice.

        Parameters
        ----------
        image : np.ndarray
            HU image.

        Returns
        -------
        np.ndarray
            Binary lung mask.
        """

        # Step 1
        body_mask = self._extract_body(image)

        # Step 2
        inside_body = self._remove_outside_air(
            image,
            body_mask
        )

        # Step 3
        lung_mask = self._extract_lungs(
            inside_body
        )

        return lung_mask

    def segment_volume(
        self,
        volume: np.ndarray
    ) -> np.ndarray:
        """
        Segment lungs for an entire CT volume.
        """

        masks = []

        for slice_image in volume:

            masks.append(
                self.segment_slice(slice_image)
            )

        return np.stack(masks)

    def apply_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:

        result = image.copy()

        result[~mask] = -1000

        return result

    # ---------------------------------------------------
    # Private Methods
    # ---------------------------------------------------

    def _extract_body(
        self,
        image: np.ndarray
    ) -> np.ndarray:

        body = Threshold.body(
            image,
            self.body_threshold
        )

        body = Morphology.close(
            body,
            kernel=7
        )

        body = Morphology.fill(body)

        body = ConnectedComponents.largest(
            body
        )

        return body

    def _remove_outside_air(
        self,
        image: np.ndarray,
        body_mask: np.ndarray
    ) -> np.ndarray:

        inside = image.copy()

        inside[~body_mask] = 1000

        return inside

    def _extract_lungs(
        self,
        image: np.ndarray
    ) -> np.ndarray:

        lungs = Threshold.lung(
            image,
            self.lung_threshold
        )

        lungs = ConnectedComponents.largest_n(
            lungs,
            n=2
        )

        lungs = Morphology.open(
            lungs,
            kernel=3
        )

        lungs = Morphology.close(
            lungs,
            kernel=5
        )

        lungs = Morphology.fill(
            lungs
        )

        return lungs