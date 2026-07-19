from __future__ import annotations

import random
from typing import Tuple

import numpy as np


class PatchSampler:
    """
    Generate balanced 3D patches for lung nodule segmentation.

    Responsibilities
    ----------------
    - Sample positive patches around nodules.
    - Sample negative patches inside lung regions.
    - Extract fixed-size 3D patches.
    - Handle boundary conditions.

    This class is model-agnostic and dataset-agnostic.
    """

    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        positive_ratio: float = 0.7,
        offset: Tuple[int, int, int] = (12, 12, 6),
        padding_mode: str = "edge",
        random_seed: int | None = None,
        patches_per_volume: int = 1,
    ):

        
        self.patch_size = np.asarray(patch_size, dtype=np.int32)
        
        self.patches_per_volume = patches_per_volume
        
        self.positive_ratio = positive_ratio

        self.offset = np.asarray(offset, dtype=np.int32)

        self.padding_mode = padding_mode

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)


    def _sample_positive_center(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Sample a random center inside the nodule mask.
        """

        positive_voxels = np.argwhere(mask > 0)

        if len(positive_voxels) == 0:
            raise RuntimeError(
                "No positive voxels found in the mask."
            )

        center = positive_voxels[
            np.random.randint(len(positive_voxels))
        ]

        return center.astype(np.int32)
    

    def _apply_random_offset(
        self,
        center: np.ndarray,
    ) -> np.ndarray:
        """
        Randomly shift the patch center.
        """

        random_offset = np.array([
            np.random.randint(
                -self.offset[0],
                self.offset[0] + 1,
            ),
            np.random.randint(
                -self.offset[1],
                self.offset[1] + 1,
            ),
            np.random.randint(
                -self.offset[2],
                self.offset[2] + 1,
            ),
        ])

        return center + random_offset
    
    def _sample_negative_center(
        self,
        lung_mask: np.ndarray,
        nodule_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Sample a random center inside the lung,
        excluding nodules.
        """

        candidates = np.argwhere(

            (lung_mask > 0)

            &

            (nodule_mask == 0)

        )

        if len(candidates) == 0:

            raise RuntimeError(

                "No negative voxel available."

            )

        center = candidates[

            np.random.randint(

                len(candidates)

            )

        ]

        return center.astype(np.int32)
    
    def _extract_patch(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        center: np.ndarray,
    ):
        """
        Extract a patch around the given center.

        Padding is NOT handled here.
        """

        half = self.patch_size // 2

        start = center - half

        stop = start + self.patch_size

        image_patch = image[
            max(start[0], 0):min(stop[0], image.shape[0]),
            max(start[1], 0):min(stop[1], image.shape[1]),
            max(start[2], 0):min(stop[2], image.shape[2]),
        ]

        mask_patch = mask[
            max(start[0], 0):min(stop[0], mask.shape[0]),
            max(start[1], 0):min(stop[1], mask.shape[1]),
            max(start[2], 0):min(stop[2], mask.shape[2]),
        ]

        return (
            image_patch,
            mask_patch,
            start,
            stop,
        )
    

    def _pad_if_needed(
        self,
        image_patch: np.ndarray,
        mask_patch: np.ndarray,
        start: np.ndarray,
        stop: np.ndarray,
        image_shape,
    ):
        """
        Apply padding if the extracted patch
        extends outside the image.
        """

        before = np.maximum(-start, 0)

        after = np.maximum(
            stop - np.array(image_shape),
            0,
        )

        pad_width = tuple(
            (
                int(before[i]),
                int(after[i]),
            )
            for i in range(3)
        )

        image_patch = np.pad(
            image_patch,
            pad_width,
            mode=self.padding_mode,
        )

        mask_patch = np.pad(
            mask_patch,
            pad_width,
            mode="constant",
            constant_values=0,
        )

        return image_patch, mask_patch
    
    def sample(
        self,
        image: np.ndarray,
        nodule_mask: np.ndarray,
        lung_mask: np.ndarray,

    ):
        """
        Sample one training patch.

        Returns
        -------
        dict
            {
                image,
                mask,
                center,
                patch_bbox,
                is_positive
            }
        """

        is_positive = (
            np.random.rand() < self.positive_ratio
        )

        if is_positive:

            center = self._sample_positive_center(
                nodule_mask
            )

        else:

            center = self._sample_negative_center(
                lung_mask,
                nodule_mask,
            )

        center = self._apply_random_offset(
            center
        )

        image_patch, mask_patch, start, stop = (
            self._extract_patch(
                image=image,
                mask=nodule_mask,
                center=center,
            )
        )

        image_patch, mask_patch = (
            self._pad_if_needed(
                image_patch=image_patch,
                mask_patch=mask_patch,
                start=start,
                stop=stop,
                image_shape=image.shape,
            )
        )

        return {

            "image": image_patch,

            "mask": mask_patch,

            "center": center,

            "patch_bbox": (

                start,

                stop,

            ),

            "is_positive": is_positive,

        }
    
    def sample_batch(
        self,
        image: np.ndarray,
        nodule_mask: np.ndarray,
        lung_mask: np.ndarray,
    ):
        """
        Sample multiple patches from one volume.
        """

        return [

            self.sample(
                image=image,
                mask=nodule_mask,
                lung_mask=lung_mask,
            )

            for _ in range(
                self.patches_per_volume
            )

        ]