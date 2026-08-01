from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class PatchSampler:
    """
    Generate and extract fixed-size 3D patches.

    Responsibilities
    ----------------
    - Sample positive centers around nodules.
    - Sample negative centers inside lung regions.
    - Apply random offsets during dynamic training sampling.
    - Extract patches at either random or predefined centers.
    - Handle boundary padding.

    This class is model-agnostic and dataset-agnostic.
    """

    def __init__(
        self,
        patch_size: tuple[int, int, int] = (
            128,
            128,
            128,
        ),
        positive_ratio: float = 0.7,
        offset: tuple[int, int, int] = (
            12,
            12,
            6,
        ),
        padding_mode: str = "edge",
        random_seed: int | None = None,
        patches_per_volume: int = 1,
    ) -> None:
        if len(patch_size) != 3:
            raise ValueError("patch_size must contain exactly three values.")

        if any(size <= 0 for size in patch_size):
            raise ValueError("All patch_size values must be greater than zero.")

        if len(offset) != 3:
            raise ValueError("offset must contain exactly three values.")

        if any(value < 0 for value in offset):
            raise ValueError("Offset values cannot be negative.")

        if not 0.0 <= positive_ratio <= 1.0:
            raise ValueError("positive_ratio must be in the range [0.0, 1.0].")

        if patches_per_volume <= 0:
            raise ValueError("patches_per_volume must be greater than zero.")

        self.patch_size = np.asarray(
            patch_size,
            dtype=np.int32,
        )

        self.positive_ratio = positive_ratio

        self.offset = np.asarray(
            offset,
            dtype=np.int32,
        )

        self.padding_mode = padding_mode

        self.patches_per_volume = patches_per_volume

        self.rng = np.random.default_rng(random_seed)

    def _sample_positive_center(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Sample a random center inside the nodule mask.
        """

        positive_voxels = np.argwhere(mask > 0)

        if len(positive_voxels) == 0:
            raise RuntimeError("No positive voxels found in the mask.")

        selected_index = int(
            self.rng.integers(
                low=0,
                high=len(positive_voxels),
            )
        )

        return positive_voxels[selected_index].astype(np.int32)

    def _sample_negative_center(
        self,
        lung_mask: np.ndarray,
        nodule_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Sample a random center inside the lung,
        excluding nodule voxels.
        """

        candidates = np.argwhere((lung_mask > 0) & (nodule_mask == 0))

        if len(candidates) == 0:
            raise RuntimeError("No negative voxel available.")

        selected_index = int(
            self.rng.integers(
                low=0,
                high=len(candidates),
            )
        )

        return candidates[selected_index].astype(np.int32)

    def _apply_random_offset(
        self,
        center: np.ndarray,
    ) -> np.ndarray:
        """
        Randomly shift a patch center.
        """

        random_offset = self.rng.integers(
            low=-self.offset,
            high=self.offset + 1,
            size=3,
        )

        return (center + random_offset).astype(np.int32)

    @staticmethod
    def _clamp_center(
        center: np.ndarray,
        image_shape: Sequence[int],
    ) -> np.ndarray:
        """
        Clamp a patch center to valid volume coordinates.

        A center outside the volume can produce an empty slice, which
        cannot be padded with modes such as ``edge``. Clamping guarantees
        that every extracted patch intersects the source volume.
        """

        image_shape_array = np.asarray(
            image_shape,
            dtype=np.int32,
        )

        if image_shape_array.shape != (3,):
            raise ValueError("image_shape must contain exactly three values.")

        if np.any(image_shape_array <= 0):
            raise ValueError("All image dimensions must be greater than zero.")

        minimum_center = np.zeros(
            3,
            dtype=np.int32,
        )

        maximum_center = image_shape_array - 1

        return np.clip(
            center,
            minimum_center,
            maximum_center,
        ).astype(np.int32)

    def _extract_patch(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        center: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Extract a patch around the given center.

        Padding is handled separately.
        """

        half_size = self.patch_size // 2

        start = center - half_size

        stop = start + self.patch_size

        image_patch = image[
            max(start[0], 0) : min(
                stop[0],
                image.shape[0],
            ),
            max(start[1], 0) : min(
                stop[1],
                image.shape[1],
            ),
            max(start[2], 0) : min(
                stop[2],
                image.shape[2],
            ),
        ]

        mask_patch = mask[
            max(start[0], 0) : min(
                stop[0],
                mask.shape[0],
            ),
            max(start[1], 0) : min(
                stop[1],
                mask.shape[1],
            ),
            max(start[2], 0) : min(
                stop[2],
                mask.shape[2],
            ),
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
        image_shape: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Pad a patch when it extends beyond volume boundaries.
        """

        image_shape_array = np.asarray(
            image_shape,
            dtype=np.int32,
        )

        before = np.maximum(
            -start,
            0,
        )

        after = np.maximum(
            stop - image_shape_array,
            0,
        )

        pad_width = tuple(
            (
                int(before[index]),
                int(after[index]),
            )
            for index in range(3)
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

    def extract_at_center(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        center: Sequence[int] | np.ndarray,
        *,
        is_positive: bool | None = None,
    ) -> dict:
        """
        Extract a deterministic patch at a predefined center.

        Centers outside the volume are clamped to the nearest valid
        voxel so extraction always intersects the source volume.
        """

        self._validate_volume_inputs(
            image=image,
            mask=mask,
        )

        center_array = np.asarray(
            center,
            dtype=np.int32,
        )

        if center_array.shape != (3,):
            raise ValueError("center must contain exactly three coordinates.")

        center_array = self._clamp_center(
            center=center_array,
            image_shape=image.shape,
        )

        (
            image_patch,
            mask_patch,
            start,
            stop,
        ) = self._extract_patch(
            image=image,
            mask=mask,
            center=center_array,
        )

        image_patch, mask_patch = self._pad_if_needed(
            image_patch=image_patch,
            mask_patch=mask_patch,
            start=start,
            stop=stop,
            image_shape=image.shape,
        )

        expected_shape = tuple(int(value) for value in self.patch_size)

        if image_patch.shape != expected_shape:
            raise RuntimeError(
                "Extracted image patch has an unexpected shape. "
                f"Expected {expected_shape}, got {image_patch.shape}."
            )

        if mask_patch.shape != expected_shape:
            raise RuntimeError(
                "Extracted mask patch has an unexpected shape. "
                f"Expected {expected_shape}, got {mask_patch.shape}."
            )

        resolved_is_positive = (
            bool(mask_patch.any()) if is_positive is None else bool(is_positive)
        )

        return {
            "image": image_patch,
            "mask": mask_patch,
            "center": center_array,
            "patch_bbox": (
                start,
                stop,
            ),
            "is_positive": resolved_is_positive,
        }

    def sample(
        self,
        image: np.ndarray,
        nodule_mask: np.ndarray,
        lung_mask: np.ndarray,
        force_negative: bool = False,
    ) -> dict:
        """
        Sample one random training patch.
        """

        self._validate_volume_inputs(
            image=image,
            mask=nodule_mask,
        )

        if lung_mask.ndim != 3:
            raise ValueError("lung_mask must be a 3D array.")

        if lung_mask.shape != image.shape:
            raise ValueError("lung_mask and image must have identical shapes.")

        if force_negative:
            is_positive = False

            center = self._sample_negative_center(
                lung_mask=lung_mask,
                nodule_mask=nodule_mask,
            )

        else:
            is_positive = bool(self.rng.random() < self.positive_ratio)

            if is_positive:
                center = self._sample_positive_center(
                    mask=nodule_mask,
                )

            else:
                center = self._sample_negative_center(
                    lung_mask=lung_mask,
                    nodule_mask=nodule_mask,
                )

        center = self._apply_random_offset(
            center=center,
        )

        return self.extract_at_center(
            image=image,
            mask=nodule_mask,
            center=center,
            is_positive=is_positive,
        )

    def sample_batch(
        self,
        image: np.ndarray,
        nodule_mask: np.ndarray,
        lung_mask: np.ndarray,
    ) -> list[dict]:
        """
        Sample multiple random patches from one volume.
        """

        return [
            self.sample(
                image=image,
                nodule_mask=nodule_mask,
                lung_mask=lung_mask,
            )
            for _ in range(self.patches_per_volume)
        ]

    @staticmethod
    def _validate_volume_inputs(
        image: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        """
        Validate common image and mask requirements.
        """

        if image.ndim != 3:
            raise ValueError("image must be a 3D array.")

        if mask.ndim != 3:
            raise ValueError("mask must be a 3D array.")

        if image.shape != mask.shape:
            raise ValueError("image and mask must have identical shapes.")
