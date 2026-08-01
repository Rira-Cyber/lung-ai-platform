from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from backend.datasets.validation_metadata import ValidationMetadata
from backend.preprocessing.patch_sampler import PatchSampler


class ValidationDataset(Dataset):
    def __init__(
        self,
        dataset_path: str | Path,
        metadata: ValidationMetadata,
        transforms=None,
        processor_factory: Callable[[Path], Any] | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.metadata = metadata
        self.transforms = transforms

        if processor_factory is None:
            from backend.medical.lidc_processor import LIDCProcessor

            processor_factory = LIDCProcessor

        self.processor = processor_factory(self.dataset_path)

        self.patch_extractor = PatchSampler(
            patch_size=metadata.patch_size,
            positive_ratio=0.0,
        )

    def __len__(
        self,
    ) -> int:
        return len(self.metadata.patches)

    def __getitem__(
        self,
        index: int,
    ) -> dict:
        if not 0 <= index < len(self):
            raise IndexError("Dataset index is out of range.")

        patch_metadata = self.metadata.patches[index]

        self.processor.load_patient(patch_metadata.patient_id)

        image = self.processor.hu_volume()

        nodule_mask = self.processor.nodule_mask()

        sample = self.patch_extractor.extract_at_center(
            image=image,
            mask=nodule_mask,
            center=patch_metadata.center,
            is_positive=(patch_metadata.is_positive),
        )

        image_patch = sample["image"]

        mask_patch = sample["mask"]

        if self.transforms is not None:
            image_patch, mask_patch = self.transforms(
                image_patch,
                mask_patch,
            )

        image_tensor = torch.from_numpy(image_patch).float().unsqueeze(0)

        mask_tensor = torch.from_numpy(mask_patch).float().unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "patient_id": (patch_metadata.patient_id),
            "patch_index": (patch_metadata.patch_index),
            "center": sample["center"],
            "patch_bbox": sample["patch_bbox"],
            "is_positive": (patch_metadata.is_positive),
        }
