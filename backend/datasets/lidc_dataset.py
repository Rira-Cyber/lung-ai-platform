from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from backend.medical.lidc_processor import LIDCProcessor


class LIDCDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        sampler,
        transforms=None,
        patches_per_patient: int = 20,
        patient_ids: Sequence[str] | None = None,
    ) -> None:
        if patches_per_patient <= 0:
            raise ValueError("patches_per_patient must be greater than zero.")

        self.dataset_path = Path(dataset_path)

        self.processor = LIDCProcessor(self.dataset_path)

        self.sampler = sampler

        self.transforms = transforms

        self.patches_per_patient = patches_per_patient

        self.cache_file = self.dataset_path / "dataset_cache.json"

        valid_patient_ids = self._load_patient_list()

        self.patient_ids = self._resolve_patient_ids(
            requested_patient_ids=patient_ids,
            valid_patient_ids=valid_patient_ids,
        )

    def _dataset_fingerprint(
        self,
    ) -> str:
        patient_names = sorted(
            folder.name for folder in self.dataset_path.iterdir() if folder.is_dir()
        )

        return hashlib.md5("".join(patient_names).encode("utf-8")).hexdigest()

    def _load_patient_list(
        self,
    ) -> list[str]:
        fingerprint = self._dataset_fingerprint()

        if self.cache_file.exists():
            with self.cache_file.open(
                "r",
                encoding="utf-8",
            ) as file:
                cache = json.load(file)

            if cache.get("fingerprint") == fingerprint:
                return list(cache["valid_patients"])

        patient_ids = sorted(
            folder.name for folder in self.dataset_path.iterdir() if folder.is_dir()
        )

        valid_patients = []

        for patient_id in patient_ids:
            try:
                self.processor.load_patient(patient_id)

                self.processor.hu_volume()

                self.processor.lung_mask()

                self.processor.nodule_mask()

                valid_patients.append(patient_id)

            except Exception as error:
                print(f"Skipping {patient_id}: {error}")

        cache = {
            "fingerprint": fingerprint,
            "valid_patients": valid_patients,
        }

        with self.cache_file.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                cache,
                file,
                indent=4,
            )

        return valid_patients

    @staticmethod
    def _resolve_patient_ids(
        requested_patient_ids: Sequence[str] | None,
        valid_patient_ids: Sequence[str],
    ) -> list[str]:
        if requested_patient_ids is None:
            return list(valid_patient_ids)

        requested = list(requested_patient_ids)

        if not requested:
            raise ValueError("patient_ids cannot be empty.")

        if len(requested) != len(set(requested)):
            raise ValueError("patient_ids contains duplicate values.")

        valid_set = set(valid_patient_ids)

        unknown_ids = sorted(set(requested) - valid_set)

        if unknown_ids:
            raise ValueError(
                "The following patient IDs "
                "are invalid or unavailable: " + ", ".join(unknown_ids)
            )

        return requested

    def __len__(
        self,
    ) -> int:
        return len(self.patient_ids) * self.patches_per_patient

    def __getitem__(
        self,
        index: int,
    ) -> dict:
        if not 0 <= index < len(self):
            raise IndexError("Dataset index is out of range.")

        patient_index = index // self.patches_per_patient

        patient_id = self.patient_ids[patient_index]

        self.processor.load_patient(patient_id)

        image = self.processor.hu_volume()

        nodule_mask = self.processor.nodule_mask()

        lung_mask = self.processor.lung_mask()

        has_nodule = bool(nodule_mask.any())

        sample = self.sampler.sample(
            image=image,
            nodule_mask=nodule_mask,
            lung_mask=lung_mask,
            force_negative=not has_nodule,
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
            "patient_id": patient_id,
            "center": sample["center"],
            "patch_bbox": sample["patch_bbox"],
            "is_positive": sample["is_positive"],
        }
