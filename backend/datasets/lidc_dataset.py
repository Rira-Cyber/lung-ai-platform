from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
from typing import Protocol

import torch
from torch.utils.data import Dataset

from backend.preprocessing.patient_preprocessor import (
    PatientPreprocessor,
)
from backend.preprocessing.store import (
    PatientArtifact,
    PreprocessedPatientStore,
)


class PatientArtifactProvider(Protocol):
    """
    Contract required by LIDCDataset for obtaining
    preprocessed patient artifacts.
    """

    def get(
        self,
        patient_id: str,
    ) -> PatientArtifact: ...


class LIDCDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        sampler,
        transforms=None,
        patches_per_patient: int = 20,
        patient_ids: Sequence[str] | None = None,
        patient_preprocessor: PatientArtifactProvider | None = None,
        processed_store_path: str | Path | None = None,
    ) -> None:
        if patches_per_patient <= 0:
            raise ValueError("patches_per_patient must be greater than zero.")

        self.dataset_path = Path(dataset_path)

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

        self.sampler = sampler
        self.transforms = transforms
        self.patches_per_patient = patches_per_patient

        self.cache_file = self.dataset_path / "dataset_cache.json"

        self.patient_preprocessor = (
            patient_preprocessor
            if patient_preprocessor is not None
            else self._create_patient_preprocessor(processed_store_path)
        )

        requested_patient_ids = list(patient_ids) if patient_ids is not None else None

        valid_patient_ids = self._load_patient_list(
            requested_patient_ids=(requested_patient_ids)
        )

        self.patient_ids = self._resolve_patient_ids(
            requested_patient_ids=(requested_patient_ids),
            valid_patient_ids=(valid_patient_ids),
        )

    def _create_patient_preprocessor(
        self,
        processed_store_path: str | Path | None,
    ) -> PatientPreprocessor:
        if processed_store_path is None:
            processed_store_path = self.dataset_path.parent / "processed" / "patients"

        store = PreprocessedPatientStore(processed_store_path)

        return PatientPreprocessor(
            dataset_path=self.dataset_path,
            store=store,
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
        requested_patient_ids: Sequence[str] | None = None,
    ) -> list[str]:
        """
        Discover and validate available patients.

        Explicit subsets validate only requested patients.

        Full-dataset discovery may reuse the lightweight patient-list
        cache. Medical preprocessing itself is delegated to the
        PatientPreprocessor.
        """

        if requested_patient_ids is not None:
            requested = list(requested_patient_ids)

            self._validate_requested_ids(requested)

            available_folders = {
                folder.name for folder in self.dataset_path.iterdir() if folder.is_dir()
            }

            missing_folders = sorted(set(requested) - available_folders)

            if missing_folders:
                raise ValueError(
                    "Patient folders are unavailable: " + ", ".join(missing_folders)
                )

            return self._validate_patients(requested)

        fingerprint = self._dataset_fingerprint()

        cached_patients = self._load_cached_patients(fingerprint=fingerprint)

        if cached_patients is not None:
            return cached_patients

        patient_ids = sorted(
            folder.name for folder in self.dataset_path.iterdir() if folder.is_dir()
        )

        valid_patients = self._validate_patients(patient_ids)

        self._save_patient_cache(
            fingerprint=fingerprint,
            valid_patients=valid_patients,
        )

        return valid_patients

    def _validate_patients(
        self,
        patient_ids: Sequence[str],
    ) -> list[str]:
        """
        Validate supplied patients through the preprocessing layer.

        Existing valid artifacts become cache hits.
        Missing or stale artifacts are built incrementally.
        """

        valid_patients: list[str] = []

        for patient_id in patient_ids:
            try:
                self.patient_preprocessor.get(patient_id)

                valid_patients.append(patient_id)

            except Exception as error:
                print(f"Skipping {patient_id}: {error}")

        return valid_patients

    @staticmethod
    def _validate_requested_ids(
        patient_ids: Sequence[str],
    ) -> None:
        if not patient_ids:
            raise ValueError("patient_ids cannot be empty.")

        if len(patient_ids) != len(set(patient_ids)):
            raise ValueError("patient_ids contains duplicate values.")

        if not all(
            isinstance(
                patient_id,
                str,
            )
            and patient_id
            for patient_id in patient_ids
        ):
            raise ValueError("Every patient ID must be a non-empty string.")

    def _load_cached_patients(
        self,
        fingerprint: str,
    ) -> list[str] | None:
        if not self.cache_file.exists():
            return None

        try:
            with self.cache_file.open(
                "r",
                encoding="utf-8",
            ) as file:
                cache = json.load(file)
        except (
            json.JSONDecodeError,
            OSError,
        ):
            return None

        if cache.get("fingerprint") != fingerprint:
            return None

        valid_patients = cache.get("valid_patients")

        if not isinstance(
            valid_patients,
            list,
        ):
            return None

        if not all(
            isinstance(
                patient_id,
                str,
            )
            for patient_id in valid_patients
        ):
            return None

        return list(valid_patients)

    def _save_patient_cache(
        self,
        fingerprint: str,
        valid_patients: Sequence[str],
    ) -> None:
        cache = {
            "fingerprint": fingerprint,
            "valid_patients": list(valid_patients),
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

    @staticmethod
    def _resolve_patient_ids(
        requested_patient_ids: Sequence[str] | None,
        valid_patient_ids: Sequence[str],
    ) -> list[str]:
        if requested_patient_ids is None:
            return list(valid_patient_ids)

        requested = list(requested_patient_ids)

        valid_set = set(valid_patient_ids)

        invalid_ids = sorted(set(requested) - valid_set)

        if invalid_ids:
            raise ValueError(
                "The following patient IDs "
                "are invalid or unavailable: " + ", ".join(invalid_ids)
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

        artifact = self.patient_preprocessor.get(patient_id)

        image = artifact.image
        lung_mask = artifact.lung_mask
        nodule_mask = artifact.nodule_mask

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
            (
                image_patch,
                mask_patch,
            ) = self.transforms(
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
