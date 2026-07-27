from pathlib import Path
import json
import hashlib

import torch
from torch.utils.data import Dataset

from backend.medical.lidc_processor import LIDCProcessor


class LIDCDataset(Dataset):

    def __init__(
        self,
        dataset_path,
        sampler,
        transforms=None,
        patches_per_patient=20,
    ):

        self.dataset_path = Path(dataset_path)

        self.processor = LIDCProcessor(self.dataset_path)

        self.sampler = sampler

        self.transforms = transforms

        self.patches_per_patient = patches_per_patient

        self.cache_file = (
            self.dataset_path / "dataset_cache.json"
        )

        self.patient_ids = self._load_patient_list()

    def _dataset_fingerprint(self):

        patient_names = sorted(
            folder.name
            for folder in self.dataset_path.iterdir()
            if folder.is_dir()
        )

        fingerprint = hashlib.md5(
            "".join(patient_names).encode()
        ).hexdigest()

        return fingerprint

    def _load_patient_list(self):

        fingerprint = self._dataset_fingerprint()

        if self.cache_file.exists():

            with open(
                self.cache_file,
                "r",
            ) as f:

                cache = json.load(f)

            if cache["fingerprint"] == fingerprint:

                print("Loading dataset cache...")

                return cache["valid_patients"]

        patient_ids = sorted(
            folder.name
            for folder in self.dataset_path.iterdir()
            if folder.is_dir()
        )

        valid_patients = []

        print("Checking dataset integrity...")

        for patient_id in patient_ids:

            try:

                self.processor.load_patient(patient_id)

                self.processor.hu_volume()

                self.processor.lung_mask()

                self.processor.nodule_mask()

                valid_patients.append(patient_id)

            except Exception as e:

                print(
                    f"Skipping {patient_id}: {e}"
                )

        cache = {

            "fingerprint": fingerprint,

            "valid_patients": valid_patients,

        }

        with open(
            self.cache_file,
            "w",
        ) as f:

            json.dump(
                cache,
                f,
                indent=4,
            )

        print(
            f"Valid patients: {len(valid_patients)}"
        )

        return valid_patients

    def __len__(self):

        return (
            len(self.patient_ids)
            * self.patches_per_patient
        )

    def __getitem__(self, index):

        patient_index = (
            index // self.patches_per_patient
        )

        patient_id = self.patient_ids[
            patient_index
        ]

        self.processor.load_patient(
            patient_id
        )

        image = self.processor.hu_volume()

        nodule_mask = (
            self.processor.nodule_mask()
        )

        lung_mask = (
            self.processor.lung_mask()
        )

        has_nodule = (
            nodule_mask.sum() > 0
        )

        sample = self.sampler.sample(
            image=image,
            nodule_mask=nodule_mask,
            lung_mask=lung_mask,
            force_negative=not has_nodule,
        )

        image = sample["image"]

        mask = sample["mask"]

        if self.transforms is not None:

            image, mask = self.transforms(
                image,
                mask,
            )

        image = (
            torch.from_numpy(image)
            .float()
            .unsqueeze(0)
        )

        mask = (
            torch.from_numpy(mask)
            .float()
            .unsqueeze(0)
        )

        return {

            "image": image,

            "mask": mask,

            "patient_id": patient_id,

            "center": sample["center"],

            "patch_bbox": sample[
                "patch_bbox"
            ],

            "is_positive": sample[
                "is_positive"
            ],
        }