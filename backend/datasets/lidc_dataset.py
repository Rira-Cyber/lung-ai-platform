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
        patches_per_patient=20,
    ):

        self.dataset_path = Path(dataset_path)

        self.processor = LIDCProcessor(self.dataset_path)

        self.sampler = sampler

        self.transforms = transforms

        self.patches_per_patient = patches_per_patient

        self.patient_ids = sorted(
            folder.name
            for folder in self.dataset_path.iterdir()
            if folder.is_dir()
        )

    def __len__(self):

        return len(self.patient_ids) * self.patches_per_patient


    def __getitem__(self, index):

        patient_index = index // self.patches_per_patient

        patient_id = self.patient_ids[patient_index]

        self.processor.load_patient(patient_id)

        image = self.processor.hu_volume()

        lung_mask = self.processor.lung_mask()

        nodule_mask = self.processor.nodule_mask()

        sample = self.sampler.sample(
            image=image,
            nodule_mask=nodule_mask,
            lung_mask=lung_mask,
        )

        image = sample["image"]
        mask = sample["mask"]

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        image = torch.from_numpy(image).float().unsqueeze(0)
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "patient_id": patient_id,
            "center": sample["center"],
            "patch_bbox": sample["patch_bbox"],
            "is_positive": sample["is_positive"],
        }