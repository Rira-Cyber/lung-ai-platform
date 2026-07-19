from torch.utils.data import DataLoader

from backend.datasets.lidc_dataset import LIDCDataset
from backend.preprocessing.patch_sampler import PatchSampler


def create_train_loader(
    dataset_path,
    batch_size=2,
    patch_size=(128, 128, 128),
    positive_ratio=0.7,
    offset=(12, 12, 6),
    transforms=None,
    patches_per_patient=20,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
):
    """
    Create DataLoader for training.
    """

    sampler = PatchSampler(
        patch_size=patch_size,
        positive_ratio=positive_ratio,
        offset=offset,
    )

    dataset = LIDCDataset(
        dataset_path=dataset_path,
        sampler=sampler,
        transforms=transforms,
        patches_per_patient=patches_per_patient,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def create_val_loader(
    dataset_path,
    batch_size=2,
    patch_size=(128, 128, 128),
    positive_ratio=0.5,
    offset=(0, 0, 0),
    transforms=None,
    patches_per_patient=10,
    num_workers=0,
    pin_memory=True,
):
    """
    Create DataLoader for validation.
    """

    sampler = PatchSampler(
        patch_size=patch_size,
        positive_ratio=positive_ratio,
        offset=offset,
    )

    dataset = LIDCDataset(
        dataset_path=dataset_path,
        sampler=sampler,
        transforms=transforms,
        patches_per_patient=patches_per_patient,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def create_test_loader(
    dataset_path,
    batch_size=1,
    patch_size=(128, 128, 128),
    positive_ratio=0.5,
    offset=(0, 0, 0),
    transforms=None,
    patches_per_patient=10,
    num_workers=0,
    pin_memory=True,
):
    """
    Create DataLoader for testing.
    """

    sampler = PatchSampler(
        patch_size=patch_size,
        positive_ratio=positive_ratio,
        offset=offset,
    )

    dataset = LIDCDataset(
        dataset_path=dataset_path,
        sampler=sampler,
        transforms=transforms,
        patches_per_patient=patches_per_patient,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )