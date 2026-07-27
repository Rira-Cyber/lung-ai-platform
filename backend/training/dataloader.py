from torch.utils.data import DataLoader

from backend.datasets.lidc_dataset import LIDCDataset
from backend.preprocessing.patch_sampler import PatchSampler


def create_train_loader(
    dataset_path,
    batch_size,
    num_workers,
    patch_size,
    positive_ratio,
):
    sampler = PatchSampler(
        patch_size=patch_size,
        positive_ratio=positive_ratio,
    )

    dataset = LIDCDataset(
        dataset_path=dataset_path,
        sampler=sampler,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )


def create_val_loader(
    dataset_path,
    batch_size,
    num_workers,
    patch_size,
    positive_ratio,
):
    sampler = PatchSampler(
        patch_size=patch_size,
        positive_ratio=positive_ratio,
    )

    dataset = LIDCDataset(
        dataset_path=dataset_path,
        sampler=sampler,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )


def create_test_loader(
    dataset_path,
    batch_size,
    num_workers,
    patch_size,
    positive_ratio,
):
    sampler = PatchSampler(
        patch_size=patch_size,
        positive_ratio=positive_ratio,
    )

    dataset = LIDCDataset(
        dataset_path=dataset_path,
        sampler=sampler,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )