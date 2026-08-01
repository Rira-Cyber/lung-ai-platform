from __future__ import annotations

import torch

from backend.configs import DEV_CONFIG
from backend.losses.dice import DiceLoss
from backend.metrics.dice import DiceMetric
from backend.models.unet3d import UNet3D
from backend.training.dataloader import (
    create_train_validation_loaders,
)


def build_loaders():
    print("=" * 60)
    print("1. Building train and validation DataLoaders")

    loaders = create_train_validation_loaders(
        dataset_path=DEV_CONFIG.dataset_path,
        batch_size=DEV_CONFIG.batch_size,
        num_workers=DEV_CONFIG.num_workers,
        patch_size=DEV_CONFIG.patch_size,
        positive_ratio=DEV_CONFIG.positive_ratio,
        patches_per_patient=DEV_CONFIG.patches_per_patient,
        val_ratio=DEV_CONFIG.val_ratio,
        split_seed=DEV_CONFIG.split_seed,
        pin_memory=DEV_CONFIG.pin_memory,
        shuffle=DEV_CONFIG.shuffle,
    )

    print(
        "Train patients:",
        len(loaders.patient_split.train_ids),
    )
    print(
        "Validation patients:",
        len(loaders.patient_split.val_ids),
    )
    print(
        "Validation metadata:",
        loaders.validation_metadata_path,
    )
    print("✓ DataLoaders OK")

    return loaders


def inspect_dataset(loader) -> None:
    print("=" * 60)
    print("2. Inspecting Dataset")

    sample = loader.dataset[0]

    print("Keys:", sample.keys())
    print("Image:", sample["image"].shape)
    print("Mask:", sample["mask"].shape)
    print("Mask sum:", sample["mask"].sum().item())
    print("Positive:", sample["is_positive"])
    print("Patient:", sample["patient_id"])

    assert sample["image"].shape == sample["mask"].shape

    print("✓ Dataset OK")


def load_batch(loader) -> dict:
    print("=" * 60)
    print("3. Loading one batch")

    batch = next(iter(loader))

    print("Image batch:", batch["image"].shape)
    print("Mask batch:", batch["mask"].shape)

    assert batch["image"].shape == batch["mask"].shape

    print("✓ Batch OK")

    return batch


def build_model(
    device: torch.device,
) -> UNet3D:
    print("=" * 60)
    print("4. Building Model")

    model = UNet3D(
        in_channels=DEV_CONFIG.in_channels,
        out_channels=DEV_CONFIG.out_channels,
        features=DEV_CONFIG.features,
    ).to(device)

    print("✓ Model OK")

    return model


def run_forward_check(
    model: UNet3D,
    batch: dict,
    device: torch.device,
) -> torch.Tensor:
    print("=" * 60)
    print("5. Running one inference-only forward pass")

    images = batch["image"].to(device)
    masks = batch["mask"].to(device)

    model.eval()

    with torch.inference_mode():
        predictions = model(images)

    print("Predictions:", predictions.shape)

    assert predictions.shape == masks.shape

    print("✓ Forward OK")

    return predictions


def check_loss_and_metric(
    predictions: torch.Tensor,
    batch: dict,
    device: torch.device,
) -> None:
    print("=" * 60)
    print("6. Checking loss and metric")

    masks = batch["mask"].to(device)

    criterion = DiceLoss()
    metric = DiceMetric()

    loss = criterion(
        predictions,
        masks,
    )
    dice = metric(
        predictions,
        masks,
    )

    print("Loss:", loss.item())
    print("Dice:", dice.item())

    print("✓ Loss and metric OK")


def run_pipeline_check() -> None:
    device = torch.device("cpu")

    loaders = build_loaders()

    train_ids = set(
        loaders.patient_split.train_ids
    )
    val_ids = set(
        loaders.patient_split.val_ids
    )

    assert train_ids.isdisjoint(val_ids)

    inspect_dataset(
        loaders.train_loader
    )

    batch = load_batch(
        loaders.train_loader
    )

    model = build_model(
        device=device
    )

    predictions = run_forward_check(
        model=model,
        batch=batch,
        device=device,
    )

    check_loss_and_metric(
        predictions=predictions,
        batch=batch,
        device=device,
    )

    print("=" * 60)
    print("MANUAL PIPELINE CHECK PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline_check()