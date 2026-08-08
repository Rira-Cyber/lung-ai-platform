from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from backend.configs import DEV_CONFIG, PROD_CONFIG
from backend.configs.config import Config
from backend.datasets.patient_manifest import (
    load_patient_manifest,
)
from backend.losses.bce_dice import BCEDiceLoss
from backend.metrics.dice import DiceMetric
from backend.models.unet3d import UNet3D
from backend.training.components import (
    create_early_stopping,
    create_logger,
    create_optimizer,
    create_scheduler,
)
from backend.training.dataloader import (
    create_train_validation_loaders,
)
from backend.training.experiment import (
    ExperimentPaths,
)
from backend.training.reproducibility import (
    set_global_seed,
)
from backend.training.trainer import Trainer


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the Lung AI Platform segmentation model."
        )
    )

    parser.add_argument(
        "--config",
        choices=(
            "development",
            "production",
        ),
        default="development",
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional patient subset manifest. "
            "Overrides patient_manifest_path "
            "from the selected config."
        ),
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help=(
            "Optional experiment name. "
            "Overrides experiment_name from config."
        ),
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint path used "
            "to resume training."
        ),
    )

    return parser.parse_args()


def resolve_config(
    config_name: str,
) -> Config:
    if config_name == "development":
        return DEV_CONFIG

    if config_name == "production":
        return PROD_CONFIG

    raise ValueError(
        f"Unsupported config: {config_name}"
    )


def resolve_device(
    configured_device: str,
) -> torch.device:
    if (
        configured_device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested, but no CUDA device is available."
        )

    return torch.device(
        configured_device
    )


def resolve_patient_ids(
    arguments: argparse.Namespace,
    config: Config,
) -> tuple[str, ...] | None:
    manifest_path = (
        arguments.manifest
        if arguments.manifest is not None
        else config.patient_manifest_path
    )

    if manifest_path is None:
        return None

    return load_patient_manifest(
        manifest_path
    )


def save_patient_manifest(
    patient_ids: tuple[str, ...] | None,
    experiment_paths: ExperimentPaths,
) -> None:
    if patient_ids is None:
        return

    payload = {
        "patient_ids": list(
            patient_ids
        )
    }

    with experiment_paths.manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=4,
        )


def main() -> None:
    arguments = parse_arguments()

    config = resolve_config(
        arguments.config
    )

    set_global_seed(
        config.split_seed
    )

    device = resolve_device(
        config.device
    )

    patient_ids = resolve_patient_ids(
        arguments=arguments,
        config=config,
    )

    experiment_name = (
        arguments.experiment
        if arguments.experiment is not None
        else config.experiment_name
    )

    experiment_paths = ExperimentPaths(
        root_dir=config.experiment_root,
        experiment_name=experiment_name,
    )

    experiment_paths.create()

    experiment_paths.save_config(
        config
    )

    save_patient_manifest(
        patient_ids=patient_ids,
        experiment_paths=experiment_paths,
    )

    loaders = create_train_validation_loaders(
        dataset_path=config.dataset_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        patch_size=config.patch_size,
        positive_ratio=config.positive_ratio,
        patches_per_patient=(
            config.patches_per_patient
        ),
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        split_seed=config.split_seed,
        pin_memory=config.pin_memory,
        shuffle=config.shuffle,
        patient_ids=patient_ids,
    )

    model = UNet3D(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        features=config.features,
    )

    optimizer = create_optimizer(
        model=model,
        config=config,
    )

    scheduler = create_scheduler(
        optimizer=optimizer,
        config=config,
    )

    early_stopping = create_early_stopping(
        config=config
    )

    logger = create_logger(
        config=config,
        experiment_paths=experiment_paths,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=BCEDiceLoss(
            bce_weight=0.5,
            dice_weight=0.5,
        ),
        metric=DiceMetric(),
        checkpoint_dir=experiment_paths.checkpoint_dir,
        device=device,
        logger=logger,
        scheduler=scheduler,
        early_stopping=early_stopping,
        best_metric=config.best_metric,
        best_mode=config.best_mode,
        max_grad_norm=config.max_grad_norm,
    )

    trainer.fit(
        train_loader=loaders.train_loader,
        val_loader=loaders.val_loader,
        epochs=config.epochs,
        resume_from=arguments.resume,
    )


if __name__ == "__main__":
    main()