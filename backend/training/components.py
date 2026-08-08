from __future__ import annotations

import torch

from backend.configs.config import Config
from backend.loggers.composite_logger import CompositeLogger
from backend.loggers.console_logger import ConsoleLogger
from backend.loggers.csv_logger import CSVLogger
from backend.loggers.tensorboard_logger import TensorBoardLogger
from backend.training.early_stopping import EarlyStopping
from backend.training.experiment import ExperimentPaths
from backend.training.scheduler import SchedulerController


def create_optimizer(
    model: torch.nn.Module,
    config: Config,
) -> torch.optim.Optimizer:
    """
    Create the default optimizer for segmentation training.
    """

    return torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Config,
) -> SchedulerController | None:
    """
    Create an optional scheduler controller from configuration.
    """

    if not config.scheduler_enabled:
        return None

    if config.scheduler_type == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr,
        )

        return SchedulerController(
            scheduler=scheduler,
            monitor=config.scheduler_monitor,
        )

    if config.scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma,
        )

        return SchedulerController(
            scheduler=scheduler,
        )

    raise ValueError(
        f"Unsupported scheduler type: {config.scheduler_type}"
    )


def create_early_stopping(
    config: Config,
) -> EarlyStopping | None:
    """
    Create optional early stopping from configuration.
    """

    if not config.early_stopping_enabled:
        return None

    return EarlyStopping(
        monitor=config.early_stopping_monitor,
        patience=config.early_stopping_patience,
        mode=config.early_stopping_mode,
        min_delta=config.early_stopping_min_delta,
    )


def create_logger(
    config: Config,
    experiment_paths: ExperimentPaths,
) -> CompositeLogger:
    """
    Build the configured logging pipeline for one isolated experiment.
    """

    loggers = [
        ConsoleLogger(),
    ]

    if config.csv_logging_enabled:
        loggers.append(
            CSVLogger(
                save_dir=experiment_paths.log_dir,
                filename="training.csv",
            )
        )

    if config.tensorboard_enabled:
        loggers.append(
            TensorBoardLogger(
                log_dir=experiment_paths.tensorboard_dir,
            )
        )

    return CompositeLogger(
        loggers=loggers
    )