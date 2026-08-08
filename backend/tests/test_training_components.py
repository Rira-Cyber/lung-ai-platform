from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from backend.training.experiment import ExperimentPaths
from backend.configs import DEV_CONFIG
from backend.loggers.composite_logger import (
    CompositeLogger,
)
from backend.training.components import (
    create_early_stopping,
    create_logger,
    create_optimizer,
    create_scheduler,
)


def create_model() -> torch.nn.Module:
    return torch.nn.Linear(
        in_features=2,
        out_features=1,
    )


def test_create_optimizer() -> None:
    model = create_model()

    optimizer = create_optimizer(
        model=model,
        config=DEV_CONFIG,
    )

    assert isinstance(
        optimizer,
        torch.optim.Adam,
    )

    assert optimizer.param_groups[0]["lr"] == (DEV_CONFIG.learning_rate)


def test_create_reduce_on_plateau_scheduler() -> None:
    model = create_model()

    optimizer = create_optimizer(
        model=model,
        config=DEV_CONFIG,
    )

    scheduler = create_scheduler(
        optimizer=optimizer,
        config=DEV_CONFIG,
    )

    assert scheduler is not None
    assert scheduler.is_metric_based is True
    assert scheduler.monitor == DEV_CONFIG.scheduler_monitor


def test_disabled_scheduler_returns_none() -> None:
    config = replace(
        DEV_CONFIG,
        scheduler_enabled=False,
    )

    optimizer = create_optimizer(
        model=create_model(),
        config=config,
    )

    assert (
        create_scheduler(
            optimizer=optimizer,
            config=config,
        )
        is None
    )


def test_create_early_stopping() -> None:
    early_stopping = create_early_stopping(config=DEV_CONFIG)

    assert early_stopping is not None
    assert early_stopping.monitor == DEV_CONFIG.early_stopping_monitor


def test_disabled_early_stopping_returns_none() -> None:
    config = replace(
        DEV_CONFIG,
        early_stopping_enabled=False,
    )

    assert create_early_stopping(config=config) is None


def test_create_logger(
    tmp_path: Path,
) -> None:
    config = replace(
        DEV_CONFIG,
        csv_logging_enabled=True,
        tensorboard_enabled=True,
    )

    experiment_paths = ExperimentPaths(
        root_dir=tmp_path,
        experiment_name="test_experiment",
    )

    experiment_paths.create()

    logger = create_logger(
        config=config,
        experiment_paths=experiment_paths,
    )

    assert isinstance(
        logger,
        CompositeLogger,
    )

    logger.close()