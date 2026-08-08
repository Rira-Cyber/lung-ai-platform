from __future__ import annotations

import pytest
import torch

from backend.training.scheduler import (
    SchedulerController,
)


def create_optimizer(
    learning_rate: float = 0.1,
) -> torch.optim.Optimizer:
    parameter = torch.nn.Parameter(torch.tensor(1.0))

    return torch.optim.SGD(
        [parameter],
        lr=learning_rate,
    )


def test_epoch_based_scheduler_step() -> None:
    optimizer = create_optimizer(learning_rate=0.1)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.5,
    )

    controller = SchedulerController(scheduler=scheduler)

    optimizer.step()

    controller.step(metrics={})

    assert controller.get_learning_rate() == pytest.approx(0.05)


def test_metric_based_scheduler_step() -> None:
    optimizer = create_optimizer(learning_rate=0.1)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=0,
    )

    controller = SchedulerController(
        scheduler=scheduler,
        monitor="val_loss",
    )

    controller.step(
        metrics={
            "val_loss": 1.0,
        }
    )

    controller.step(
        metrics={
            "val_loss": 1.1,
        }
    )

    assert controller.get_learning_rate() == pytest.approx(0.05)


def test_metric_based_scheduler_requires_metric() -> None:
    optimizer = create_optimizer()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    controller = SchedulerController(
        scheduler=scheduler,
        monitor="val_loss",
    )

    with pytest.raises(
        KeyError,
        match="val_loss",
    ):
        controller.step(
            metrics={
                "train_loss": 1.0,
            }
        )


def test_is_metric_based_property() -> None:
    optimizer = create_optimizer()

    epoch_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
    )

    metric_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    epoch_controller = SchedulerController(epoch_scheduler)

    metric_controller = SchedulerController(
        metric_scheduler,
        monitor="val_loss",
    )

    assert epoch_controller.is_metric_based is False
    assert metric_controller.is_metric_based is True


def test_get_all_learning_rates() -> None:
    first_parameter = torch.nn.Parameter(torch.tensor(1.0))

    second_parameter = torch.nn.Parameter(torch.tensor(2.0))

    optimizer = torch.optim.SGD(
        [
            {
                "params": [first_parameter],
                "lr": 0.1,
            },
            {
                "params": [second_parameter],
                "lr": 0.01,
            },
        ]
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
    )

    controller = SchedulerController(scheduler)

    assert controller.get_learning_rates() == pytest.approx(
        (
            0.1,
            0.01,
        )
    )


def test_state_dict_round_trip() -> None:
    first_optimizer = create_optimizer(learning_rate=0.1)

    first_scheduler = torch.optim.lr_scheduler.StepLR(
        first_optimizer,
        step_size=1,
        gamma=0.5,
    )

    first_controller = SchedulerController(first_scheduler)

    first_optimizer.step()

    first_controller.step(metrics={})

    state = first_controller.state_dict()

    second_optimizer = create_optimizer(learning_rate=0.1)

    second_scheduler = torch.optim.lr_scheduler.StepLR(
        second_optimizer,
        step_size=1,
        gamma=0.5,
    )

    second_controller = SchedulerController(second_scheduler)

    second_controller.load_state_dict(state)

    assert second_controller.state_dict() == first_controller.state_dict()


def test_none_scheduler_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be None",
    ):
        SchedulerController(scheduler=None)


def test_empty_monitor_raises_error() -> None:
    optimizer = create_optimizer()

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
    )

    with pytest.raises(
        ValueError,
        match="monitor cannot be empty",
    ):
        SchedulerController(
            scheduler=scheduler,
            monitor="",
        )


def test_invalid_state_dict_raises_error() -> None:
    optimizer = create_optimizer()

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
    )

    controller = SchedulerController(scheduler)

    with pytest.raises(
        TypeError,
        match="must be a dictionary",
    ):
        controller.load_state_dict(state_dict=[])
