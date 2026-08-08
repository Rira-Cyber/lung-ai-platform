from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class SchedulerController:
    """
    Adapt PyTorch learning-rate schedulers to a common interface.

    Epoch-based schedulers:
        SchedulerController(scheduler)

    Metric-based schedulers:
        SchedulerController(
            scheduler,
            monitor="val_loss",
        )
    """

    def __init__(
        self,
        scheduler: Any,
        monitor: str | None = None,
    ) -> None:
        if scheduler is None:
            raise ValueError("scheduler cannot be None.")

        if monitor is not None and not monitor:
            raise ValueError("monitor cannot be empty.")

        if not hasattr(
            scheduler,
            "step",
        ):
            raise TypeError("scheduler must provide a step() method.")

        if not hasattr(
            scheduler,
            "state_dict",
        ):
            raise TypeError("scheduler must provide a state_dict() method.")

        if not hasattr(
            scheduler,
            "load_state_dict",
        ):
            raise TypeError("scheduler must provide a load_state_dict() method.")

        if not hasattr(
            scheduler,
            "optimizer",
        ):
            raise TypeError("scheduler must expose its optimizer.")

        self.scheduler = scheduler
        self.monitor = monitor

    @property
    def is_metric_based(
        self,
    ) -> bool:
        return self.monitor is not None

    def step(
        self,
        metrics: Mapping[str, float],
    ) -> None:
        """
        Advance the scheduler by one epoch.

        For epoch-based schedulers, metrics are ignored.

        For metric-based schedulers, the configured monitored
        metric is passed to scheduler.step().
        """

        if self.monitor is None:
            self.scheduler.step()
            return

        if self.monitor not in metrics:
            raise KeyError(f"Scheduler monitor metric is unavailable: {self.monitor}")

        metric_value = float(metrics[self.monitor])

        self.scheduler.step(metric_value)

    def get_learning_rate(
        self,
    ) -> float:
        """
        Return the learning rate of the first optimizer group.
        """

        param_groups = self.scheduler.optimizer.param_groups

        if not param_groups:
            raise RuntimeError("Optimizer has no parameter groups.")

        if "lr" not in param_groups[0]:
            raise KeyError("Optimizer parameter group does not contain 'lr'.")

        return float(param_groups[0]["lr"])

    def get_learning_rates(
        self,
    ) -> tuple[float, ...]:
        """
        Return learning rates for all optimizer parameter groups.
        """

        param_groups = self.scheduler.optimizer.param_groups

        if not param_groups:
            raise RuntimeError("Optimizer has no parameter groups.")

        return tuple(float(group["lr"]) for group in param_groups)

    def state_dict(
        self,
    ) -> dict:
        return self.scheduler.state_dict()

    def load_state_dict(
        self,
        state_dict: dict,
    ) -> None:
        if not isinstance(
            state_dict,
            dict,
        ):
            raise TypeError("Scheduler state_dict must be a dictionary.")

        self.scheduler.load_state_dict(state_dict)
