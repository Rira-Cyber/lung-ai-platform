from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class SchedulerController:
    """
    Adapt epoch-based and metric-based PyTorch schedulers
    to a common Trainer-facing interface.

    If monitor is None, scheduler.step() is called.
    Otherwise scheduler.step(metrics[monitor]) is called.
    """

    def __init__(
        self,
        scheduler: Any,
        monitor: str | None = None,
    ) -> None:
        if scheduler is None:
            raise ValueError("scheduler cannot be None.")

        self.scheduler = scheduler
        self.monitor = monitor

    def step(
        self,
        metrics: Mapping[str, float],
    ) -> None:
        if self.monitor is None:
            self.scheduler.step()
            return

        if self.monitor not in metrics:
            raise KeyError(f"Scheduler monitor metric is unavailable: {self.monitor}")

        self.scheduler.step(metrics[self.monitor])

    def state_dict(self) -> dict:
        return self.scheduler.state_dict()

    def load_state_dict(
        self,
        state_dict: dict,
    ) -> None:
        self.scheduler.load_state_dict(state_dict)

    def get_learning_rate(self) -> float:
        param_groups = self.scheduler.optimizer.param_groups

        if not param_groups:
            raise RuntimeError("Optimizer has no parameter groups.")

        return float(param_groups[0]["lr"])
