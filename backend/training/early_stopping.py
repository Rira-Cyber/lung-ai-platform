from __future__ import annotations

from collections.abc import Mapping
from typing import Literal


class EarlyStopping:
    """
    Stop training after a monitored metric stops improving.
    """

    def __init__(
        self,
        monitor: str,
        patience: int,
        mode: Literal["min", "max"] = "min",
        min_delta: float = 0.0,
    ) -> None:
        if not monitor:
            raise ValueError("monitor cannot be empty.")

        if patience <= 0:
            raise ValueError("patience must be greater than zero.")

        if mode not in {"min", "max"}:
            raise ValueError("mode must be either 'min' or 'max'.")

        if min_delta < 0:
            raise ValueError("min_delta cannot be negative.")

        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.best_score: float | None = None
        self.bad_epochs = 0
        self.should_stop = False

    def step(
        self,
        metrics: Mapping[str, float],
    ) -> bool:
        if self.monitor not in metrics:
            raise KeyError(
                f"Early stopping monitor metric is unavailable: {self.monitor}"
            )

        current_score = float(metrics[self.monitor])

        if self._is_improvement(current_score):
            self.best_score = current_score
            self.bad_epochs = 0
            self.should_stop = False
        else:
            self.bad_epochs += 1
            self.should_stop = self.bad_epochs >= self.patience

        return self.should_stop

    def state_dict(self) -> dict:
        return {
            "monitor": self.monitor,
            "patience": self.patience,
            "mode": self.mode,
            "min_delta": self.min_delta,
            "best_score": self.best_score,
            "bad_epochs": self.bad_epochs,
            "should_stop": self.should_stop,
        }

    def load_state_dict(
        self,
        state_dict: dict,
    ) -> None:
        self.best_score = state_dict.get("best_score")

        self.bad_epochs = int(
            state_dict.get(
                "bad_epochs",
                0,
            )
        )

        self.should_stop = bool(
            state_dict.get(
                "should_stop",
                False,
            )
        )

    def _is_improvement(
        self,
        score: float,
    ) -> bool:
        if self.best_score is None:
            return True

        if self.mode == "min":
            return score < self.best_score - self.min_delta

        return score > self.best_score + self.min_delta
