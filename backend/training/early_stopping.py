from __future__ import annotations

from collections.abc import Mapping
from typing import Literal


class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    The component is independent of Trainer, model, optimizer,
    logging, and checkpoint persistence.
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
        """
        Update early-stopping state from the current epoch metrics.

        Returns True when training should stop.
        """

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

    def state_dict(
        self,
    ) -> dict:
        """
        Return resumable early-stopping state.
        """

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
        """
        Restore runtime state from a checkpoint.
        """

        if not isinstance(
            state_dict,
            dict,
        ):
            raise TypeError("Early stopping state_dict must be a dictionary.")

        self._validate_compatible_state(state_dict)

        best_score = state_dict.get("best_score")

        self.best_score = None if best_score is None else float(best_score)

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

        if self.bad_epochs < 0:
            raise ValueError("bad_epochs cannot be negative.")

    def reset(
        self,
    ) -> None:
        """
        Reset runtime state while preserving configuration.
        """

        self.best_score = None
        self.bad_epochs = 0
        self.should_stop = False

    def _is_improvement(
        self,
        score: float,
    ) -> bool:
        if self.best_score is None:
            return True

        if self.mode == "min":
            return score < self.best_score - self.min_delta

        return score > self.best_score + self.min_delta

    def _validate_compatible_state(
        self,
        state_dict: dict,
    ) -> None:
        """
        Prevent loading state created with incompatible settings.
        """

        expected_configuration = {
            "monitor": self.monitor,
            "patience": self.patience,
            "mode": self.mode,
            "min_delta": self.min_delta,
        }

        for key, expected_value in expected_configuration.items():
            if key not in state_dict:
                continue

            saved_value = state_dict[key]

            if saved_value != expected_value:
                raise ValueError(
                    "Early stopping state is incompatible "
                    f"for '{key}': expected "
                    f"{expected_value!r}, got "
                    f"{saved_value!r}."
                )
