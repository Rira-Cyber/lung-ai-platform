from __future__ import annotations

from typing import Any

from backend.loggers.base_logger import BaseLogger


class ConsoleLogger(BaseLogger):
    def log_epoch(
        self,
        epoch: int,
        epochs: int,
        metrics: dict[str, Any],
    ) -> None:
        print("\n" + "=" * 60)
        print(f"Epoch {epoch}/{epochs}")
        print("=" * 60)

        train_metrics = {}
        val_metrics = {}
        other_metrics = {}

        for name, value in metrics.items():
            if value is None:
                continue

            if name.startswith("train_"):
                train_metrics[name.replace("train_", "")] = value

            elif name.startswith("val_"):
                val_metrics[name.replace("val_", "")] = value

            else:
                other_metrics[name] = value

        if train_metrics:
            print("\nTrain")

            for name, value in train_metrics.items():
                self._print_metric(name, value)

        if val_metrics:
            print("\nValidation")

            for name, value in val_metrics.items():
                self._print_metric(name, value)

        if other_metrics:
            print("\nOther")

            for name, value in other_metrics.items():
                self._print_metric(name, value)

    @staticmethod
    def _print_metric(
        name: str,
        value: Any,
    ) -> None:
        label = name.replace("_", " ").title()

        if isinstance(value, float):
            print(f"{label:<20}: {value:.4f}")
        else:
            print(f"{label:<20}: {value}")
