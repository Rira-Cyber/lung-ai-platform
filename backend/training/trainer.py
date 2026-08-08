from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from tqdm import tqdm

from backend.loggers.base_logger import BaseLogger
from backend.loggers.console_logger import ConsoleLogger
from backend.training.checkpoint import CheckpointManager
from backend.training.early_stopping import EarlyStopping
from backend.training.scheduler import SchedulerController


class Trainer:
    """
    Orchestrate model training, validation, checkpointing,
    scheduling, early stopping, and metric logging.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        metric: nn.Module,
        checkpoint_dir: str | Path = "checkpoints",
        device: str | torch.device = "cpu",
        logger: BaseLogger | None = None,
        scheduler: SchedulerController | None = None,
        early_stopping: EarlyStopping | None = None,
        best_metric: str = "val_dice",
        best_mode: Literal["min", "max"] = "max",
        max_grad_norm: float | None = 1.0,
    ) -> None:
        if best_mode not in {"min", "max"}:
            raise ValueError("best_mode must be either 'min' or 'max'.")

        if not best_metric:
            raise ValueError("best_metric cannot be empty.")

        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be greater than zero or None.")

        self.device = torch.device(device)

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric = metric

        self.checkpoint = CheckpointManager(
            save_dir=checkpoint_dir,
        )

        self.logger = logger or ConsoleLogger()

        self.scheduler = scheduler
        self.early_stopping = early_stopping

        self.best_metric = best_metric
        self.best_mode = best_mode
        self.best_score: float | None = None

        self.max_grad_norm = max_grad_norm

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_dice": [],
            "val_loss": [],
            "val_dice": [],
            "learning_rate": [],
        }

    ####################################################################
    # Train Step
    ####################################################################

    def train_step(
        self,
        batch: dict,
    ) -> tuple[float, float]:
        image = batch["image"].to(self.device)

        mask = batch["mask"].to(self.device)

        self.optimizer.zero_grad()

        logits = self.model(image)

        loss = self.criterion(
            logits,
            mask,
        )

        loss.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.max_grad_norm,
            )

        self.optimizer.step()

        dice = self.metric(
            logits,
            mask,
        )

        return (
            float(loss.item()),
            float(dice.item()),
        )

    ####################################################################
    # Validation Step
    ####################################################################

    def validation_step(
        self,
        batch: dict,
    ) -> tuple[float, float]:
        images = batch["image"].to(self.device)

        masks = batch["mask"].to(self.device)

        with torch.inference_mode():
            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                masks,
            )

            dice = self.metric(
                outputs,
                masks,
            )

        return (
            float(loss.item()),
            float(dice.item()),
        )

    ####################################################################
    # Train Epoch
    ####################################################################

    def train_epoch(
        self,
        train_loader,
    ) -> tuple[float, float]:
        if len(train_loader) == 0:
            raise ValueError("train_loader cannot be empty.")

        self.model.train()

        epoch_loss = 0.0
        epoch_dice = 0.0

        progress = tqdm(
            train_loader,
            desc="Training",
            leave=False,
        )

        for batch in progress:
            loss, dice = self.train_step(batch)

            epoch_loss += loss
            epoch_dice += dice

            progress.set_postfix(
                loss=f"{loss:.4f}",
                dice=f"{dice:.4f}",
            )

        return (
            epoch_loss / len(train_loader),
            epoch_dice / len(train_loader),
        )

    ####################################################################
    # Validation Epoch
    ####################################################################

    def validation_epoch(
        self,
        val_loader,
    ) -> tuple[float, float]:
        if len(val_loader) == 0:
            raise ValueError("val_loader cannot be empty.")

        self.model.eval()

        epoch_loss = 0.0
        epoch_dice = 0.0

        progress = tqdm(
            val_loader,
            desc="Validation",
            leave=False,
        )

        for batch in progress:
            loss, dice = self.validation_step(batch)

            epoch_loss += loss
            epoch_dice += dice

            progress.set_postfix(
                loss=f"{loss:.4f}",
                dice=f"{dice:.4f}",
            )

        return (
            epoch_loss / len(val_loader),
            epoch_dice / len(val_loader),
        )

    ####################################################################
    # Fit
    ####################################################################

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 1,
        resume_from: str | Path | None = None,
    ) -> None:
        """
        Train the model for the requested total number of epochs.

        When resuming, ``epochs`` still represents the final target
        epoch, not the number of additional epochs.
        """

        if epochs <= 0:
            raise ValueError("epochs must be greater than zero.")

        self._validate_training_controls(val_loader=val_loader)

        start_epoch = 0

        if resume_from is not None:
            start_epoch = self._restore_checkpoint(checkpoint_path=resume_from)

        if start_epoch >= epochs:
            raise ValueError(
                "Checkpoint epoch is greater than or equal to "
                "the requested total number of epochs."
            )

        try:
            for epoch_index in range(
                start_epoch,
                epochs,
            ):
                metrics = self._run_epoch(
                    train_loader=train_loader,
                    val_loader=val_loader,
                )

                is_best = self._update_best_score(metrics=metrics)

                if self.scheduler is not None:
                    self.scheduler.step(metrics=metrics)

                metrics["learning_rate"] = self._get_learning_rate()

                should_stop = False

                if self.early_stopping is not None:
                    should_stop = self.early_stopping.step(metrics=metrics)

                self._update_history(metrics=metrics)

                checkpoint_payload = self._build_checkpoint_payload(
                    epoch=epoch_index + 1,
                    metrics=metrics,
                )

                self.checkpoint.save(
                    checkpoint=checkpoint_payload,
                    is_best=is_best,
                )

                self.logger.log_epoch(
                    epoch=epoch_index + 1,
                    epochs=epochs,
                    metrics=metrics,
                )

                if should_stop:
                    break
        finally:
            self.close()

    ####################################################################
    # Epoch Orchestration
    ####################################################################

    def _run_epoch(
        self,
        train_loader,
        val_loader,
    ) -> dict[str, float]:
        train_loss, train_dice = self.train_epoch(train_loader)

        metrics = {
            "train_loss": train_loss,
            "train_dice": train_dice,
        }

        if val_loader is not None:
            val_loss, val_dice = self.validation_epoch(val_loader)

            metrics.update(
                {
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                }
            )

        return metrics

    ####################################################################
    # Best Model Selection
    ####################################################################

    def _update_best_score(
        self,
        metrics: dict[str, float],
    ) -> bool:
        if self.best_metric not in metrics:
            return False

        current_score = float(metrics[self.best_metric])

        if self.best_score is None:
            self.best_score = current_score
            return True

        if self.best_mode == "max":
            is_best = current_score > self.best_score
        else:
            is_best = current_score < self.best_score

        if is_best:
            self.best_score = current_score

        return is_best

    ####################################################################
    # Learning Rate
    ####################################################################

    def _get_learning_rate(
        self,
    ) -> float:
        if self.scheduler is not None:
            return self.scheduler.get_learning_rate()

        if not self.optimizer.param_groups:
            raise RuntimeError("Optimizer has no parameter groups.")

        first_group = self.optimizer.param_groups[0]

        if "lr" not in first_group:
            raise KeyError("Optimizer parameter group does not contain 'lr'.")

        return float(first_group["lr"])

    ####################################################################
    # History
    ####################################################################

    def _update_history(
        self,
        metrics: dict[str, float],
    ) -> None:
        for name, value in metrics.items():
            if name not in self.history:
                self.history[name] = []

            self.history[name].append(float(value))

    ####################################################################
    # Checkpoint Build / Restore
    ####################################################################

    def _build_checkpoint_payload(
        self,
        epoch: int,
        metrics: dict[str, float],
    ) -> dict:
        return {
            "epoch": epoch,
            "metrics": dict(metrics),
            "model_state_dict": (self.model.state_dict()),
            "optimizer_state_dict": (self.optimizer.state_dict()),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "early_stopping_state": (
                self.early_stopping.state_dict()
                if self.early_stopping is not None
                else None
            ),
            "best_metric": self.best_metric,
            "best_mode": self.best_mode,
            "best_score": self.best_score,
            "history": {name: values.copy() for name, values in self.history.items()},
        }

    def _restore_checkpoint(
        self,
        checkpoint_path: str | Path,
    ) -> int:
        checkpoint = self.checkpoint.load(
            checkpoint_path=checkpoint_path,
            map_location=self.device,
        )

        self._validate_checkpoint_payload(checkpoint)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scheduler_state = checkpoint.get("scheduler_state_dict")

        if scheduler_state is not None:
            if self.scheduler is None:
                raise ValueError(
                    "Checkpoint contains scheduler state, but Trainer has no scheduler."
                )

            self.scheduler.load_state_dict(scheduler_state)

        early_stopping_state = checkpoint.get("early_stopping_state")

        if early_stopping_state is not None:
            if self.early_stopping is None:
                raise ValueError(
                    "Checkpoint contains early-stopping state, "
                    "but Trainer has no EarlyStopping component."
                )

            self.early_stopping.load_state_dict(early_stopping_state)

        saved_best_metric = checkpoint.get(
            "best_metric",
            self.best_metric,
        )

        saved_best_mode = checkpoint.get(
            "best_mode",
            self.best_mode,
        )

        if saved_best_metric != self.best_metric:
            raise ValueError(
                "Checkpoint best_metric is incompatible "
                "with the current Trainer configuration."
            )

        if saved_best_mode != self.best_mode:
            raise ValueError(
                "Checkpoint best_mode is incompatible "
                "with the current Trainer configuration."
            )

        best_score = checkpoint.get("best_score")

        self.best_score = None if best_score is None else float(best_score)

        saved_history = checkpoint.get("history")

        if saved_history is not None:
            if not isinstance(
                saved_history,
                dict,
            ):
                raise ValueError("Checkpoint history must be a dictionary.")

            self.history = {
                name: [float(value) for value in values]
                for name, values in saved_history.items()
            }

        return int(checkpoint["epoch"])

    @staticmethod
    def _validate_checkpoint_payload(
        checkpoint: dict,
    ) -> None:
        required_fields = {
            "epoch",
            "model_state_dict",
            "optimizer_state_dict",
        }

        missing_fields = required_fields - checkpoint.keys()

        if missing_fields:
            raise ValueError(
                "Checkpoint is missing required fields: "
                + ", ".join(sorted(missing_fields))
            )

        if int(checkpoint["epoch"]) < 0:
            raise ValueError("Checkpoint epoch cannot be negative.")

    ####################################################################
    # Training Control Validation
    ####################################################################

    def _validate_training_controls(
        self,
        val_loader,
    ) -> None:
        validation_metrics = {
            "val_loss",
            "val_dice",
        }

        if val_loader is None and self.best_metric in validation_metrics:
            raise ValueError("best_metric requires a validation DataLoader.")

        if (
            val_loader is None
            and self.scheduler is not None
            and self.scheduler.monitor in validation_metrics
        ):
            raise ValueError("Scheduler monitor requires a validation DataLoader.")

        if (
            val_loader is None
            and self.early_stopping is not None
            and self.early_stopping.monitor in validation_metrics
        ):
            raise ValueError("Early-stopping monitor requires a validation DataLoader.")

    ####################################################################
    # Lifecycle
    ####################################################################

    def close(
        self,
    ) -> None:
        self.logger.close()
