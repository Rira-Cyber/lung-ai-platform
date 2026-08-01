from __future__ import annotations

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
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        metric: nn.Module,
        checkpoint_dir: str = "checkpoints",
        device: str = "cpu",
        logger: BaseLogger | None = None,
        scheduler: SchedulerController | None = None,
        early_stopping: EarlyStopping | None = None,
        best_metric: str = "val_dice",
        best_mode: Literal["min", "max"] = "max",
    ) -> None:
        if best_mode not in {"min", "max"}:
            raise ValueError("best_mode must be either 'min' or 'max'.")

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric = metric
        self.device = device

        self.checkpoint = CheckpointManager(
            save_dir=checkpoint_dir,
        )

        self.logger = logger or ConsoleLogger()
        self.scheduler = scheduler
        self.early_stopping = early_stopping

        self.best_metric = best_metric
        self.best_mode = best_mode
        self.best_score: float | None = None

        self.history = {
            "train_loss": [],
            "train_dice": [],
            "val_loss": [],
            "val_dice": [],
            "learning_rate": [],
        }

    def train_step(
        self,
        batch,
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

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0,
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

    def validation_step(
        self,
        batch,
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

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 1,
        resume_from=None,
    ) -> None:
        if epochs <= 0:
            raise ValueError("epochs must be greater than zero.")

        start_epoch = 0

        if resume_from is not None:
            start_epoch = self._restore_checkpoint(resume_from)

        try:
            for epoch in range(
                start_epoch,
                epochs,
            ):
                metrics = self._run_epoch(
                    train_loader=train_loader,
                    val_loader=val_loader,
                )

                is_best = self._update_best_score(metrics)

                if self.scheduler is not None:
                    self.scheduler.step(metrics)

                learning_rate = self._get_learning_rate()

                metrics["learning_rate"] = learning_rate

                self._update_history(metrics)

                should_stop = False

                if self.early_stopping is not None:
                    should_stop = self.early_stopping.step(metrics)

                checkpoint = self._build_checkpoint(
                    epoch=epoch + 1,
                    metrics=metrics,
                )

                self.checkpoint.save(
                    checkpoint=checkpoint,
                    is_best=is_best,
                )

                self.logger.log_epoch(
                    epoch=epoch + 1,
                    epochs=epochs,
                    metrics=metrics,
                )

                if should_stop:
                    break
        finally:
            self.close()

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

    def _get_learning_rate(self) -> float:
        if self.scheduler is not None:
            return self.scheduler.get_learning_rate()

        if not self.optimizer.param_groups:
            raise RuntimeError("Optimizer has no parameter groups.")

        return float(self.optimizer.param_groups[0]["lr"])

    def _update_history(
        self,
        metrics: dict[str, float],
    ) -> None:
        for name, value in metrics.items():
            if name in self.history:
                self.history[name].append(float(value))

    def _build_checkpoint(
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
            "history": self.history,
        }

    def _restore_checkpoint(
        self,
        checkpoint_path,
    ) -> int:
        checkpoint = self.checkpoint.load(
            checkpoint_path=checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scheduler_state = checkpoint.get("scheduler_state_dict")

        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)

        early_stopping_state = checkpoint.get("early_stopping_state")

        if self.early_stopping is not None and early_stopping_state is not None:
            self.early_stopping.load_state_dict(early_stopping_state)

        self.best_score = checkpoint.get("best_score")

        saved_history = checkpoint.get("history")

        if isinstance(saved_history, dict):
            self.history = saved_history

        return int(checkpoint["epoch"])

    def close(self) -> None:
        self.logger.close()
