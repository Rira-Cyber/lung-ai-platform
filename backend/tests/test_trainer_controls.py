from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from backend.training.early_stopping import EarlyStopping
from backend.training.scheduler import SchedulerController
from backend.training.trainer import Trainer


class TinyDataset(Dataset):
    def __init__(
        self,
        size: int = 4,
    ) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(
        self,
        index: int,
    ) -> dict:
        value = float(index + 1)

        image = torch.full(
            (1, 2, 2, 2),
            value,
            dtype=torch.float32,
        )

        mask = torch.ones(
            (1, 2, 2, 2),
            dtype=torch.float32,
        )

        return {
            "image": image,
            "mask": mask,
        }


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return inputs * self.weight


class MeanSquaredError(nn.Module):
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return torch.mean((predictions - targets) ** 2)


class SimpleDice(nn.Module):
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        predictions = torch.sigmoid(predictions)

        intersection = torch.sum(predictions * targets)

        denominator = torch.sum(predictions) + torch.sum(targets)

        return (2.0 * intersection + 1e-6) / (denominator + 1e-6)


class SilentLogger:
    def __init__(self) -> None:
        self.records: list[dict] = []
        self.closed = False

    def log_epoch(
        self,
        epoch: int,
        epochs: int,
        metrics: dict,
    ) -> None:
        self.records.append(
            {
                "epoch": epoch,
                "epochs": epochs,
                "metrics": dict(metrics),
            }
        )

    def close(self) -> None:
        self.closed = True


def create_loaders() -> tuple[
    DataLoader,
    DataLoader,
]:
    train_loader = DataLoader(
        TinyDataset(size=4),
        batch_size=2,
        shuffle=False,
    )

    val_loader = DataLoader(
        TinyDataset(size=2),
        batch_size=1,
        shuffle=False,
    )

    return train_loader, val_loader


def create_trainer(
    tmp_path: Path,
    *,
    scheduler: SchedulerController | None = None,
    early_stopping: EarlyStopping | None = None,
    logger: SilentLogger | None = None,
) -> Trainer:
    model = TinyModel()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )

    return Trainer(
        model=model,
        optimizer=optimizer,
        criterion=MeanSquaredError(),
        metric=SimpleDice(),
        checkpoint_dir=tmp_path,
        device="cpu",
        logger=logger or SilentLogger(),
        scheduler=scheduler,
        early_stopping=early_stopping,
        best_metric="val_loss",
        best_mode="min",
        max_grad_norm=1.0,
    )


def test_scheduler_updates_learning_rate(
    tmp_path: Path,
) -> None:
    model = TinyModel()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )

    scheduler = SchedulerController(
        torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=0.5,
        )
    )

    logger = SilentLogger()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=MeanSquaredError(),
        metric=SimpleDice(),
        checkpoint_dir=tmp_path,
        logger=logger,
        scheduler=scheduler,
        best_metric="val_loss",
        best_mode="min",
    )

    train_loader, val_loader = create_loaders()

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
    )

    assert trainer.history["learning_rate"] == pytest.approx(
        [
            0.05,
            0.025,
        ]
    )

    assert logger.closed is True


def test_latest_and_best_checkpoints_are_saved(
    tmp_path: Path,
) -> None:
    trainer = create_trainer(tmp_path)

    train_loader, val_loader = create_loaders()

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
    )

    assert trainer.checkpoint.exists()
    assert trainer.checkpoint.exists(best=True)

    latest = trainer.checkpoint.load()
    best = trainer.checkpoint.load(best=True)

    assert latest["epoch"] == 2
    assert best["epoch"] in {
        1,
        2,
    }

    assert latest["best_metric"] == "val_loss"

    assert latest["best_mode"] == "min"


def test_checkpoint_contains_training_control_state(
    tmp_path: Path,
) -> None:
    model = TinyModel()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )

    scheduler = SchedulerController(
        torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=0.5,
        )
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=MeanSquaredError(),
        metric=SimpleDice(),
        checkpoint_dir=tmp_path,
        logger=SilentLogger(),
        scheduler=scheduler,
        early_stopping=early_stopping,
        best_metric="val_loss",
        best_mode="min",
    )

    train_loader, val_loader = create_loaders()

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
    )

    checkpoint = trainer.checkpoint.load()

    assert checkpoint["scheduler_state_dict"] is not None

    assert checkpoint["early_stopping_state"] is not None

    assert checkpoint["best_score"] is not None
    assert checkpoint["history"]


def test_resume_restores_scheduler_history_and_best_score(
    tmp_path: Path,
) -> None:
    first_model = TinyModel()

    first_optimizer = torch.optim.SGD(
        first_model.parameters(),
        lr=0.1,
    )

    first_scheduler = SchedulerController(
        torch.optim.lr_scheduler.StepLR(
            first_optimizer,
            step_size=1,
            gamma=0.5,
        )
    )

    first_early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
    )

    first_trainer = Trainer(
        model=first_model,
        optimizer=first_optimizer,
        criterion=MeanSquaredError(),
        metric=SimpleDice(),
        checkpoint_dir=tmp_path,
        logger=SilentLogger(),
        scheduler=first_scheduler,
        early_stopping=first_early_stopping,
        best_metric="val_loss",
        best_mode="min",
    )

    train_loader, val_loader = create_loaders()

    first_trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
    )

    saved_history = {
        key: values.copy() for key, values in first_trainer.history.items()
    }

    saved_best_score = first_trainer.best_score

    second_model = TinyModel()

    second_optimizer = torch.optim.SGD(
        second_model.parameters(),
        lr=0.1,
    )

    second_scheduler = SchedulerController(
        torch.optim.lr_scheduler.StepLR(
            second_optimizer,
            step_size=1,
            gamma=0.5,
        )
    )

    second_early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
    )

    second_trainer = Trainer(
        model=second_model,
        optimizer=second_optimizer,
        criterion=MeanSquaredError(),
        metric=SimpleDice(),
        checkpoint_dir=tmp_path,
        logger=SilentLogger(),
        scheduler=second_scheduler,
        early_stopping=second_early_stopping,
        best_metric="val_loss",
        best_mode="min",
    )

    second_trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
        resume_from=(tmp_path / "latest.pt"),
    )

    assert len(second_trainer.history["train_loss"]) == 3

    assert second_trainer.history["train_loss"][:2] == saved_history["train_loss"]

    assert saved_best_score is not None
    assert second_trainer.best_score is not None

    assert second_trainer.scheduler.get_learning_rate() == pytest.approx(0.0125)


def test_early_stopping_breaks_training_loop(
    tmp_path: Path,
    monkeypatch,
) -> None:
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        mode="min",
    )

    trainer = create_trainer(
        tmp_path,
        early_stopping=early_stopping,
    )

    train_results = iter(
        [
            (1.0, 0.5),
            (1.0, 0.5),
            (1.0, 0.5),
            (1.0, 0.5),
        ]
    )

    val_results = iter(
        [
            (1.0, 0.5),
            (1.1, 0.5),
            (1.2, 0.5),
            (1.3, 0.5),
        ]
    )

    monkeypatch.setattr(
        trainer,
        "train_epoch",
        lambda loader: next(train_results),
    )

    monkeypatch.setattr(
        trainer,
        "validation_epoch",
        lambda loader: next(val_results),
    )

    trainer.fit(
        train_loader=[{}],
        val_loader=[{}],
        epochs=10,
    )

    assert len(trainer.history["val_loss"]) == 3

    assert early_stopping.should_stop is True
    assert early_stopping.bad_epochs == 2

    latest = trainer.checkpoint.load()

    assert latest["epoch"] == 3


def test_validation_control_requires_validation_loader(
    tmp_path: Path,
) -> None:
    trainer = create_trainer(tmp_path)

    train_loader, _ = create_loaders()

    with pytest.raises(
        ValueError,
        match="requires a validation DataLoader",
    ):
        trainer.fit(
            train_loader=train_loader,
            val_loader=None,
            epochs=1,
        )


def test_resume_rejects_missing_scheduler_component(
    tmp_path: Path,
) -> None:
    first_model = TinyModel()

    first_optimizer = torch.optim.SGD(
        first_model.parameters(),
        lr=0.1,
    )

    scheduler = SchedulerController(
        torch.optim.lr_scheduler.StepLR(
            first_optimizer,
            step_size=1,
        )
    )

    first_trainer = Trainer(
        model=first_model,
        optimizer=first_optimizer,
        criterion=MeanSquaredError(),
        metric=SimpleDice(),
        checkpoint_dir=tmp_path,
        logger=SilentLogger(),
        scheduler=scheduler,
        best_metric="val_loss",
        best_mode="min",
    )

    train_loader, val_loader = create_loaders()

    first_trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1,
    )

    second_trainer = create_trainer(tmp_path)

    with pytest.raises(
        ValueError,
        match="has no scheduler",
    ):
        second_trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2,
            resume_from=(tmp_path / "latest.pt"),
        )
