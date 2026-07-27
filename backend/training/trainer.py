from __future__ import annotations

from pathlib import Path
from backend.training.checkpoint import CheckpointManager
import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        metric: nn.Module,
        checkpoint_dir="checkpoints",
        device: str = "cpu",
        
    ) -> None:

        self.model = model.to(device)

        self.checkpoint = CheckpointManager(
            save_dir=checkpoint_dir,
        )

        self.optimizer = optimizer

        self.criterion = criterion

        self.metric = metric

        self.device = device

        self.best_score = float("-inf")

        self.history = {
            "train_loss": [],
            "train_dice": [],
            "val_loss": [],
            "val_dice": [],
        }

    ####################################################################
    # Train Step
    ####################################################################

    def train_step(
        self,
        batch,
    ):

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
            loss.item(),
            dice.item(),
        )
    ####################################################################
# Validation Step
####################################################################

    def validation_step(
        self,
        batch,
    ):

        images = batch["image"].to(self.device)

        masks = batch["mask"].to(self.device)

        with torch.no_grad():

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
            loss.item(),
            dice.item(),
        )

    ####################################################################
    # Train Epoch
    ####################################################################

    def train_epoch(
        self,
        train_loader,
    ):

        self.model.train()

        epoch_loss = 0.0

        epoch_dice = 0.0

        progress = tqdm(
            train_loader,
            desc="Training",
            leave=False,
        )

        for batch in progress:

            loss, dice = self.train_step(
                batch,
            )

            epoch_loss += loss

            epoch_dice += dice

            progress.set_postfix(

                loss=f"{loss:.4f}",

                dice=f"{dice:.4f}",
            )

        epoch_loss /= len(train_loader)

        epoch_dice /= len(train_loader)

        return epoch_loss, epoch_dice

    ####################################################################
    # Validation Epoch
    ####################################################################

    def validation_epoch(
        self,
        val_loader,
    ):

        self.model.eval()

        epoch_loss = 0.0

        epoch_dice = 0.0

        progress = tqdm(
            val_loader,
            desc="Validation",
            leave=False,
        )

        for batch in progress:

            loss, dice = self.validation_step(
                batch,
            )

            epoch_loss += loss

            epoch_dice += dice

            progress.set_postfix(

                loss=f"{loss:.4f}",

                dice=f"{dice:.4f}",
            )

        epoch_loss /= len(val_loader)

        epoch_dice /= len(val_loader)

        return epoch_loss, epoch_dice
    
        ####################################################################
    # Fit
    ####################################################################

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs=1,
    ):
        """
        Train model.
        """

        for epoch in range(epochs):

            print(f"\nEpoch {epoch+1}/{epochs}")

            train_loss, train_dice = self.train_epoch(
                train_loader,
            )

            print(
                f"Train Loss: {train_loss:.4f} | "
                f"Train Dice: {train_dice:.4f}"
            )

            if val_loader is not None:

                val_loss, val_dice = self.validation_epoch(
                    val_loader,
                )

                print(
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Dice: {val_dice:.4f}"
                )

            self.history["train_loss"].append(train_loss)
            self.history["train_dice"].append(train_dice)

            if val_loader is not None:

                self.history["val_loss"].append(val_loss)
                self.history["val_dice"].append(val_dice)

                is_best = val_dice > self.best_score

                if is_best:
                    self.best_score = val_dice

            else:

                is_best = False

            self.checkpoint.save(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch + 1,
                loss=train_loss,
                is_best=is_best,
            )

    