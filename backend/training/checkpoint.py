from pathlib import Path

import torch


class CheckpointManager:
    """
    Save and load training checkpoints.
    """

    def __init__(
        self,
        save_dir: str = "checkpoints",
    ):

        self.save_dir = Path(save_dir)

        self.save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.latest_checkpoint = (
            self.save_dir / "latest.pt"
        )

        self.best_checkpoint = (
            self.save_dir / "best.pt"
        )

    def save(
        self,
        model,
        optimizer,
        epoch: int,
        loss: float,
        is_best: bool = False,
    ):

        checkpoint = {

            "epoch": epoch,

            "loss": loss,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

        }

        torch.save(
            checkpoint,
            self.latest_checkpoint,
        )

        if is_best:

            torch.save(
                checkpoint,
                self.best_checkpoint,
            )

    def load(
        self,
        model,
        optimizer=None,
        best: bool = False,
        map_location="cpu",
    ):

        checkpoint_path = (
            self.best_checkpoint
            if best
            else self.latest_checkpoint
        )

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if (
            optimizer is not None
            and
            "optimizer_state_dict"
            in checkpoint
        ):

            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        return (

            checkpoint["epoch"],

            checkpoint["loss"],

        )

    def exists(
        self,
        best: bool = False,
    ) -> bool:

        checkpoint_path = (
            self.best_checkpoint
            if best
            else self.latest_checkpoint
        )

        return checkpoint_path.exists()

    def latest_path(self):

        return self.latest_checkpoint

    def best_path(self):

        return self.best_checkpoint