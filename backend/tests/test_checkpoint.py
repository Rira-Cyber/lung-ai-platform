import torch

from backend.configs import DEV_CONFIG

from backend.models.unet3d import UNet3D
from backend.training.checkpoint import CheckpointManager


device = torch.device("cpu")


def test_checkpoint():

    print("=" * 60)
    print("Testing CheckpointManager")
    print("=" * 60)

    model = UNet3D(
        in_channels=1,
        out_channels=1,
        features=DEV_CONFIG.features,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=DEV_CONFIG.learning_rate,
    )

    checkpoint = CheckpointManager()

    epoch = 5
    loss = 0.1234

    print("\nSaving checkpoint...")

    checkpoint.save(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        loss=loss,
    )

    assert checkpoint.exists()

    print("✓ Save OK")

    print("\nLoading checkpoint...")

    loaded_epoch, loaded_loss = checkpoint.load(
        model=model,
        optimizer=optimizer,
    )

    assert loaded_epoch == epoch
    assert loaded_loss == loss

    print("Loaded Epoch :", loaded_epoch)
    print("Loaded Loss  :", loaded_loss)

    print("✓ Load OK")

    print("=" * 60)
    print("Checkpoint Test Passed")
    print("=" * 60)


if __name__ == "__main__":

    test_checkpoint()