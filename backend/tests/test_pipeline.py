import torch

from backend.configs import DEV_CONFIG

from backend.training.dataloader import create_train_loader
from backend.training.trainer import Trainer

from backend.models.unet3d import UNet3D

from backend.losses.dice import DiceLoss
from backend.metrics.dice import DiceMetric


device = torch.device("cpu")


def test_dataloader():

    print("=" * 60)
    print("1. Building DataLoader")

    loader = create_train_loader(
        dataset_path=DEV_CONFIG.dataset_path,
        batch_size=DEV_CONFIG.batch_size,
        num_workers=DEV_CONFIG.num_workers,
        patch_size=DEV_CONFIG.patch_size,
        positive_ratio=DEV_CONFIG.positive_ratio,
    )

    print("✓ DataLoader OK")
    return loader


def test_dataset(loader):

    print("=" * 60)
    print("2. Testing Dataset")

    sample = loader.dataset[0]

    print(sample.keys())

    print("Image :", sample["image"].shape)
    print("Mask  :", sample["mask"].shape)
    print("Mask Sum :", sample["mask"].sum().item())
    print("Positive :", sample["is_positive"])

    assert sample["image"].shape == sample["mask"].shape

    print("✓ Dataset OK")


def test_batch(loader):

    print("=" * 60)
    print("3. Testing Batch")

    batch = next(iter(loader))

    print(batch["image"].shape)
    print(batch["mask"].shape)

    assert batch["image"].shape == batch["mask"].shape

    print("✓ Batch OK")

    return batch


def test_model():

    print("=" * 60)
    print("4. Building Model")

    model = UNet3D(
        in_channels=1,
        out_channels=1,
        features=DEV_CONFIG.features,
    ).to(device)

    print("✓ Model OK")

    return model


def test_forward(model, batch):

    print("=" * 60)
    print("5. Forward Pass")

    images = batch["image"].to(device)

    predictions = model(images)

    print(predictions.shape)

    assert predictions.shape == batch["mask"].shape

    print("✓ Forward OK")

    return predictions


def test_loss(predictions, batch):

    print("=" * 60)
    print("6. Dice Loss")

    criterion = DiceLoss()

    masks = batch["mask"].to(device)

    loss = criterion(
        predictions,
        masks,
    )

    print(loss.item())

    print("✓ Loss OK")

    return criterion, loss


def test_metric(predictions, batch):

    print("=" * 60)
    print("7. Dice Metric")

    metric = DiceMetric()

    masks = batch["mask"].to(device)

    dice = metric(
        predictions,
        masks,
    )

    print(dice.item())

    print("✓ Metric OK")

    return metric


def test_backward(model, loss):

    print("=" * 60)
    print("8. Backward")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=DEV_CONFIG.learning_rate,
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print("✓ Backward OK")

    return optimizer


def test_trainer(
    model,
    optimizer,
    criterion,
    metric,
    batch,
):

    print("=" * 60)
    print("9. Trainer")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        metric=metric,
        device=device,
    )

    loss, dice = trainer.train_step(batch)

    print("Loss :", loss)
    print("Dice :", dice)

    print("✓ Trainer OK")


def run_pipeline_test():

    loader = test_dataloader()

    test_dataset(loader)

    batch = test_batch(loader)

    model = test_model()

    predictions = test_forward(
        model,
        batch,
    )

    criterion, loss = test_loss(
        predictions,
        batch,
    )

    metric = test_metric(
        predictions,
        batch,
    )

    optimizer = test_backward(
        model,
        loss,
    )

    test_trainer(
        model,
        optimizer,
        criterion,
        metric,
        batch,
    )

    print("=" * 60)
    print("PIPELINE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline_test()