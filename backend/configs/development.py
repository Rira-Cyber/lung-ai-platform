from .config import Config


DEV_CONFIG = Config(
    # Dataset
    dataset_path=(r"E:\My codes\Medical AI\lung-ai-platform\data\raw\lidc_idri"),
    patch_size=(64, 64, 64),
    positive_ratio=0.7,
    patches_per_patient=1,
    # Dataset split
    val_ratio=0.2,
    split_seed=42,
    # DataLoader
    batch_size=1,
    num_workers=0,
    pin_memory=False,
    shuffle=True,
    # Model
    in_channels=1,
    out_channels=1,
    features=(8, 16, 32, 64),
    # Training
    learning_rate=1e-3,
    epochs=2,
    device="cpu",
)
