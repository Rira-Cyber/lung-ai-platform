from .config import Config


DEV_CONFIG = Config(
    # Dataset
    dataset_path=(
        r"E:\My codes\Medical AI"
        r"\lung-ai-platform\data\raw\lidc_idri"
    ),
    patch_size=(64, 64, 64),
    positive_ratio=0.7,
    patches_per_patient=1,
    # Dataset split
    val_ratio=0.2,
    test_ratio=0.0,
    split_seed=42,
    patient_manifest_path=None,
    experiment_root="experiments",
    experiment_name="development",
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
    max_grad_norm=1.0,
    # Scheduler
    scheduler_enabled=True,
    scheduler_type="reduce_on_plateau",
    scheduler_monitor="val_loss",
    scheduler_factor=0.5,
    scheduler_patience=2,
    scheduler_min_lr=1e-6,
    scheduler_step_size=10,
    scheduler_gamma=0.5,
    # Early stopping
    early_stopping_enabled=True,
    early_stopping_monitor="val_loss",
    early_stopping_mode="min",
    early_stopping_patience=5,
    early_stopping_min_delta=1e-4,
    # Best model selection
    best_metric="val_dice",
    best_mode="max",
    # Checkpoint
    checkpoint_dir="checkpoints/development",
    # Logging
    log_dir="logs/development",
    csv_logging_enabled=True,
    tensorboard_enabled=True,
    tensorboard_log_dir=("logs/development/tensorboard"),
)
