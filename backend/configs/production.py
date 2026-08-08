from .config import Config


PROD_CONFIG = Config(
    # Dataset
    dataset_path=(
        r"E:\My codes\Medical AI"
        r"\lung-ai-platform\data\raw\lidc_idri"
    ),
    patch_size=(128, 128, 128),
    positive_ratio=0.7,
    patches_per_patient=10,
    # Dataset split
    val_ratio=0.15,
    test_ratio=0.10,
    split_seed=42,
    patient_manifest_path=None,
    experiment_root="experiments",
    experiment_name="production",
    # DataLoader
    batch_size=2,
    num_workers=2,
    pin_memory=True,
    shuffle=True,
    # Model
    in_channels=1,
    out_channels=1,
    features=(32, 64, 128, 256),
    # Training
    learning_rate=1e-4,
    epochs=100,
    device="cuda",
    max_grad_norm=1.0,
    # Scheduler
    scheduler_enabled=True,
    scheduler_type="reduce_on_plateau",
    scheduler_monitor="val_loss",
    scheduler_factor=0.5,
    scheduler_patience=5,
    scheduler_min_lr=1e-7,
    scheduler_step_size=20,
    scheduler_gamma=0.5,
    # Early stopping
    early_stopping_enabled=True,
    early_stopping_monitor="val_loss",
    early_stopping_mode="min",
    early_stopping_patience=15,
    early_stopping_min_delta=1e-4,
    # Best model selection
    best_metric="val_dice",
    best_mode="max",
    # Checkpoint
    checkpoint_dir="checkpoints/production",
    # Logging
    log_dir="logs/production",
    csv_logging_enabled=True,
    tensorboard_enabled=True,
    tensorboard_log_dir=("logs/production/tensorboard"),
)
