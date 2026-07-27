from .config import Config


DEV_CONFIG = Config(

    dataset_path=r"E:\My codes\Medical AI\lung-ai-platform\data\raw\lidc_idri",

    patch_size=(64, 64, 64),

    positive_ratio=0.7,

    patches_per_patient=1,

    batch_size=1,

    num_workers=0,

    pin_memory=False,

    shuffle=True,

    in_channels=1,

    out_channels=1,

    features=(8, 16, 32, 64),

    learning_rate=1e-3,

    epochs=2,

    device="cpu",
)