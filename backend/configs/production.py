from .config import Config


PROD_CONFIG = Config(

    dataset_path=r"E:\My codes\Medical AI\lung-ai-platform\data\raw\lidc_idri",

    patch_size=(128, 128, 128),

    positive_ratio=0.7,

    patches_per_patient=10,

    batch_size=2,

    num_workers=2,

    pin_memory=True,

    shuffle=True,

    in_channels=1,

    out_channels=1,

    features=(32, 64, 128, 256),

    learning_rate=1e-4,

    epochs=100,

    device="cuda",
)