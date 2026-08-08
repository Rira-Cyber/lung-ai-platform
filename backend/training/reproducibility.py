from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seed(
    seed: int,
    *,
    deterministic: bool = False,
) -> None:
    """
    Seed Python, NumPy, and PyTorch random generators.
    """

    if seed < 0:
        raise ValueError("seed cannot be negative.")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
