"""Small helpers shared across practice scripts and notebooks."""

from __future__ import annotations

import random

import numpy as np
import torch


def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 0) -> None:
    """Seed Python, NumPy and PyTorch for reproducible practice runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
