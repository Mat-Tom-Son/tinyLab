"""Determinism utilities for reproducible experiments."""
import random
import numpy as np
import torch


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Note: MPS determinism is not guaranteed by PyTorch
    # Always run multiple seeds and aggregate statistics
