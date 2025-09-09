"""
Training fail-fast guards - catch classic breakages early
"""
import math
import os
import random
from typing import Optional

import numpy as np
import torch


def set_determinism(seed: int = 1337) -> None:
    """Set deterministic behavior for all random number generators"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def assert_no_nan_inf(model: torch.nn.Module) -> None:
    """Assert no NaN or Inf values in model gradients"""
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad = param.grad
            if not torch.isfinite(grad).all():
                raise RuntimeError(f"Non-finite grad in {name}")


def worker_init_fn(worker_id: int) -> None:
    """Ensures per-worker deterministic shuffles"""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
