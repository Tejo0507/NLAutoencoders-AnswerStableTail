"""Deterministic seeding.

Seeds are derived, not shared. A stage asks for ``derive(cfg.seed, "traces",
problem_id)`` and gets a stable 32-bit value, so re-running one problem
reproduces exactly that problem regardless of what order the others ran in.
That property is what makes the resumable stages reproducible; a single global
RNG advanced by iteration order would not be.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any


def derive(*parts: Any) -> int:
    """Stable 32-bit seed from any sequence of hashable-as-string parts."""
    blob = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:4], "big")


def seed_everything(seed: int, deterministic_torch: bool = False) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            # Off by default: it makes bf16 matmul noticeably slower and this
            # study's generation is sampled anyway, so the gain is small.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def rng(*parts: Any):
    """A numpy Generator seeded from ``derive(*parts)``."""
    import numpy as np

    return np.random.default_rng(derive(*parts))
