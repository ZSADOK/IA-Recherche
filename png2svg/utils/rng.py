# rng.py
from __future__ import annotations

import random
import numpy as np
from typing import Optional


def seed_all(value: Optional[int]) -> None:
    """
    Seed Python + NumPy RNG for reproducibility.
    """
    random.seed(value)
    np.random.seed(value)
