# serialization.py
from __future__ import annotations

import pickle
from typing import Any

from core.genotype import Genotype


def save_solution(path: str, genotype: Genotype) -> None:
    """Save genotype for restart / analysis."""
    with open(path, "wb") as f:
        pickle.dump(genotype, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_solution(path: str) -> Genotype:
    """Load genotype from file."""
    with open(path, "rb") as f:
        obj: Any = pickle.load(f)
    if not isinstance(obj, Genotype):
        raise ValueError("Loaded object is not a Genotype.")
    return obj
