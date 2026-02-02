# genotype.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List
from core.shapes import Shape


@dataclass
class Genotype:
    shapes: List[Shape]

    def copy(self) -> "Genotype":
        return Genotype([s.copy() for s in self.shapes])

    def __len__(self) -> int:
        return len(self.shapes)
