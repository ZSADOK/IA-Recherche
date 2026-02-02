# crossover.py
from __future__ import annotations
import random
from core.genotype import Genotype


def crossover(a: Genotype, b: Genotype, max_shapes: int) -> Genotype:
    cut = random.randint(0, min(len(a), len(b)))
    shapes = a.shapes[:cut] + b.shapes[cut:]
    return Genotype([s.copy() for s in shapes[:max_shapes]])
