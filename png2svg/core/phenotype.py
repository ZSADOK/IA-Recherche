# phenotype.py
from __future__ import annotations
import numpy as np
from core.genotype import Genotype


class Phenotype:
    def __init__(self, width: int, height: int, background_bgr: tuple[int, int, int], scale: int = 1):
        self.scale = max(1, int(scale))
        self.width = max(1, int(width // self.scale))
        self.height = max(1, int(height // self.scale))
        self.background_bgr = tuple(int(c) for c in background_bgr)

    def render(self, genotype: Genotype) -> np.ndarray:
        canvas = np.empty((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = self.background_bgr
        for s in genotype.shapes:
            s.draw_on(canvas, scale=self.scale)
        return canvas
