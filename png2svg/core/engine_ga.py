# engine.py
from __future__ import annotations

import time
import random
import cv2
import numpy as np

from core.genotype import Genotype
from core.phenotype import Phenotype
from core.fitness import l1_loss
from core.mutation import random_shape, mutate_one_shape_inplace
from utils.visualizer import Visualizer


class GAEngine:
    def __init__(
            self,
            target_bgr: np.ndarray,
            shape_mode: str,
            n_shapes: int,
            time_limit: float,
            enable_viz: bool = True,
            fitness_scale: int = 4,
            population_size: int = 20,
            mutation_rate: float = 0.25,
    ):
        self.target = target_bgr
        self.height, self.width = target_bgr.shape[:2]
        self.shape_mode = shape_mode
        self.n_shapes = int(n_shapes)
        self.time_limit = float(time_limit)
        self.enable_viz = enable_viz

        mean = self.target.mean(axis=(0, 1))
        self.background_bgr = tuple(int(c) for c in mean)

        self.scale = max(2, int(fitness_scale))
        self.pop_size = max(6, int(population_size))
        self.mutation_rate = float(max(0.0, min(1.0, mutation_rate)))

        self.phen_small = Phenotype(self.width, self.height, self.background_bgr, scale=self.scale)
        self.target_small = cv2.resize(
            self.target,
            (self.phen_small.width, self.phen_small.height),
            interpolation=cv2.INTER_AREA,
        )
        self.phen_full = Phenotype(self.width, self.height, self.background_bgr, scale=1)

        self.best_fitness = float("inf")
        self.best: Genotype | None = None

    def _fitness(self, g: Genotype) -> float:
        img = self.phen_small.render(g)
        return l1_loss(self.target_small, img)

    def _init_individual(self) -> Genotype:
        shapes = [
            random_shape(self.width, self.height, self.target, self.shape_mode,
                         min_size=6, max_size=int(min(self.width, self.height) * 0.35), alpha_floor=0.65)
            for _ in range(self.n_shapes)
        ]
        return Genotype(shapes)

    def run(self) -> Genotype:
        start = time.time()
        t_end = start + self.time_limit
        viz = Visualizer(self.target) if self.enable_viz else None

        pop = [self._init_individual() for _ in range(self.pop_size)]

        scored = [(self._fitness(g), g) for g in pop]
        scored.sort(key=lambda x: x[0])
        self.best_fitness = scored[0][0]
        self.best = scored[0][1].copy()

        gen = 0
        last_print = 0.0

        while time.time() < t_end:
            gen += 1
            scored = [(self._fitness(g), g) for g in pop]
            scored.sort(key=lambda x: x[0])

            if scored[0][0] < self.best_fitness:
                self.best_fitness = scored[0][0]
                self.best = scored[0][1].copy()

            new_pop = [scored[0][1].copy(), scored[1][1].copy()]

            def pick_parent() -> Genotype:
                k = 4
                cand = random.sample(scored[: max(6, self.pop_size // 2)], k=min(k, len(scored)))
                cand.sort(key=lambda x: x[0])
                return cand[0][1]

            while len(new_pop) < self.pop_size:
                p = pick_parent().copy()

                if random.random() < self.mutation_rate:
                    for _ in range(random.randint(1, 4)):
                        idx = random.randrange(len(p.shapes))
                        mutate_one_shape_inplace(p.shapes[idx], self.width, self.height, self.target, self.scale, 0.65)

                new_pop.append(p)

            pop = new_pop

            now = time.time()
            if now - last_print >= 0.40:
                pct = 100.0 * min(1.0, (now - start) / self.time_limit)
                print(f"\r[{pct:5.1f}%] GA best L1={self.best_fitness:8.2f} gen={gen:4d}", end="", flush=True)
                last_print = now
                if viz and self.best:
                    viz.update(self.phen_full.render(self.best))

        if viz:
            viz.close()

        print()
        assert self.best is not None
        return self.best
