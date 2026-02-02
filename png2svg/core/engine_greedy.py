# engine_greedy.py
from __future__ import annotations

import time
import random
import cv2
import numpy as np

from core.genotype import Genotype
from core.phenotype import Phenotype
from core.fitness import l1_loss, error_map_gray
from core.mutation import propose_shape_near, mutate_one_shape_inplace
from utils.visualizer import Visualizer


class GreedyEngine:
    def __init__(
            self,
            target_bgr: np.ndarray,
            shape_mode: str,
            n_shapes: int,
            time_limit: float,
            enable_viz: bool = True,
            fitness_scale: int = 4,
            candidates_per_shape: int = 45,
            refine_fraction: float = 0.60,
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
        self.candidates = max(10, int(candidates_per_shape))
        self.refine_fraction = float(max(0.0, min(0.95, refine_fraction)))

        self.phen_small = Phenotype(self.width, self.height, self.background_bgr, scale=self.scale)
        self.target_small = cv2.resize(
            self.target,
            (self.phen_small.width, self.phen_small.height),
            interpolation=cv2.INTER_AREA,
        )
        self.phen_full = Phenotype(self.width, self.height, self.background_bgr, scale=1)

        self.best_fitness = float("inf")
        self.best: Genotype | None = None

    def _pick_hotspot(self, current_small: np.ndarray) -> tuple[int, int]:
        em = error_map_gray(self.target_small, current_small)
        flat = em.reshape(-1)
        k = max(10, int(0.10 * flat.size))
        idx = np.argpartition(flat, -k)[-k:]
        pick = int(np.random.choice(idx))
        y, x = divmod(pick, em.shape[1])
        return int(x), int(y)

    def run(self) -> Genotype:
        start = time.time()
        t_end = start + self.time_limit
        t_refine_start = start + (1.0 - self.refine_fraction) * self.time_limit

        viz = Visualizer(self.target) if self.enable_viz else None

        g = Genotype([])
        current_small = self.phen_small.render(g)
        current_fit = l1_loss(self.target_small, current_small)

        self.best = g.copy()
        self.best_fitness = current_fit

        i = 0
        while i < self.n_shapes and time.time() < t_refine_start:
            current_small = self.phen_small.render(g)
            hx, hy = self._pick_hotspot(current_small)

            ratio = i / max(1, self.n_shapes - 1)
            max_size = int(max(6, (1.0 - ratio) * min(self.width, self.height) * 0.55))
            min_size = 4 if ratio > 0.55 else 10

            best_s = None
            best_fit = current_fit

            for _ in range(self.candidates):
                s = propose_shape_near(
                    width=self.width,
                    height=self.height,
                    target_bgr=self.target,
                    shape_mode=self.shape_mode,
                    hotspot_small=(hx, hy),
                    small_scale=self.scale,
                    min_size=min_size,
                    max_size=max_size,
                    alpha_floor=0.70,
                )
                tmp = g.copy()
                tmp.shapes.append(s)
                img = self.phen_small.render(tmp)
                f = l1_loss(self.target_small, img)
                if f < best_fit:
                    best_fit = f
                    best_s = s

            if best_s is None:
                best_s = propose_shape_near(
                    width=self.width,
                    height=self.height,
                    target_bgr=self.target,
                    shape_mode=self.shape_mode,
                    hotspot_small=(hx, hy),
                    small_scale=self.scale,
                    min_size=min_size,
                    max_size=max_size,
                    alpha_floor=0.70,
                )
                tmp = g.copy()
                tmp.shapes.append(best_s)
                best_fit = l1_loss(self.target_small, self.phen_small.render(tmp))

            g.shapes.append(best_s)
            current_fit = best_fit
            i += 1

            if current_fit < self.best_fitness:
                self.best_fitness = current_fit
                self.best = g.copy()

            if i % 5 == 0:
                pct = 100.0 * min(1.0, (time.time() - start) / self.time_limit)
                print(f"\r[{pct:5.1f}%] build L1={self.best_fitness:8.2f} shapes={len(g):4d}/{self.n_shapes}",
                      end="", flush=True)
                if viz and self.best:
                    viz.update(self.phen_full.render(self.best))

        while len(g) < self.n_shapes and time.time() < t_refine_start:
            current_small = self.phen_small.render(g)
            hx, hy = self._pick_hotspot(current_small)
            s = propose_shape_near(
                width=self.width,
                height=self.height,
                target_bgr=self.target,
                shape_mode=self.shape_mode,
                hotspot_small=(hx, hy),
                small_scale=self.scale,
                min_size=6,
                max_size=int(max(10, min(self.width, self.height) * 0.25)),
                alpha_floor=0.70,
            )
            g.shapes.append(s)
            current_fit = l1_loss(self.target_small, self.phen_small.render(g))
            if current_fit < self.best_fitness:
                self.best_fitness = current_fit
                self.best = g.copy()

        mut_attempt = 0
        mut_accept = 0
        last_print = 0.0

        while time.time() < t_end and len(g) > 0:
            mut_attempt += 1
            idx = random.randrange(len(g.shapes))
            old = g.shapes[idx].copy()

            mutate_one_shape_inplace(
                s=g.shapes[idx],
                width=self.width,
                height=self.height,
                target_bgr=self.target,
                small_scale=self.scale,
                alpha_floor=0.70,
            )

            new_fit = l1_loss(self.target_small, self.phen_small.render(g))
            if new_fit <= current_fit:
                current_fit = new_fit
                mut_accept += 1
                if current_fit < self.best_fitness:
                    self.best_fitness = current_fit
                    self.best = g.copy()
            else:
                g.shapes[idx] = old  # inverser

            now = time.time()
            if now - last_print >= 0.35:
                pct = 100.0 * min(1.0, (now - start) / self.time_limit)
                rate = (mut_accept / mut_attempt) if mut_attempt else 0.0
                print(f"\r[{pct:5.1f}%] refine L1={self.best_fitness:8.2f} shapes={len(g):4d}/{self.n_shapes} "
                      f"mut={mut_attempt}/{mut_accept} ({rate * 100:4.1f}%)",
                      end="", flush=True)
                last_print = now
                if viz and self.best:
                    viz.update(self.phen_full.render(self.best))

        if viz:
            viz.close()

        print()
        assert self.best is not None
        return self.best
