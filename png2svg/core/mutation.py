# mutation.py
from __future__ import annotations

import random
from typing import Tuple
import numpy as np

from core.shapes import Rectangle, Circle, Ellipse, Shape, clamp_int, sample_rgb_from_target


def _choose_mode(shape_mode: str) -> str:
    if shape_mode in ("rectangle", "circle", "ellipse"):
        return shape_mode
    r = random.random()
    return "rectangle" if r < 0.34 else "circle" if r < 0.67 else "ellipse"


def random_shape(width: int, height: int, target_bgr: np.ndarray, shape_mode: str,
                 min_size: int, max_size: int, alpha_floor: float) -> Shape:
    mode = _choose_mode(shape_mode)
    cx = random.randint(0, width - 1)
    cy = random.randint(0, height - 1)
    color = sample_rgb_from_target(target_bgr, cx, cy)
    alpha = random.uniform(alpha_floor, 0.95)

    if mode == "rectangle":
        w = random.randint(min_size, max(min_size + 1, max_size))
        h = random.randint(min_size, max(min_size + 1, max_size))
        angle = random.uniform(0.0, 360.0)
        return Rectangle(cx, cy, w, h, color, alpha, angle, _age=0)

    if mode == "circle":
        r = random.randint(min_size, max(min_size + 1, max_size))
        return Circle(cx, cy, r, color, alpha, _age=0)

    rx = random.randint(min_size, max(min_size + 1, max_size))
    ry = random.randint(min_size, max(min_size + 1, max_size))
    angle = random.uniform(0.0, 360.0)
    return Ellipse(cx, cy, rx, ry, color, alpha, angle, _age=0)


def propose_shape_near(
        width: int,
        height: int,
        target_bgr: np.ndarray,
        shape_mode: str,
        hotspot_small: Tuple[int, int],
        small_scale: int,
        min_size: int,
        max_size: int,
        alpha_floor: float,
) -> Shape:
    hx_s, hy_s = hotspot_small
    hx = int(hx_s * small_scale)
    hy = int(hy_s * small_scale)

    jitter = max(8, 16 * small_scale)
    cx = clamp_int(hx + random.randint(-jitter, jitter), 0, width - 1)
    cy = clamp_int(hy + random.randint(-jitter, jitter), 0, height - 1)

    mode = _choose_mode(shape_mode)
    color = sample_rgb_from_target(target_bgr, cx, cy)
    alpha = random.uniform(alpha_floor, 0.95)

    if mode == "rectangle":
        w = random.randint(min_size, max(min_size + 1, max_size))
        h = random.randint(min_size, max(min_size + 1, max_size))
        angle = random.uniform(0.0, 360.0)
        return Rectangle(cx, cy, w, h, color, alpha, angle, _age=0)

    if mode == "circle":
        r = random.randint(min_size, max(min_size + 1, max_size))
        return Circle(cx, cy, r, color, alpha, _age=0)

    rx = random.randint(min_size, max(min_size + 1, max_size))
    ry = random.randint(min_size, max(min_size + 1, max_size))
    angle = random.uniform(0.0, 360.0)
    return Ellipse(cx, cy, rx, ry, color, alpha, angle, _age=0)


def mutate_one_shape_inplace(
        s: Shape,
        width: int,
        height: int,
        target_bgr: np.ndarray,
        small_scale: int,
        alpha_floor: float,
) -> None:
    step = max(2, int(6 * small_scale / 4))

    if hasattr(s, "cx"):
        s.cx = clamp_int(int(s.cx + random.randint(-step, step)), 0, width - 1)
    if hasattr(s, "cy"):
        s.cy = clamp_int(int(s.cy + random.randint(-step, step)), 0, height - 1)

    if isinstance(s, Rectangle):
        s.w = clamp_int(int(s.w + random.randint(-10, 10)), 4, max(8, width))
        s.h = clamp_int(int(s.h + random.randint(-10, 10)), 4, max(8, height))
        s.angle_deg = (float(s.angle_deg) + random.uniform(-8, 8)) % 360.0
    elif isinstance(s, Circle):
        s.radius = clamp_int(int(s.radius + random.randint(-8, 8)), 3, max(6, min(width, height)))
    elif isinstance(s, Ellipse):
        s.rx = clamp_int(int(s.rx + random.randint(-8, 8)), 3, max(6, width))
        s.ry = clamp_int(int(s.ry + random.randint(-8, 8)), 3, max(6, height))
        s.angle_deg = (float(s.angle_deg) + random.uniform(-8, 8)) % 360.0

    if random.random() < 0.45:
        s.color_rgb = sample_rgb_from_target(target_bgr, int(getattr(s, "cx", 0)), int(getattr(s, "cy", 0)))
    else:
        r, g, b = s.color_rgb
        s.color_rgb = (
            clamp_int(r + random.randint(-8, 8), 0, 255),
            clamp_int(g + random.randint(-8, 8), 0, 255),
            clamp_int(b + random.randint(-8, 8), 0, 255),
        )

    s.alpha = float(s.alpha + random.uniform(-0.03, 0.03))
    s.alpha = max(alpha_floor, min(0.98, s.alpha))
