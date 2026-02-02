# shapes.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple
import math
import random

import cv2
import numpy as np

RGB = Tuple[int, int, int]
BGR = Tuple[int, int, int]


def clamp_int(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x


def rgb_to_bgr(rgb: RGB) -> BGR:
    r, g, b = rgb
    return (b, g, r)


def sample_rgb_from_target(target_bgr: np.ndarray, x: int, y: int) -> RGB:
    h, w = target_bgr.shape[:2]
    x = clamp_int(x, 0, w - 1)
    y = clamp_int(y, 0, h - 1)
    b, g, r = map(int, target_bgr[y, x])
    return (r, g, b)


class Shape(ABC):
    def __init__(self) -> None:
        self.age: int = 0

    @property
    @abstractmethod
    def alpha(self) -> float: ...

    @alpha.setter
    @abstractmethod
    def alpha(self, v: float) -> None: ...

    @abstractmethod
    def draw_on(self, canvas_bgr: np.ndarray, scale: int = 1) -> None: ...

    @abstractmethod
    def to_svg(self) -> str: ...

    @abstractmethod
    def copy(self) -> "Shape": ...

    @abstractmethod
    def area(self) -> float: ...


@dataclass
class Rectangle(Shape):
    cx: int
    cy: int
    w: int
    h: int
    color_rgb: RGB
    _alpha: float
    angle_deg: float
    _age: int = 0

    def __post_init__(self) -> None:
        super().__init__()
        self.age = int(self._age)

    @property
    def alpha(self) -> float:
        return float(self._alpha)

    @alpha.setter
    def alpha(self, v: float) -> None:
        self._alpha = float(v)

    def area(self) -> float:
        return float(max(0, self.w) * max(0, self.h))

    def draw_on(self, canvas_bgr: np.ndarray, scale: int = 1) -> None:
        overlay = canvas_bgr.copy()
        cx = int(self.cx // scale)
        cy = int(self.cy // scale)
        w = max(1, int(self.w // scale))
        h = max(1, int(self.h // scale))

        rect = ((float(cx), float(cy)), (float(w), float(h)), float(self.angle_deg))
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.drawContours(overlay, [box], 0, rgb_to_bgr(self.color_rgb), thickness=-1)
        cv2.addWeighted(overlay, self.alpha, canvas_bgr, 1.0 - self.alpha, 0.0, canvas_bgr)

    def to_svg(self) -> str:
        r, g, b = self.color_rgb
        x = self.cx - self.w / 2.0
        y = self.cy - self.h / 2.0
        return (
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{self.w:.2f}" height="{self.h:.2f}" '
            f'fill="rgb({r},{g},{b})" fill-opacity="{self.alpha:.4f}" '
            f'transform="rotate({self.angle_deg:.2f} {self.cx:.2f} {self.cy:.2f})" />'
        )

    def copy(self) -> "Rectangle":
        c = Rectangle(self.cx, self.cy, self.w, self.h, tuple(self.color_rgb), self.alpha, self.angle_deg,
                      _age=self.age)
        c.age = self.age
        return c


@dataclass
class Circle(Shape):
    cx: int
    cy: int
    radius: int
    color_rgb: RGB
    _alpha: float
    _age: int = 0

    def __post_init__(self) -> None:
        super().__init__()
        self.age = int(self._age)

    @property
    def alpha(self) -> float:
        return float(self._alpha)

    @alpha.setter
    def alpha(self, v: float) -> None:
        self._alpha = float(v)

    def area(self) -> float:
        r = max(0, self.radius)
        return float(math.pi * r * r)

    def draw_on(self, canvas_bgr: np.ndarray, scale: int = 1) -> None:
        overlay = canvas_bgr.copy()
        cx = int(self.cx // scale)
        cy = int(self.cy // scale)
        r = max(1, int(self.radius // scale))
        cv2.circle(overlay, (cx, cy), r, rgb_to_bgr(self.color_rgb), thickness=-1)
        cv2.addWeighted(overlay, self.alpha, canvas_bgr, 1.0 - self.alpha, 0.0, canvas_bgr)

    def to_svg(self) -> str:
        r, g, b = self.color_rgb
        return (
            f'<circle cx="{self.cx:.2f}" cy="{self.cy:.2f}" r="{self.radius:.2f}" '
            f'fill="rgb({r},{g},{b})" fill-opacity="{self.alpha:.4f}" />'
        )

    def copy(self) -> "Circle":
        c = Circle(self.cx, self.cy, self.radius, tuple(self.color_rgb), self.alpha, _age=self.age)
        c.age = self.age
        return c


@dataclass
class Ellipse(Shape):
    cx: int
    cy: int
    rx: int
    ry: int
    color_rgb: RGB
    _alpha: float
    angle_deg: float
    _age: int = 0

    def __post_init__(self) -> None:
        super().__init__()
        self.age = int(self._age)

    @property
    def alpha(self) -> float:
        return float(self._alpha)

    @alpha.setter
    def alpha(self, v: float) -> None:
        self._alpha = float(v)

    def area(self) -> float:
        rx = max(0, self.rx)
        ry = max(0, self.ry)
        return float(math.pi * rx * ry)

    def draw_on(self, canvas_bgr: np.ndarray, scale: int = 1) -> None:
        overlay = canvas_bgr.copy()
        cx = int(self.cx // scale)
        cy = int(self.cy // scale)
        rx = max(1, int(self.rx // scale))
        ry = max(1, int(self.ry // scale))
        cv2.ellipse(
            overlay,
            (cx, cy),
            (rx, ry),
            float(self.angle_deg),
            0.0,
            360.0,
            rgb_to_bgr(self.color_rgb),
            thickness=-1,
        )
        cv2.addWeighted(overlay, self.alpha, canvas_bgr, 1.0 - self.alpha, 0.0, canvas_bgr)

    def to_svg(self) -> str:
        r, g, b = self.color_rgb
        return (
            f'<ellipse cx="{self.cx:.2f}" cy="{self.cy:.2f}" rx="{self.rx:.2f}" ry="{self.ry:.2f}" '
            f'fill="rgb({r},{g},{b})" fill-opacity="{self.alpha:.4f}" '
            f'transform="rotate({self.angle_deg:.2f} {self.cx:.2f} {self.cy:.2f})" />'
        )

    def copy(self) -> "Ellipse":
        e = Ellipse(self.cx, self.cy, self.rx, self.ry, tuple(self.color_rgb), self.alpha, self.angle_deg,
                    _age=self.age)
        e.age = self.age
        return e
