# visualizer.py
from __future__ import annotations
import cv2
import numpy as np


class Visualizer:
    def __init__(self, target_bgr: np.ndarray, window_name: str = "PNG2SVG - Live"):
        self.target = target_bgr
        self.h, self.w = target_bgr.shape[:2]
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def update(self, current_bgr: np.ndarray) -> None:
        if current_bgr.shape != self.target.shape:
            current_bgr = cv2.resize(current_bgr, (self.w, self.h), interpolation=cv2.INTER_AREA)
        display = np.hstack([self.target, current_bgr])
        cv2.imshow(self.window_name, display)
        cv2.waitKey(1)

    def close(self) -> None:
        cv2.destroyWindow(self.window_name)
