# fitness.py
from __future__ import annotations
import numpy as np
import cv2


def l1_loss(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def error_map_gray(target_bgr: np.ndarray, current_bgr: np.ndarray) -> np.ndarray:
    t = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    c = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return np.abs(t - c)
