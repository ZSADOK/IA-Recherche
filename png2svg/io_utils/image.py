# image.py
from __future__ import annotations
import os
import cv2
import numpy as np


def load_image_bgr(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input image not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Unable to decode image: {path}")
    return img
