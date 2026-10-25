"""Frame quality gate (DESIGN.md §6.3): reject frames before spending CPU on them."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

BLUR_MIN_VAR = 60.0       # variance of Laplacian at 320 px height
BLOWN_P95 = 250
MIN_FACE_PX = 60


@dataclass
class Quality:
    mean: float
    p5: float
    p95: float
    sharpness: float

    def reason(self, dark_threshold: float) -> str | None:
        if self.mean < dark_threshold:
            return "dark"
        if self.p95 > BLOWN_P95 and self.p5 > 200:
            return "blown-out"
        if self.sharpness < BLUR_MIN_VAR:
            return "blurred"
        return None

    def score(self) -> float:
        """0..1, used to rank frames for fusion: well-exposed and sharp wins."""
        exposure = 1.0 - min(abs(self.mean - 110) / 110, 1.0)
        sharp = min(self.sharpness / 400.0, 1.0)
        return 0.5 * exposure + 0.5 * sharp


def measure(gray: np.ndarray, max_height: int = 320) -> Quality:
    h = gray.shape[0]
    g = gray if h <= max_height else cv2.resize(gray, (int(gray.shape[1] * max_height / h), max_height))
    lap = cv2.Laplacian(g, cv2.CV_64F).var()
    return Quality(float(g.mean()), float(np.percentile(g, 5)), float(np.percentile(g, 95)), float(lap))
