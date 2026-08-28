"""§7.3 Emitter-modulation shadow test (IR).

Toggle the IR LEDs between two frames; a real face near the LEDs shows a large and
UNEVEN brightness change (self-shadowing around nose and brows), a flat print changes
uniformly. Toggling reuses the UVC instruction linux-enable-ir-emitter discovered.

Status: the metric is implemented; the toggle hook is pluggable and OFF until
`faceup emitter toggle-bench` proves the camera can switch in < 50 ms (DESIGN.md §15.1).
"""
from __future__ import annotations

import cv2
import numpy as np

from ..detect import Box
from . import ABSTAIN, FAIL, PASS, Vote

NAME = "shadow"
PASS_CV = 0.25
FAIL_CV = 0.10
MIN_DELTA = 12.0      # mean brightness change inside the face; a distant screen barely changes


def metric(off_gray: np.ndarray, on_gray: np.ndarray, box: Box) -> tuple[float, float]:
    """(mean delta, coefficient of variation of delta across a 4x4 grid of face regions)."""
    y0, y1 = max(0, box.y), min(off_gray.shape[0], box.y + box.h)
    x0, x1 = max(0, box.x), min(off_gray.shape[1], box.x + box.w)
    off = off_gray[y0:y1, x0:x1].astype(np.float32)
    on = on_gray[y0:y1, x0:x1].astype(np.float32)
    if off.size == 0 or off.shape != on.shape:
        return 0.0, 0.0
    delta = cv2.GaussianBlur(on - off, (0, 0), 3)
    cells = []
    for i in range(4):
        for j in range(4):
            c = delta[i * delta.shape[0] // 4:(i + 1) * delta.shape[0] // 4,
                      j * delta.shape[1] // 4:(j + 1) * delta.shape[1] // 4]
            cells.append(float(c.mean()))
    cells = np.array(cells)
    mean = float(np.mean(cells))
    cv_ = float(np.std(cells) / abs(mean)) if abs(mean) > 1e-3 else 0.0
    return mean, cv_


def vote(off_gray: np.ndarray | None, on_gray: np.ndarray | None, box: Box | None) -> Vote:
    if off_gray is None or on_gray is None or box is None:
        return ABSTAIN(NAME, "toggle unavailable")
    mean, cv_ = metric(off_gray, on_gray, box)
    if mean < MIN_DELTA:
        return ABSTAIN(NAME, f"delta too small ({mean:.1f})", cv_)
    if cv_ >= PASS_CV:
        return PASS(NAME, f"self-shadowing cv {cv_:.2f}", cv_)
    if cv_ <= FAIL_CV:
        return FAIL(NAME, f"uniform illumination cv {cv_:.2f}", cv_)
    return ABSTAIN(NAME, f"inconclusive cv {cv_:.2f}", cv_)
