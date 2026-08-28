"""§7.2 Anti-spoof CNN on the RGB crop (Silent-Face-Anti-Spoofing / MiniFASNet, ONNX via cv2.dnn).

The model file is optional: without it this voter is disabled and `doctor` says so.
`faceup setup` downloads it from a pinned release with a checksum — never at auth time.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import ANTISPOOF_MODEL
from ..detect import Box
from . import ABSTAIN, FAIL, PASS, Vote

NAME = "cnn"
INPUT = 80            # MiniFASNet v2 input size
SCALE = 2.7           # crop scale used by the reference implementation
PASS_P = 0.7
FAIL_P = 0.3
MIN_CROP_PX = 80

_net = None
_missing = False


def available() -> bool:
    return ANTISPOOF_MODEL.exists()


def _load():
    global _net, _missing
    if _net is None and not _missing:
        if not available():
            _missing = True
        else:
            _net = cv2.dnn.readNetFromONNX(str(ANTISPOOF_MODEL))
    return _net


def _crop(rgb: np.ndarray, box: Box) -> np.ndarray | None:
    h, w = rgb.shape[:2]
    cx, cy = box.x + box.w / 2, box.y + box.h / 2
    side = int(max(box.w, box.h) * SCALE)
    x0, y0 = int(cx - side / 2), int(cy - side / 2)
    x1, y1 = x0 + side, y0 + side
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < MIN_CROP_PX or y1 - y0 < MIN_CROP_PX:
        return None
    return cv2.resize(rgb[y0:y1, x0:x1], (INPUT, INPUT))


def vote(rgb: np.ndarray | None, box: Box | None, rgb_dark: bool) -> Vote:
    net = _load()
    if net is None:
        return ABSTAIN(NAME, "model not installed")
    if rgb is None or box is None:
        return ABSTAIN(NAME, "no rgb face")
    if rgb_dark:
        return ABSTAIN(NAME, "rgb dark")
    crop = _crop(rgb, box)
    if crop is None:
        return ABSTAIN(NAME, "face too small")
    blob = cv2.dnn.blobFromImage(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), 1.0, (INPUT, INPUT), (0, 0, 0), swapRB=False)
    net.setInput(blob)
    out = net.forward().reshape(-1)
    probs = np.exp(out - out.max()); probs /= probs.sum()
    # MiniFASNet classes: 0 = print/paper, 1 = real, 2 = replay/screen
    p_real = float(probs[1]) if len(probs) >= 3 else float(probs[-1])
    if p_real >= PASS_P:
        return PASS(NAME, f"real {p_real:.2f}", p_real)
    if p_real <= FAIL_P:
        return FAIL(NAME, f"spoof {1 - p_real:.2f}", p_real)
    return ABSTAIN(NAME, f"uncertain {p_real:.2f}", p_real)
