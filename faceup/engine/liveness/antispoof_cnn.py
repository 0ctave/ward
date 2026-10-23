"""§7.2 Anti-spoof CNN on the RGB crop (Silent-Face-Anti-Spoofing MiniFASNetV2, ONNX via cv2.dnn).

The model is bundled (faceup/models/minifasnet_v2.onnx, Apache-2.0, 1.7 MB, sha256 pinned below
and checked once at load). Preprocessing follows the reference implementation exactly: the
detector box (x, y, w, h) scaled ×2.7 around its centre (clamped so it fits the frame), cropped,
resized to 80×80, BGR, raw 0–255 floats, no mean/std. Output = 3 logits → softmax; class 1 = real.

Measured on this machine's recorded daylight RGB frames (80 frames, engine box): p_real min 0.997,
p5 0.999, median 1.000. The network is trained on colour images: on IR frames converted to RGB it
scores real faces as low as 0.01 — so it only ever runs on a genuine colour frame (a grey frame
abstains). Because a FAIL is sticky (ends the attempt as spoof), the voter is asymmetric:
  PASS     median p_real over the recent RGB frames ≥ PASS_P
  FAIL     at least FAIL_MIN_FRAMES scored frames and EVERY one ≤ FAIL_P (a confident, repeated spoof)
  ABSTAIN  everything else — an odd frame never sends a genuine user to the password.
"""
from __future__ import annotations

import hashlib

import cv2
import numpy as np

from ..config import ANTISPOOF_MODEL
from . import ABSTAIN, FAIL, PASS, Vote

NAME = "cnn"
SHA256 = "b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907"
INPUT = 80            # MiniFASNetV2 input size
SCALE = 2.7           # crop scale of the reference implementation for V2
REAL = 1              # class index: 0 print, 1 real, 2 replay
PASS_P = 0.90         # real faces measured ≥ 0.985 on every frame
FAIL_P = 0.02         # ~50× below the lowest real-face score measured
FAIL_MIN_FRAMES = 2
MAX_FRAMES = 3        # newest RGB frames scored per vote (~3 ms each)
MIN_FACE_PX = 40      # box side below this: too few pixels for texture cues
GREY_TOL = 2.0        # mean |R-G|+|G-B| below this = a grey frame (IR or mono sensor) → abstain

_net = None
_error = ""


def available() -> bool:
    return ANTISPOOF_MODEL.exists()


def verify() -> tuple[bool, str]:
    """(ok, sha256) of the bundled model file."""
    if not available():
        return False, ""
    got = hashlib.sha256(ANTISPOOF_MODEL.read_bytes()).hexdigest()
    return got == SHA256, got


def _load():
    global _net, _error
    if _net is None and not _error:
        try:
            ok, got = verify()
            if not ok:
                _error = "model missing" if not got else f"checksum mismatch {got[:12]}"
            else:
                _net = cv2.dnn.readNetFromONNX(str(ANTISPOOF_MODEL))
        except Exception as e:           # fail open: a broken model disables the voter
            _error = f"load failed: {str(e).splitlines()[0][:80]}"
    return _net


def _crop(img: np.ndarray, box) -> np.ndarray | None:
    """Reference crop (yakhyo/minivision `crop_face`): scale the box, clamp, resize to 80×80."""
    H, W = img.shape[:2]
    bw, bh = max(1, int(box.w)), max(1, int(box.h))
    if min(bw, bh) < MIN_FACE_PX:
        return None
    s = min((H - 1) / bh, (W - 1) / bw, SCALE)
    nw, nh = bw * s, bh * s
    cx, cy = int(box.x) + bw / 2, int(box.y) + bh / 2
    x1, y1 = max(0, int(cx - nw / 2)), max(0, int(cy - nh / 2))
    x2, y2 = min(W - 1, int(cx + nw / 2)), min(H - 1, int(cy + nh / 2))
    if x2 - x1 < MIN_FACE_PX or y2 - y1 < MIN_FACE_PX:
        return None
    return cv2.resize(img[y1:y2 + 1, x1:x2 + 1], (INPUT, INPUT))


def is_grey(rgb: np.ndarray) -> bool:
    if rgb.ndim < 3 or rgb.shape[2] < 3:
        return True
    a = rgb[::4, ::4].astype(np.int16)
    return float(np.mean(np.abs(a[..., 0] - a[..., 1])) + np.mean(np.abs(a[..., 1] - a[..., 2]))) < GREY_TOL


def score(rgb: np.ndarray, box, net=None) -> float | None:
    """p(real) for one RGB-ordered frame and face box, or None when it cannot be scored."""
    net = net if net is not None else _load()
    if net is None or rgb is None or box is None or is_grey(rgb):
        return None
    crop = _crop(rgb, box)
    if crop is None:
        return None
    blob = cv2.dnn.blobFromImage(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))   # 1×3×80×80 float, 0–255, BGR
    net.setInput(blob)
    out = np.asarray(net.forward(), dtype=np.float64).reshape(-1)
    if out.size < 2:
        return None
    p = np.exp(out - out.max()); p /= p.sum()
    return float(p[REAL] if out.size > REAL else p[-1])


def decide(scores: list[float]) -> Vote:
    if not scores:
        return ABSTAIN(NAME, "no scorable rgb face")
    med = float(np.median(scores))
    if med >= PASS_P:
        return PASS(NAME, f"real {med:.2f}", med)
    if len(scores) >= FAIL_MIN_FRAMES and max(scores) <= FAIL_P:
        return FAIL(NAME, f"spoof p_real≤{max(scores):.3f} on {len(scores)} frames", max(scores))
    return ABSTAIN(NAME, f"uncertain {med:.2f}", med)


def vote_frames(frames: list[tuple[np.ndarray | None, object, bool]]) -> Vote:
    """frames = [(rgb, box, dark)] oldest→newest; the newest MAX_FRAMES lit faces are scored."""
    net = _load()
    if net is None:
        return ABSTAIN(NAME, _error or "model not installed")
    lit = [(f, b) for f, b, dark in frames if f is not None and b is not None and not dark]
    if not lit:
        return ABSTAIN(NAME, "rgb dark" if frames and any(d for _, _, d in frames) else "no rgb face")
    scores = []
    for f, b in lit[-MAX_FRAMES:]:
        try:
            s = score(f, b, net)
        except Exception:
            s = None
        if s is not None:
            scores.append(s)
    return decide(scores)


def vote(rgb: np.ndarray | None, box, rgb_dark: bool) -> Vote:
    """Single-frame form (kept for callers with one frame): can PASS, never reaches FAIL_MIN_FRAMES."""
    return vote_frames([(rgb, box, rgb_dark)])
