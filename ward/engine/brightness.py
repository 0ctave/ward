"""Brightness management (DESIGN.md §6.3a).

Three layers, applied in this order and all camera-agnostic:

1. **Metering on the face**, not the frame. Room light, a bright window or a dark desk must
   not decide whether a frame is usable — only the face region does.
2. **Exposure control** when the camera has it. Many UVC cameras expose manual exposure/gain
   (V4L2 `exposure_time_absolute`, `gain`); some (like the reference IR module) run
   auto-exposure in firmware and expose nothing. `Exposure` probes what is there and, when
   manual control works, steers exposure so the face lands in the target band. Otherwise it
   leaves auto-exposure alone.
3. **Normalisation in software** (CLAHE + gamma) for frames that are still dim or blown
   out. Deterministic from the frame's own statistics, so enrollment and authentication see
   the same transform.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

# face-region brightness targets (mean of the box), per spectrum
TARGET = {"ir": (70.0, 150.0), "rgb": (80.0, 170.0)}
FACE_DARK = {"ir": 28.0, "rgb": 35.0}       # below → the frame is "dark" even if the room isn't
FACE_BLOWN = 235.0
NORMALISE_BELOW = 60.0                      # frame/face mean below → CLAHE + gamma
NORMALISE_ABOVE = 190.0


@dataclass
class FaceLight:
    mean: float
    p5: float
    p95: float

    def state(self, spectrum: str) -> str:
        lo, hi = TARGET[spectrum]
        if self.mean < FACE_DARK[spectrum]:
            return "dark"
        if self.p95 >= FACE_BLOWN and self.mean > hi:
            return "blown"
        if self.mean < lo:
            return "dim"
        if self.mean > hi:
            return "bright"
        return "ok"


def meter(gray: np.ndarray, box) -> FaceLight:
    """Brightness of the inner face (skips the box border where hair/background leak in)."""
    h, w = gray.shape[:2]
    mx, my = int(box.w * 0.15), int(box.h * 0.15)
    y0, y1 = max(0, box.y + my), min(h, box.y + box.h - my)
    x0, x1 = max(0, box.x + mx), min(w, box.x + box.w - mx)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        roi = gray
    return FaceLight(float(roi.mean()), float(np.percentile(roi, 5)), float(np.percentile(roi, 95)))


_clahe = None


def normalise(gray: np.ndarray, mean: float | None = None) -> np.ndarray:
    """Contrast-normalise a dim or blown-out frame for detection/landmarks/embedding.
    Identity when the frame is already in range, so well-lit frames are untouched."""
    global _clahe
    m = float(gray.mean()) if mean is None else mean
    if NORMALISE_BELOW <= m <= NORMALISE_ABOVE:
        return gray
    if _clahe is None:
        _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out = _clahe.apply(gray)
    if m < NORMALISE_BELOW:                       # lift shadows: gamma < 1
        gamma = max(0.45, m / NORMALISE_BELOW)
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        out = cv2.LUT(out, lut)
    return out


class Exposure:
    """Manual exposure/gain control on a cv2.VideoCapture, if the camera supports it."""

    def __init__(self, cap: cv2.VideoCapture, spectrum: str):
        self.cap, self.spectrum = cap, spectrum
        self.manual = False
        self.exposure = None
        self.gain = None
        self.orig = {}
        self.min_exp, self.max_exp = 1.0, 10000.0
        self.last_step = 0.0
        self._probe()

    def _probe(self) -> None:
        c = self.cap
        try:
            auto = c.get(cv2.CAP_PROP_AUTO_EXPOSURE)
            exp = c.get(cv2.CAP_PROP_EXPOSURE)
            gain = c.get(cv2.CAP_PROP_GAIN)
        except cv2.error:
            return
        if exp in (-1.0, 0.0) and gain in (-1.0, 0.0):
            return                                 # nothing exposed (firmware auto-exposure only)
        self.orig = {"auto": auto, "exposure": exp, "gain": gain}
        # V4L2 via OpenCV: AUTO_EXPOSURE 1 = manual, 3 = aperture-priority auto
        if c.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) and exp > 0:
            probe = max(1.0, exp * 0.8)
            if c.set(cv2.CAP_PROP_EXPOSURE, probe) and abs(c.get(cv2.CAP_PROP_EXPOSURE) - probe) < max(2.0, probe * 0.2):
                self.manual = True          # supported — but stay on auto until steering is needed
                self.exposure = exp
                self.gain = gain if gain > 0 else None
        self.engaged = False
        self.restore()

    def steer(self, light: FaceLight) -> bool:
        """One proportional step toward the target band; at most 4 steps/s. Returns True if changed."""
        if not self.manual or self.exposure is None:
            return False
        now = time.monotonic()
        if now - self.last_step < 0.25:
            return False
        lo, hi = TARGET[self.spectrum]
        target = (lo + hi) / 2
        if lo <= light.mean <= hi:
            return False
        if not self.engaged:                        # auto-exposure could not get the face in band: take over
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            cur = self.cap.get(cv2.CAP_PROP_EXPOSURE)
            self.exposure = cur if cur > 0 else self.exposure
            self.engaged = True
        ratio = target / max(light.mean, 1.0)
        ratio = min(1.6, max(0.6, ratio))            # bounded steps: no oscillation on fast changes
        new = min(self.max_exp, max(self.min_exp, self.exposure * ratio))
        if abs(new - self.exposure) < 0.5:
            return False
        if self.cap.set(cv2.CAP_PROP_EXPOSURE, new):
            self.exposure = new
            self.last_step = now
            return True
        return False

    def restore(self) -> None:
        self.engaged = False
        if not self.orig:
            return
        try:
            if self.orig.get("auto") not in (None, -1.0):
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, self.orig["auto"])
            if self.orig.get("exposure", -1.0) > 0:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, self.orig["exposure"])
            if self.orig.get("gain", -1.0) > 0:
                self.cap.set(cv2.CAP_PROP_GAIN, self.orig["gain"])
        except cv2.error:
            pass

    def describe(self) -> str:
        if self.manual:
            return f"auto-exposure, manual control available (ward takes over when the face is out of range)"
        if self.orig:
            return "auto-exposure (controls present, manual mode not accepted)"
        return "auto-exposure in firmware (no controls)"


def normalise_rgb(rgb: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Same normalisation for a colour frame: applied to luminance only, colours kept."""
    m = float(gray.mean())
    if NORMALISE_BELOW <= m <= NORMALISE_ABOVE:
        return rgb
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    ycc[:, :, 0] = normalise(ycc[:, :, 0], m)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)
