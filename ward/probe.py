"""Measurements: lighting on the camera, detector speed, environment sanity."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

from . import howdy


@dataclass
class Lighting:
    mean: float
    p5: float
    p95: float
    frames: int
    face_mean: float | None = None      # brightness of the detected face (None = no face seen)

    @property
    def verdict(self) -> str:
        if self.frames == 0:
            return "no frames"
        if self.mean < 25:
            return "dark"
        if self.mean < 50:
            return "dim"
        if self.mean > 200:
            return "blown out"
        return "good"


def lighting(device: str, frames: int = 25) -> Lighting:
    """Grab frames, let auto-exposure settle, report brightness of the last one."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return Lighting(0, 0, 0, 0)
    idx = int(device.replace("/dev/video", "")) if device.startswith("/dev/video") else device
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    last = None
    n = 0
    for _ in range(frames):
        ok, f = cap.read()
        if ok:
            last, n = f, n + 1
    cap.release()
    if last is None:
        return Lighting(0, 0, 0, 0)
    g = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY) if last.ndim == 3 else last
    L = Lighting(float(g.mean()), float(np.percentile(g, 5)), float(np.percentile(g, 95)), n)
    try:                                   # the face, if any: an IR emitter lights the face and leaves the room black
        from .engine.detect import Models
        from .engine import brightness
        box = Models("yunet", 320).detect(g)
        if box is not None:
            L.face_mean = brightness.meter(g, box).mean
    except Exception:
        pass
    return L


def grab_gray(device: str):
    import cv2
    idx = int(device.replace("/dev/video", ""))
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    f = None
    for _ in range(15):
        ok, fr = cap.read()
        if ok:
            f = fr
    cap.release()
    if f is None:
        return None
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f


@dataclass
class Bench:
    hog_ms: float
    cnn_ms: dict[int, float]   # max_height -> ms
    faces_hog: int
    faces_cnn: int


def benchmark(device: str, heights=(320, 240, 180)) -> Bench | None:
    """Time Howdy's two detectors on a real frame from this camera, at Howdy's max_height values."""
    try:
        import cv2
        import dlib
    except ImportError:
        return None
    g = grab_gray(device)
    if g is None:
        return None
    hog = dlib.get_frontal_face_detector()
    t = time.perf_counter()
    faces_hog = len(hog(g, 1))
    hog_ms = (time.perf_counter() - t) * 1000
    cnn_ms: dict[int, float] = {}
    faces_cnn = 0
    model = howdy.dlib_model("mmod_human_face_detector.dat")
    if model:
        cnn = dlib.cnn_face_detection_model_v1(str(model))
        for h in heights:
            scale = h / g.shape[0]
            img = cv2.resize(g, (int(g.shape[1] * scale), h)) if scale < 1 else g
            cnn(img, 0)  # warm-up
            t = time.perf_counter()
            r = cnn(img, 0)
            cnn_ms[h] = (time.perf_counter() - t) * 1000
            faces_cnn = max(faces_cnn, len(r))
    return Bench(hog_ms, cnn_ms, faces_hog, faces_cnn)


def python_shadowed() -> tuple[str, bool]:
    """Howdy's PAM module execs plain `python3` from PATH. Return (path, dlib_ok)."""
    py = shutil.which("python3") or sys.executable
    try:
        ok = subprocess.run([py, "-c", "import dlib, cv2"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    return py, ok


def emitter_status() -> dict:
    """linux-enable-ir-emitter: installed? service enabled? a device configured?"""
    tool = shutil.which("linux-enable-ir-emitter")
    st = {"installed": tool is not None, "enabled": False, "configured": None}
    if not tool:
        return st
    try:
        st["enabled"] = subprocess.run(["systemctl", "is-enabled", "linux-enable-ir-emitter"],
                                       capture_output=True, text=True).stdout.strip() == "enabled"
    except OSError:
        pass
    from .engine import emitter
    state = emitter.tool_state()
    st["state"] = state                    # working | none (files, but no instruction marked start) | absent
    if os.access(str(emitter.TOOL_DIR), os.R_OK):
        st["configured"] = state == "working"
    return st


def is_root() -> bool:
    return os.geteuid() == 0
