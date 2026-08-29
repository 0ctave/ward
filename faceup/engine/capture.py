"""Synchronized capture from the IR and/or RGB camera (DESIGN.md §6.2).

Frames are delivered as `Frame` objects carrying both a greyscale and an RGB view,
because dlib's detector wants grey/RGB and its recogniser wants RGB.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import cv2
import numpy as np

CAPTURE_WIDTH = 640   # RGB cameras are asked for this size; IR streams are usually already small
SYNC_TOLERANCE_MS = 40
LOOPBACK_DRIVERS = ("v4l2 loopback", "v4l2loopback", "vivid")


@dataclass
class Frame:
    spectrum: str        # "ir" | "rgb"
    t: float             # monotonic seconds
    gray: np.ndarray     # HxW uint8
    rgb: np.ndarray      # HxWx3 uint8 (grey replicated for IR)


class CaptureError(Exception):
    pass


def driver_name(device: str) -> str:
    """Kernel driver behind /dev/videoN, to refuse virtual devices (§2)."""
    n = os.path.basename(device)
    try:
        real = os.path.realpath(f"/sys/class/video4linux/{n}/device")
        # USB: .../usb1/1-1/1-1:1.0 ; the 'driver' symlink names the driver (uvcvideo)
        drv = os.path.basename(os.path.realpath(os.path.join(real, "driver")))
        name = open(f"/sys/class/video4linux/{n}/name").read().strip()
        return f"{drv}:{name}"
    except OSError:
        return "?"


def is_physical(device: str) -> bool:
    d = driver_name(device).lower()
    return not any(x in d for x in LOOPBACK_DRIVERS)


class Capture:
    def __init__(self, devices: dict[str, str]):
        """devices: {"ir": "/dev/video2", "rgb": "/dev/video0"} — missing/empty entries are skipped."""
        self.caps: dict[str, cv2.VideoCapture] = {}
        for spectrum, path in devices.items():
            if not path:
                continue
            if not os.path.exists(path):
                raise CaptureError(f"{spectrum} camera {path} does not exist")
            if not is_physical(path):
                raise CaptureError(f"{path} is a virtual video device; refusing")
            idx = int(path.replace("/dev/video", ""))
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if not cap.isOpened():
                raise CaptureError(f"cannot open {spectrum} camera {path} (busy?)")
            if spectrum == "rgb":
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_WIDTH * 9 // 16)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.caps[spectrum] = cap
        if not self.caps:
            raise CaptureError("no camera configured")

    @property
    def spectra(self) -> list[str]:
        return list(self.caps)

    def read_set(self) -> dict[str, Frame]:
        """One frame from every open camera, taken back-to-back (< SYNC_TOLERANCE_MS apart)."""
        out: dict[str, Frame] = {}
        for spectrum, cap in self.caps.items():
            ok, img = cap.read()
            t = time.monotonic()
            if not ok or img is None:
                continue
            if img.ndim == 2:
                gray = img
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if spectrum == "ir":
                rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            out[spectrum] = Frame(spectrum, t, gray, rgb)
        if len(out) == 2:
            ts = [f.t for f in out.values()]
            if (max(ts) - min(ts)) * 1000 > SYNC_TOLERANCE_MS:
                # keep the newer one only; the pair is not simultaneous
                older = min(out, key=lambda s: out[s].t)
                del out[older]
        return out

    def read_one(self, spectrum: str, flush: int = 1) -> Frame | None:
        """A fresh frame from one camera. With BUFFERSIZE=1 the first read returns the frame
        queued since the last read (possibly old); `flush` extra reads skip it."""
        cap = self.caps.get(spectrum)
        if cap is None:
            return None
        for _ in range(flush):
            cap.grab()
        ok, img = cap.read()
        if not ok or img is None:
            return None
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if spectrum == "ir":
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return Frame(spectrum, time.monotonic(), gray, rgb)

    def warm_up(self, n: int = 3) -> None:
        for _ in range(n):
            for cap in self.caps.values():
                cap.read()

    def close(self) -> None:
        for cap in self.caps.values():
            cap.release()
        self.caps.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
