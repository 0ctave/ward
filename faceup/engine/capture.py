"""Synchronized capture from the IR and/or RGB camera (DESIGN.md §6.2).

Frames are delivered as `Frame` objects carrying both a greyscale and an RGB view,
because dlib's detector wants grey/RGB and its recogniser wants RGB.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

CAPTURE_WIDTH = 640   # RGB cameras are asked for this size; IR streams are usually already small
TARGET_FPS = 30       # ask every camera for its fastest common mode (IR modules default to 15)
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


def _to_frame(spectrum: str, img: np.ndarray) -> Frame:
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if spectrum == "ir":
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return Frame(spectrum, time.monotonic(), gray, rgb)


class _Reader(threading.Thread):
    """Drains one camera continuously so callers always get the newest frame without blocking
    on the driver. Decoding happens here too, off the processing thread."""

    def __init__(self, spectrum: str, cap: cv2.VideoCapture):
        super().__init__(daemon=True, name=f"faceup-{spectrum}")
        self.spectrum, self.cap = spectrum, cap
        self.cond = threading.Condition()
        self.frame: Frame | None = None
        self.seq = 0
        self.stop = False

    def run(self) -> None:
        while not self.stop:
            ok, img = self.cap.read()
            if not ok or img is None:
                time.sleep(0.005)
                continue
            fr = _to_frame(self.spectrum, img)
            with self.cond:
                self.frame, self.seq = fr, self.seq + 1
                self.cond.notify_all()

    def latest(self, after_seq: int = -1, timeout: float = 0.2) -> tuple[Frame | None, int]:
        """Newest frame; if after_seq is given, wait (bounded) for a frame newer than it."""
        with self.cond:
            if after_seq >= 0 and self.seq <= after_seq:
                self.cond.wait_for(lambda: self.seq > after_seq or self.stop, timeout)
            return self.frame, self.seq


class Capture:
    def __init__(self, devices: dict[str, str]):
        """devices: {"ir": "/dev/video2", "rgb": "/dev/video0"} — missing/empty entries are skipped."""
        self.caps: dict[str, cv2.VideoCapture] = {}
        self.readers: dict[str, _Reader] = {}
        self._seen: dict[str, int] = {}
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
            # NOTE: never set CAP_PROP_BUFFERSIZE=1 here — on UVC IR modules it halves the
            # frame rate (measured 29.7 → 15.3 fps). The reader thread keeps frames fresh instead.
            cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
            self.caps[spectrum] = cap
        if not self.caps:
            raise CaptureError("no camera configured")
        for spectrum, cap in self.caps.items():
            r = _Reader(spectrum, cap)
            r.start()
            self.readers[spectrum] = r
            self._seen[spectrum] = -1

    @property
    def spectra(self) -> list[str]:
        return list(self.caps)

    def read_set(self) -> dict[str, Frame]:
        """The newest frame of every camera (waits for one fresh frame of the first camera),
        dropped to the newer one if they are more than SYNC_TOLERANCE_MS apart."""
        out: dict[str, Frame] = {}
        first = True
        for spectrum, r in self.readers.items():
            fr, seq = r.latest(self._seen[spectrum] if first else -1)
            first = False
            if fr is None:
                continue
            self._seen[spectrum] = seq
            out[spectrum] = fr
        if len(out) == 2:
            ts = [f.t for f in out.values()]
            if (max(ts) - min(ts)) * 1000 > SYNC_TOLERANCE_MS:
                # keep the newer one only; the pair is not simultaneous
                older = min(out, key=lambda s: out[s].t)
                del out[older]
        return out

    def read_one(self, spectrum: str, flush: int = 1) -> Frame | None:
        """One camera. flush=0: newest frame not yet returned to this caller (waits for it);
        flush>=1: wait for a frame captured from now on (fresh, e.g. after a pause)."""
        r = self.readers.get(spectrum)
        if r is None:
            return None
        if flush:
            _, seq = r.latest()
            fr, seq = r.latest(seq)
        else:
            fr, seq = r.latest(self._seen[spectrum])
        self._seen[spectrum] = seq
        return fr

    def warm_up(self, n: int = 3) -> None:
        """Wait until every camera has delivered n frames."""
        for spectrum, r in self.readers.items():
            deadline = time.monotonic() + 2.0
            while r.seq < n and time.monotonic() < deadline:
                r.latest(r.seq, 0.1)

    def close(self) -> None:
        for r in self.readers.values():
            r.stop = True
        for r in self.readers.values():
            r.join(timeout=1.0)
        for cap in self.caps.values():
            cap.release()
        self.caps.clear(); self.readers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
