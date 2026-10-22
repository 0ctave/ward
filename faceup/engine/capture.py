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

from . import ambient, brightness

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
    if img.dtype != np.uint8:                      # Y16 / 10-bit IR nodes: scale to 8 bits for the detector
        img = cv2.convertScaleAbs(img, alpha=255.0 / max(1.0, float(img.max())))
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
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
            if (after_seq >= 0 and self.seq <= after_seq) or self.frame is None:
                self.cond.wait_for(lambda: (self.seq > after_seq and self.frame is not None) or self.stop, timeout)   # also: no frame yet
            return self.frame, self.seq


def _open_one(spectrum: str, path: str, retries: int = 2, retry_ms: int = 150) -> cv2.VideoCapture:
    """Open one V4L2 device. A busy device is retried briefly: the most common cause is the
    previous attempt (or a preview) still releasing it (roadmap R3)."""
    if not os.path.exists(path):
        raise CaptureError(f"{spectrum} camera {path} does not exist")
    if not is_physical(path):
        raise CaptureError(f"{path} is a virtual video device; refusing")
    idx = int(path.replace("/dev/video", ""))
    for attempt in range(retries + 1):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            break
        cap.release()
        if attempt < retries:
            time.sleep(retry_ms / 1000)
    else:
        raise CaptureError(f"cannot open {spectrum} camera {path} (busy?)")
    if spectrum == "rgb":
        # MJPG keeps 30 fps at any size; raw YUYV drops to 5–10 fps above VGA on USB 2
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_WIDTH * 9 // 16)
    # NOTE: never set CAP_PROP_BUFFERSIZE=1 here — on UVC IR modules it halves the
    # frame rate (measured 29.7 → 15.3 fps). The reader thread keeps frames fresh instead.
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    return cap


class Capture:
    def __init__(self, devices: dict[str, str], _opened: dict[str, cv2.VideoCapture] | None = None):
        """devices: {"ir": "/dev/video2", "rgb": "/dev/video0"} — missing/empty entries are skipped.
        The devices are opened in parallel (each V4L2 open blocks ~150–500 ms in the driver and
        releases the GIL, roadmap R1); `Capture.open_async` starts that while the caller loads models."""
        self.caps: dict[str, cv2.VideoCapture] = {}
        self.readers: dict[str, _Reader] = {}
        self._seen: dict[str, int] = {}
        wanted = {s: p for s, p in devices.items() if p}
        if not wanted:
            raise CaptureError("no camera configured")
        self.notes: list[str] = []
        if _opened is None:
            wanted, note = ambient.for_light(wanted)
            if note:
                self.notes.append(note)
        self.devices = {s: p for s, p in wanted.items() if _opened is None or s in _opened}
        self._starved = 0
        self.lost = ""                     # set when the watchdog could not reopen IR
        self.caps = _opened if _opened is not None else _open_all(wanted)
        self.exposure: dict[str, brightness.Exposure] = {}
        for spectrum, cap in self.caps.items():
            self.exposure[spectrum] = brightness.Exposure(cap, spectrum)
            r = _Reader(spectrum, cap)
            r.start()
            self.readers[spectrum] = r
            self._seen[spectrum] = -1

    @classmethod
    def open_async(cls, devices: dict[str, str]) -> "PendingCapture":
        return PendingCapture(devices)

    @property
    def spectra(self) -> list[str]:
        return list(self.caps)

    def read_set(self) -> dict[str, Frame]:
        """The newest frame of every camera (waits for one fresh frame of the first camera),
        dropped to the newer one if they are more than SYNC_TOLERANCE_MS apart."""
        if self.lost:
            raise CaptureError(self.lost)      # no camera left: the caller ends the attempt (fail open)
        out: dict[str, Frame] = {}
        first = True
        for spectrum, r in self.readers.items():
            fr, seq = r.latest(self._seen[spectrum] if first else -1)
            first = False
            if fr is None:
                continue
            self._seen[spectrum] = seq
            out[spectrum] = fr
        if "ir" in out and self._watch(out["ir"]):
            return {}
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
        if spectrum == "ir" and fr is not None and self._watch(fr):
            return None
        return fr

    STARVED_FRAMES = 6                 # ~200 ms: the first frame after open can be flat on a healthy stream

    def _watch(self, ir: Frame) -> bool:
        """IR starvation watchdog (see ambient.py): with RGB open in a dark room the IR node
        delivers flat frames and the LED goes out. After a few of them, drop RGB and reopen IR
        (closing RGB alone does not revive it). True when the fallback just happened — the
        caller's frame is stale and should be skipped."""
        if "rgb" not in self.caps:
            return False
        # only when RGB is dark too: a flat IR frame beside a usable RGB picture is an IR camera
        # with its LED off (or covered) — dropping RGB there would throw away the working camera
        rgb = self.readers["rgb"].frame if "rgb" in self.readers else None
        rgb_dark = rgb is None or float(rgb.gray.mean()) < brightness.FACE_DARK["rgb"]
        self._starved = self._starved + 1 if (ambient.starved(ir.gray) and rgb_dark) else 0
        if self._starved < self.STARVED_FRAMES:
            return False
        self._starved = 0
        try:
            self._release("rgb")
            self._release("ir")
            cap = _open_one("ir", self.devices["ir"])
        except Exception as e:                       # never throw from a read: record it instead
            self.lost = f"IR starved and could not be reopened: {e}"
            self.notes.append(self.lost)
            return True
        self.caps["ir"] = cap
        self.exposure["ir"] = brightness.Exposure(cap, "ir")
        r = _Reader("ir", cap); r.start()
        self.readers["ir"] = r; self._seen["ir"] = -1
        self.devices.pop("rgb", None)
        self.notes.append("IR starved while RGB was open (dark room): continuing IR only")
        return True

    def _release(self, spectrum: str) -> None:
        r = self.readers.pop(spectrum, None)
        if r is not None:
            r.stop = True; r.join(timeout=1.0)
        ctl = self.exposure.pop(spectrum, None)
        if ctl is not None:
            ctl.restore()
        cap = self.caps.pop(spectrum, None)
        if cap is not None:
            cap.release()
        self._seen.pop(spectrum, None)

    def meter(self, spectrum: str, gray: np.ndarray, box) -> brightness.FaceLight:
        """Meter the face and, if this camera accepts manual exposure, steer it toward the target."""
        light = brightness.meter(gray, box)
        ctl = self.exposure.get(spectrum)
        if ctl is not None:
            ctl.steer(light)
        return light

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
        for ctl in self.exposure.values():
            ctl.restore()                      # leave the camera as we found it
        for cap in self.caps.values():
            cap.release()
        self.caps.clear(); self.readers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _open_all(wanted: dict[str, str]) -> dict[str, cv2.VideoCapture]:
    """Open every device in its own thread; the first error wins and the others are released."""
    results: dict[str, cv2.VideoCapture] = {}
    errors: dict[str, CaptureError] = {}

    def work(spectrum: str, path: str) -> None:
        try:
            results[spectrum] = _open_one(spectrum, path)
        except CaptureError as e:
            errors[spectrum] = e

    threads = [threading.Thread(target=work, args=(s, p), name=f"faceup-open-{s}", daemon=True) for s, p in wanted.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        for cap in results.values():
            cap.release()
        raise errors[next(iter(errors))]
    return results


class PendingCapture:
    """A Capture being opened in the background. `.result()` blocks until the devices are open
    (or raises CaptureError) and returns the Capture with its reader threads running."""

    def __init__(self, devices: dict[str, str]):
        self.devices, note = ambient.for_light({s: p for s, p in devices.items() if p})
        self.notes = [note] if note else []
        self._caps: dict[str, cv2.VideoCapture] | None = None
        self._error: CaptureError | None = None
        self._thread = threading.Thread(target=self._run, name="faceup-open", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            if not self.devices:
                raise CaptureError("no camera configured")
            self._caps = _open_all(self.devices)
        except CaptureError as e:
            self._error = e

    def result(self, timeout: float | None = None) -> Capture:
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise CaptureError("camera open timed out")
        if self._error is not None:
            raise self._error
        cap = Capture(self.devices, _opened=self._caps)
        cap.notes[:0] = self.notes
        return cap

    def cancel(self) -> None:
        """Release the devices if nobody will use them."""
        self._thread.join()
        for cap in (self._caps or {}).values():
            cap.release()
