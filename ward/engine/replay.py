"""Replay harness (roadmap L7): run the engine on recorded frame sets instead of cameras.

`ward lab --record-frames DIR` writes every frame set the lab sees as
    DIR/000123_ir.png, DIR/000123_rgb.png   (greyscale / BGR)
    DIR/index.json                          ([{"n": 123, "t": 1.234, "ir": true, "rgb": true}, …])
`ReplayCapture(DIR)` then serves them with the Capture interface (`read_set`, `read_one`,
`meter`, `warm_up`, `close`, `spectra`), at recorded pace or as fast as possible. The engine
uses it when WARD_REPLAY is set, so a recorded night, a phone replay or a stranger becomes a
regression test that needs no hardware — and CI can run the whole pipeline.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from . import brightness
from .capture import CaptureError, Frame


class Recorder:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index: list[dict] = []
        self.n = 0
        self.t0 = time.monotonic()

    def write(self, frames: dict[str, Frame]) -> None:
        entry = {"n": self.n, "t": round(time.monotonic() - self.t0, 4)}
        for s, fr in frames.items():
            img = fr.gray if s == "ir" else cv2.cvtColor(fr.rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(self.dir / f"{self.n:06d}_{s}.png"), img)
            entry[s] = True
        self.index.append(entry)
        self.n += 1
        if self.n % 10 == 0:
            self.flush()

    def flush(self) -> None:
        (self.dir / "index.json").write_text(json.dumps(self.index))

    def close(self) -> None:
        self.flush()


class ReplayCapture:
    """Serves recorded frame sets. `realtime=False` returns sets as fast as the caller asks."""

    def __init__(self, directory: str | Path, realtime: bool = False, loop: bool = False):
        self.dir = Path(directory)
        try:
            self.index = json.loads((self.dir / "index.json").read_text())
        except (OSError, ValueError) as e:
            raise CaptureError(f"replay: cannot read {self.dir / 'index.json'}: {e}")
        if not self.index:
            raise CaptureError("replay: empty recording")
        self.realtime, self.loop = realtime, loop
        self.pos = 0
        self.t0 = time.monotonic()
        self._spectra = sorted({s for e in self.index for s in ("ir", "rgb") if e.get(s)})
        self.exposure: dict = {}

    @property
    def spectra(self) -> list[str]:
        return list(self._spectra)

    def _load(self, entry: dict) -> dict[str, Frame]:
        out = {}
        for s in ("ir", "rgb"):
            if not entry.get(s):
                continue
            p = self.dir / f"{entry['n']:06d}_{s}.png"
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE if s == "ir" else cv2.IMREAD_COLOR)
            if img is None:
                continue
            gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out[s] = Frame(s, time.monotonic(), gray, rgb)
        return out

    def read_set(self) -> dict[str, Frame]:
        if self.pos >= len(self.index):
            if not self.loop:
                time.sleep(0.01)              # like a camera with no new frame yet
                return {}
            self.pos = 0
        entry = self.index[self.pos]
        if self.realtime:
            due = self.t0 + float(entry.get("t", 0))
            now = time.monotonic()
            if due > now:
                time.sleep(due - now)
        self.pos += 1
        return self._load(entry)

    def read_one(self, spectrum: str, flush: int = 1) -> Frame | None:
        for _ in range(len(self.index)):
            fs = self.read_set()
            if spectrum in fs:
                return fs[spectrum]
            if not fs:
                break
        return None

    def meter(self, spectrum: str, gray: np.ndarray, box) -> brightness.FaceLight:
        return brightness.meter(gray, box)

    def warm_up(self, n: int = 3) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def exhausted(self) -> bool:
        return self.pos >= len(self.index) and not self.loop


class PendingReplay:
    """Mimics capture.PendingCapture for Engine.authenticate."""

    def __init__(self, directory: str | Path):
        self.directory = directory

    def result(self, timeout: float | None = None) -> ReplayCapture:
        return ReplayCapture(self.directory)

    def cancel(self) -> None:
        pass


def replay_dir() -> str | None:
    return os.environ.get("WARD_REPLAY") or None
