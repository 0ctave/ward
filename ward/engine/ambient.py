"""Room light from the laptop's ambient light sensor, and the dark-room camera rule.

Measured on the reference module (2026-10-22): in a dark room, opening the RGB camera kills
the IR stream — the IR node keeps delivering a flat pedestal frame (mean ~24, std < 1), the LED
goes out, and closing RGB does not revive it (the IR node must be reopened, ~1.4 s). With IR
alone the face is lit and the daylight-enrolled models match (nearest distance 0.31). In the
dark the RGB camera sees nothing anyway, so the rule is: below DARK_LUX, open IR only.
The ALS is a plain sysfs read (no root, ~0 ms); without one, `Capture`'s starvation watchdog
catches the same case after a few frames.
"""
from __future__ import annotations

import os
from pathlib import Path

DARK_LUX = 5.0           # below this the room is dark: RGB is useless and may starve the IR stream
STARVED_STD = 1.0        # a live IR frame always has texture; a starved node delivers a flat pedestal (measured std 0.3–0.7; a live frame ≥ 9, the first frame after open ~1.5)


def iio_dir() -> Path:
    return Path(os.environ.get("WARD_IIO", "/sys/bus/iio/devices"))


def _read(p: Path) -> float | None:
    try:
        return float(p.read_text().strip())
    except (OSError, ValueError):
        return None


def lux() -> float | None:
    """Brightest ambient light sensor reading in lux, or None when there is no usable sensor.
    (Several ALS nodes may exist — e.g. one per lid side — so take the brightest.) A raw count
    without a scale is not lux: such a node is ignored rather than compared against DARK_LUX."""
    override = os.environ.get("WARD_LUX")
    if override:
        try:
            return float(override)
        except ValueError:
            pass
    best = None
    try:
        devs = sorted(iio_dir().iterdir())
    except OSError:
        return None
    for d in devs:
        val = None
        for f in sorted(d.glob("in_illuminance*_input")):
            val = _read(f)
            if val is not None:
                break
        if val is None:
            for f in sorted(d.glob("in_illuminance*_raw")):
                raw = _read(f)
                stem = f.name[:-len("_raw")]
                scale = _read(d / f"{stem}_scale") or _read(d / "in_illuminance_scale")
                if raw is None or scale is None:
                    continue
                off = _read(d / f"{stem}_offset") or _read(d / "in_illuminance_offset") or 0.0
                val = (raw + off) * scale
                break
        if val is not None:
            best = val if best is None else max(best, val)
    return best


def usb_device(dev: str, sysfs: str = "/sys/class/video4linux") -> str:
    """The USB device (not interface) behind /dev/videoN: '…/usb5/5-1' for '…/5-1:1.2'."""
    try:
        return os.path.dirname(os.path.realpath(os.path.join(sysfs, os.path.basename(dev), "device")))
    except OSError:
        return ""


def same_module(a: str, b: str) -> bool:
    """Both nodes on one physical camera module (a Windows-Hello module exposes RGB and IR as two
    interfaces of one USB device). Only there has RGB been seen to starve IR in the dark."""
    ua, ub = usb_device(a), usb_device(b)
    return bool(ua) and ua == ub and "/usb" in ua


def for_light(devices: dict[str, str], threshold: float = DARK_LUX) -> tuple[dict[str, str], str]:
    """The cameras worth opening in this light: IR alone when the room is dark (see module doc).
    Returns (devices, note); note is '' when nothing was dropped."""
    if not (devices.get("ir") and devices.get("rgb")):
        return devices, ""
    if not same_module(devices["ir"], devices["rgb"]):
        return devices, ""                 # separate cameras: RGB cannot starve IR, and may still help
    v = lux()
    if v is not None and v < threshold:
        return {s: p for s, p in devices.items() if s != "rgb"}, f"dark room ({v:.1f} lux): IR only"
    return devices, ""


def starved(gray) -> bool:
    """A flat frame: the IR node is delivering no image (RGB opened in the dark)."""
    return float(gray.std()) < STARVED_STD
