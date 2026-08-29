"""Enumerate V4L2 cameras and tell IR sensors apart from RGB ones.

Uses raw ioctls so it works without v4l-utils. An infrared camera exposes an
8-bit greyscale stream (fourcc GREY); a colour camera exposes MJPG/YUYV/etc.
"""
from __future__ import annotations

import ctypes
import fcntl
import glob
import os
from dataclasses import dataclass, field

VIDIOC_ENUM_FMT = 0xC0405602
VIDIOC_ENUM_FRAMESIZES = 0xC02C564A
V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_FRMSIZE_TYPE_DISCRETE = 1


class _FmtDesc(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32), ("type", ctypes.c_uint32), ("flags", ctypes.c_uint32),
        ("description", ctypes.c_char * 32), ("pixelformat", ctypes.c_uint32),
        ("mbus_code", ctypes.c_uint32), ("reserved", ctypes.c_uint32 * 3),
    ]


class _FrmSize(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32), ("pixel_format", ctypes.c_uint32), ("type", ctypes.c_uint32),
        ("width", ctypes.c_uint32), ("height", ctypes.c_uint32), ("reserved", ctypes.c_uint32 * 7),
    ]


@dataclass
class Camera:
    path: str
    name: str
    formats: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    vid: str = ""          # USB vendor id (hex), "" for non-USB
    pid: str = ""
    serial: str = ""
    iface: str = ""        # USB interface number: one physical module can carry several cameras
    bus: str = ""          # e.g. "5-1": port path, changes if the module is plugged elsewhere

    @property
    def identity(self) -> str:
        """Stable across reboots and re-enumeration: usb:<vid>:<pid>:<iface>. Not the /dev path."""
        if self.vid:
            return f"usb:{self.vid}:{self.pid}:{self.iface or '00'}"
        return f"name:{self.name}"

    @property
    def role(self) -> str:
        return "ir" if self.is_ir else "rgb"

    @property
    def is_ir(self) -> bool:
        return bool(self.formats) and set(self.formats) <= {"GREY", "Y8  ", "Y10 ", "Y12 ", "Y16 "}

    @property
    def max_size(self) -> tuple[int, int]:
        sizes = [s for v in self.formats.values() for s in v]
        return max(sizes, key=lambda s: s[0] * s[1]) if sizes else (0, 0)

    def describe(self) -> str:
        kind = "IR (greyscale)" if self.is_ir else "RGB"
        w, h = self.max_size
        fmts = "/".join(f.strip() for f in self.formats)
        return f"{self.path}  {kind}  {w}x{h}  [{fmts}]  {self.name}"


def _formats(path: str) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = {}
    try:
        fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return out
    try:
        i = 0
        while True:
            f = _FmtDesc(index=i, type=V4L2_BUF_TYPE_VIDEO_CAPTURE)
            try:
                fcntl.ioctl(fd, VIDIOC_ENUM_FMT, f)
            except OSError:
                break
            fourcc = f.pixelformat.to_bytes(4, "little").decode("ascii", "replace")
            sizes: list[tuple[int, int]] = []
            j = 0
            while True:
                s = _FrmSize(index=j, pixel_format=f.pixelformat)
                try:
                    fcntl.ioctl(fd, VIDIOC_ENUM_FRAMESIZES, s)
                except OSError:
                    break
                if s.type == V4L2_FRMSIZE_TYPE_DISCRETE:
                    sizes.append((s.width, s.height))
                j += 1
            out[fourcc] = sizes
            i += 1
    finally:
        os.close(fd)
    return out


def _read(p: str) -> str:
    try:
        return open(p).read().strip()
    except OSError:
        return ""


def _usb_info(node: str) -> dict[str, str]:
    """USB ids for /sys/class/video4linux/<node>: the interface dir, then its parent device."""
    dev = os.path.realpath(f"/sys/class/video4linux/{node}/device")
    usb = os.path.realpath(os.path.join(dev, ".."))
    info = {"iface": _read(os.path.join(dev, "bInterfaceNumber")),
            "vid": _read(os.path.join(usb, "idVendor")), "pid": _read(os.path.join(usb, "idProduct")),
            "serial": _read(os.path.join(usb, "serial")), "bus": os.path.basename(usb) if _read(os.path.join(usb, "idVendor")) else ""}
    return info


def list_cameras() -> list[Camera]:
    """Capture-capable video nodes only (metadata nodes have no formats)."""
    cams = []
    for path in sorted(glob.glob("/dev/video*"), key=lambda p: int(p[10:] or 0)):
        n = os.path.basename(path)
        name = _read(f"/sys/class/video4linux/{n}/name") or "?"
        fmts = _formats(path)
        if fmts:
            cams.append(Camera(path, name, fmts, **_usb_info(n)))
    return cams


def resolve(spec: str, role: str, cams: list[Camera] | None = None) -> tuple[str, str]:
    """Turn a config spec into today's /dev path.
    spec: "usb:<vid>:<pid>:<iface>" (stable identity, what setup writes) · "auto" (pick by role) ·
    "/dev/videoN" (legacy path; honoured while it exists and has the right role, else re-resolved by role).
    Returns (path, note) — path "" if nothing suitable is connected; note explains any re-resolution."""
    if not spec:
        return "", ""
    cams = list_cameras() if cams is None else cams
    same_role = [c for c in cams if c.role == role]
    if spec.startswith("usb:") or spec.startswith("name:"):
        hit = [c for c in same_role if c.identity == spec]
        if hit:
            return hit[0].path, ""
        return "", f"{role} camera {spec} is not connected"
    if spec.startswith("/dev/"):
        hit = [c for c in cams if c.path == spec]
        if hit and hit[0].role == role:
            return spec, ""
        if same_role:
            return same_role[0].path, f"{role} camera moved: {spec} → {same_role[0].path} ({same_role[0].identity}); run `sudo faceup setup` to store the stable id"
        return "", f"{role} camera {spec} not found and no other {role} camera connected"
    if spec == "auto":
        return (same_role[0].path, "") if same_role else ("", f"no {role} camera connected")
    return "", f"unknown camera spec {spec!r}"


def pick_default(cams: list[Camera]) -> Camera | None:
    """Prefer an IR camera; otherwise the highest-resolution RGB one."""
    ir = [c for c in cams if c.is_ir]
    if ir:
        return max(ir, key=lambda c: c.max_size[0] * c.max_size[1])
    return max(cams, key=lambda c: c.max_size[0] * c.max_size[1]) if cams else None
