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


GREY_FORMATS = {"GREY", "Y8  ", "Y10 ", "Y12 ", "Y16 ", "Y10B", "Y16B"}
IR_NAME_HINTS = ("ir cam", "ir camera", "infrared", " ir ", "ircam", "ir-", "depth")
GENERIC_SERIALS = {"", "0", "01.00.00", "00000000", "1", "n/a", "none", "default"}


@dataclass
class Camera:
    path: str
    name: str
    formats: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    vid: str = ""          # USB vendor id (hex), "" for non-USB
    pid: str = ""
    serial: str = ""
    iface: str = ""        # USB interface number: one physical module can carry several cameras
    bus: str = ""          # USB port path ("5-1") or PCI address / platform name for non-USB
    kind: str = "usb"      # usb | pci | platform | other
    driver: str = ""

    @property
    def module(self) -> str:
        """The physical device this sensor belongs to (both sensors of a Windows-Hello module share it)."""
        return f"{self.kind}:{self.vid}:{self.pid}:{self.bus}" if self.vid else f"{self.kind}:{self.bus}"

    @property
    def identity(self) -> str:
        """Stable across reboots and re-enumeration. USB: usb:<vid>:<pid>:<iface>[:<serial>], serial
        only when the device has a real one (identical webcams are told apart by it, else by port).
        PCI/platform: <kind>:<bus id>:<node name>. Never the /dev path."""
        if self.kind == "usb" and self.vid:
            base = f"usb:{self.vid}:{self.pid}:{self.iface or '00'}"
            return f"{base}:{self.serial}" if self.serial.lower() not in GENERIC_SERIALS else base
        return f"{self.kind}:{self.bus}:{self.name}"

    @property
    def role(self) -> str:
        return "ir" if self.is_ir else "rgb"

    @property
    def is_ir(self) -> bool:
        """Greyscale-only stream, or a greyscale stream on a device that names itself IR/infrared.
        Modules that expose IR as YUYV need the name hint or `cameras.force_roles`."""
        if not self.formats:
            return False
        fmts = set(self.formats)
        if fmts <= GREY_FORMATS:
            return True
        if fmts & GREY_FORMATS and not (fmts & {"MJPG"}):
            return True
        return any(h in f" {self.name.lower()} " for h in IR_NAME_HINTS)

    @property
    def usable(self) -> bool:
        return bool(self.formats)

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


def _bus_info(node: str) -> dict[str, str]:
    """Where /sys/class/video4linux/<node> hangs: USB interface → its device; else PCI or platform."""
    dev = os.path.realpath(f"/sys/class/video4linux/{node}/device")
    driver = os.path.basename(os.path.realpath(os.path.join(dev, "driver"))) if os.path.exists(os.path.join(dev, "driver")) else ""
    usb = os.path.realpath(os.path.join(dev, ".."))
    if _read(os.path.join(usb, "idVendor")):
        return {"kind": "usb", "iface": _read(os.path.join(dev, "bInterfaceNumber")),
                "vid": _read(os.path.join(usb, "idVendor")), "pid": _read(os.path.join(usb, "idProduct")),
                "serial": _read(os.path.join(usb, "serial")), "bus": os.path.basename(usb), "driver": driver}
    # walk up until a PCI device (has 'vendor' + 'device') or give up as platform
    p = dev
    for _ in range(6):
        if _read(os.path.join(p, "vendor")).startswith("0x") and _read(os.path.join(p, "device")).startswith("0x"):
            return {"kind": "pci", "vid": _read(os.path.join(p, "vendor"))[2:], "pid": _read(os.path.join(p, "device"))[2:],
                    "bus": os.path.basename(p), "driver": driver}
        p = os.path.dirname(p)
        if p in ("/sys/devices", "/sys", "/"):
            break
    return {"kind": "platform" if "/platform/" in dev else "other", "bus": os.path.basename(dev), "driver": driver}


@dataclass
class Unusable:
    """A video node the kernel exposes but V4L2 capture cannot use (metadata nodes are filtered
    out silently; these are reported by `doctor`)."""
    path: str
    name: str
    reason: str


def list_cameras(include_unusable: bool = False):
    """Capture-capable video nodes (metadata nodes have no formats). With include_unusable=True
    returns (cameras, unusable) where unusable lists nodes that cannot be opened or enumerate no
    formats — IPU6/libcamera-only sensors, permission problems, busy devices."""
    cams, bad = [], []
    for path in sorted(glob.glob("/dev/video*"), key=lambda p: int(p[10:] or 0)):
        n = os.path.basename(path)
        name = _read(f"/sys/class/video4linux/{n}/name") or "?"
        if not os.access(path, os.R_OK | os.W_OK):
            bad.append(Unusable(path, name, "no permission (add your user to the `video` group, or run as root)"))
            continue
        fmts = _formats(path)
        info = _bus_info(n)
        if fmts:
            cams.append(Camera(path, name, fmts, **info))
        else:
            # capture nodes without formats: either a metadata node (index odd, same name) or a
            # sensor behind a media controller (IPU6, rkisp…) that needs libcamera
            is_meta = _read(f"/sys/class/video4linux/{n}/index") not in ("", "0") or "metadata" in name.lower()
            if not is_meta and info.get("kind") in ("pci", "platform"):
                bad.append(Unusable(path, name, f"{info['kind']} camera ({info.get('driver') or 'no driver'}) exposes no V4L2 capture format — needs libcamera/PipeWire, not supported yet"))
    return (cams, bad) if include_unusable else cams


def resolve(spec: str, role: str, cams: list[Camera] | None = None, force: bool = False) -> tuple[str, str]:
    """Turn a config spec into today's /dev path.
    spec: "usb:<vid>:<pid>:<iface>" (stable identity, what setup writes) · "auto" (pick by role) ·
    "/dev/videoN" (legacy path; honoured while it exists and has the right role, else re-resolved by role).
    Returns (path, note) — path "" if nothing suitable is connected; note explains any re-resolution."""
    if not spec:
        return "", ""
    cams = list_cameras() if cams is None else cams
    # force=True: trust the config's role assignment (IR modules that stream IR as YUYV)
    same_role = list(cams) if force else [c for c in cams if c.role == role]
    if spec.startswith(("usb:", "pci:", "platform:", "other:", "name:")):
        base, _, port = spec.partition("@")
        hit = [c for c in same_role if c.identity == base]
        if not hit and base.startswith("usb:"):
            # an id written with a serial, device now reports none (or vice-versa): match on vid:pid:iface
            short = ":".join(base.split(":")[:4])
            hit = [c for c in same_role if ":".join(c.identity.split(":")[:4]) == short]
        if len(hit) > 1 and port:
            hit = [c for c in hit if c.bus == port] or hit
        if len(hit) > 1:
            return hit[0].path, f"{len(hit)} identical {role} cameras match {base}; using {hit[0].path} (port {hit[0].bus}) — pin one with {base}@{hit[0].bus}"
        if hit:
            return hit[0].path, ""
        return "", f"{role} camera {spec} is not connected"
    if spec.startswith("/dev/"):
        hit = [c for c in cams if c.path == spec]
        if hit and hit[0].role == role:
            return spec, ""
        if same_role:
            return same_role[0].path, f"{role} camera moved: {spec} → {same_role[0].path} ({same_role[0].identity}); run `sudo ward setup` to store the stable id"
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


def pick_pair(cams: list[Camera]) -> tuple[Camera | None, Camera | None]:
    """(ir, rgb) for setup. Built-in Windows-Hello modules expose both sensors on one physical
    device, so the RGB camera on the IR camera's module wins over an external webcam; otherwise
    the highest-resolution colour camera."""
    ir = [c for c in cams if c.is_ir]
    rgb = [c for c in cams if not c.is_ir]
    ir_cam = max(ir, key=lambda c: c.max_size[0] * c.max_size[1]) if ir else None
    if ir_cam:
        same = [c for c in rgb if c.module == ir_cam.module]
        if same:
            return ir_cam, max(same, key=lambda c: c.max_size[0] * c.max_size[1])
    return ir_cam, (max(rgb, key=lambda c: c.max_size[0] * c.max_size[1]) if rgb else None)


def spec_for(cam: Camera, cams: list[Camera]) -> str:
    """Config spec for a camera: its identity, plus @port when another connected camera shares it."""
    twins = [c for c in cams if c is not cam and c.identity == cam.identity]
    return f"{cam.identity}@{cam.bus}" if twins else cam.identity
