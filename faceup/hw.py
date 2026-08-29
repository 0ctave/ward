"""Hardware profile: what this machine offers and how faceup uses it (DESIGN.md §3a)."""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field

import cv2


@dataclass
class Profile:
    cpu: str = ""
    cores: int = 1
    avx2: bool = False
    opencl: bool = False
    opencl_device: str = ""
    dlib_avx: bool = False
    opencv: str = ""
    threads: int = 1
    power_profile: str = ""
    emitter_tool: bool = False
    notes: list[str] = field(default_factory=list)


def _cpuinfo() -> tuple[str, bool]:
    model, flags = "", ""
    try:
        for ln in open("/proc/cpuinfo"):
            if ln.startswith("model name") and not model:
                model = ln.split(":", 1)[1].strip()
            elif ln.startswith("flags") and not flags:
                flags = ln.split(":", 1)[1]
    except OSError:
        pass
    return model, " avx2 " in f" {flags} "


def profile() -> Profile:
    p = Profile()
    p.cpu, p.avx2 = _cpuinfo()
    p.cores = os.cpu_count() or 1
    p.opencv = cv2.__version__
    try:
        p.opencl = bool(cv2.ocl.haveOpenCL())
        if p.opencl:
            cv2.ocl.setUseOpenCL(True)
            dev = cv2.ocl.Device_getDefault()
            p.opencl_device = dev.name() if dev else ""
            cv2.ocl.setUseOpenCL(False)
    except cv2.error:
        p.opencl = False
    try:
        import dlib
        p.dlib_avx = bool(getattr(dlib, "DLIB_USE_AVX", False)) if hasattr(dlib, "DLIB_USE_AVX" ) else False
    except ImportError:
        pass
    # OpenCV's own thread pool: enough for DNN + image ops, leave cores for dlib and the UI
    p.threads = max(1, min(4, p.cores // 2))
    try:
        p.power_profile = subprocess.run(["powerprofilesctl", "get"], capture_output=True, text=True, timeout=3).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    p.emitter_tool = shutil.which("linux-enable-ir-emitter") is not None
    if not p.avx2:
        p.notes.append("CPU without AVX2: dlib and OpenCV DNN run their slow paths")
    if p.power_profile == "power-saver":
        p.notes.append("power-saver profile: CPU throttled, everything ~1.5–2× slower")
    return p


def apply(p: Profile | None = None) -> Profile:
    """Configure OpenCV for this machine (thread count; OpenCL stays opt-in via config)."""
    p = p or profile()
    cv2.setNumThreads(p.threads)
    return p


def dnn_target(accel: str, p: Profile | None = None) -> tuple[int, int]:
    """(backend, target) for OpenCV DNN models. 'opencl' only if the runtime is actually there."""
    p = p or profile()
    if accel == "opencl" and p.opencl:
        return cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_OPENCL
    return cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU
