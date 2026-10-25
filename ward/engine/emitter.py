"""IR emitter control (DESIGN.md §7.3, roadmap L1): switch the LEDs from Python.

Windows-Hello modules expose the emitter through a UVC *extension unit* (XU) control —
vendor-specific bytes sent with the UVCIOC_CTRL_QUERY ioctl. linux-enable-ir-emitter finds the
right (unit, selector, bytes) by brute force and stores them in /etc/linux-enable-ir-emitter/
with `status: start`; the service replays them at boot. We reuse those instructions:

- `load_instructions()` parses the tool's files (a small YAML subset, parsed by hand: no
  dependency) and keeps the entries marked `start`.
- `XU` wraps the ioctl (ctypes) for GET_CUR / SET_CUR / GET_DEF / GET_MIN.
- `Emitter.on()` / `.off()`: "on" = the tool's bytes; "off" = what the control read *before* we
  switched it on, or the device default, or the minimum — whichever exists and differs from "on".
- `toggle_bench()` measures how fast the frame brightness follows a switch; the shadow-test
  voter (liveness/shadow.py) is only enabled once this proves < SHADOW_MAX_LATENCY_MS.

Everything here fails soft: no instruction, no permission, an ioctl error → `Emitter.available`
is False and callers ABSTAIN. The emitter is left ON when we are done (recognition needs it).
"""
from __future__ import annotations

import ctypes
import fcntl
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

TOOL_DIR = Path(os.environ.get("WARD_EMITTER_DIR", "/etc/linux-enable-ir-emitter"))
UVC_SET_CUR, UVC_GET_CUR, UVC_GET_MIN, UVC_GET_MAX, UVC_GET_LEN, UVC_GET_DEF = 0x01, 0x81, 0x82, 0x83, 0x85, 0x87
SHADOW_MAX_LATENCY_MS = 150      # slower than this and two settled frames no longer fit the auth window (reference module: 133 ms)
SHADOW_MIN_DELTA = 12.0          # mean IR brightness change the LEDs must make to be measurable


class _Query(ctypes.Structure):
    _fields_ = [("unit", ctypes.c_uint8), ("selector", ctypes.c_uint8), ("query", ctypes.c_uint8),
                ("size", ctypes.c_uint16), ("data", ctypes.POINTER(ctypes.c_uint8))]


def _ioc(direction: int, typ: str, nr: int, size: int) -> int:
    return (direction << 30) | (size << 16) | (ord(typ) << 8) | nr


UVCIOC_CTRL_QUERY = _ioc(3, "u", 0x21, ctypes.sizeof(_Query))   # _IOWR('u', 0x21, struct uvc_xu_control_query)


@dataclass
class Instruction:
    unit: int
    selector: int
    on: bytes
    minimum: bytes = b""
    maximum: bytes = b""
    status: str = "idle"


def parse_tool_file(text: str) -> list[Instruction]:
    """The tool writes a YAML list of mappings with scalar lists; this reads exactly that."""
    out: list[Instruction] = []
    cur: dict | None = None
    key = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        s = line.strip()
        if s.startswith("- ") and ":" in s and not s.startswith("- -"):
            if cur:
                out.append(_instruction(cur))
            cur = {}
            s = s[2:].strip()
        if cur is None:
            continue
        if s.startswith("- "):                       # list item under `key`
            if key is not None:
                cur.setdefault(key, []).append(int(s[2:].strip()))
            continue
        k, _, v = s.partition(":")
        k, v = k.strip(), v.strip()
        if v == "":
            key = k; cur[k] = []
        else:
            key = None; cur[k] = v
    if cur:
        out.append(_instruction(cur))
    return out


def _instruction(d: dict) -> Instruction:
    return Instruction(int(d.get("unit", 0)), int(d.get("selector", 0)), bytes(d.get("current", [])),
                       bytes(d.get("minimum", [])), bytes(d.get("maximum", [])), str(d.get("status", "idle")))


def load_instructions(tool_dir: Path = TOOL_DIR, only_start: bool = True) -> list[Instruction]:
    found: list[Instruction] = []
    try:
        files = sorted(p for p in tool_dir.iterdir() if p.is_file() and not p.name.startswith("."))
    except OSError:
        return found
    for p in files:
        try:
            found.extend(parse_tool_file(p.read_text()))
        except (OSError, ValueError):
            continue
    return [i for i in found if i.status == "start"] if only_start else found


def config_instruction(cfg: dict | None = None) -> Instruction | None:
    """ward's own instruction from config [emitter] (written by `ward emitter probe`)."""
    from . import config as cfgmod
    e = (cfg or cfgmod.load()).get("emitter", {})
    try:
        unit, sel = int(e.get("unit", 0)), int(e.get("selector", 0))
        on = bytes(int(x) for x in str(e.get("on", "")).split(",") if x.strip() != "")
        off = bytes(int(x) for x in str(e.get("off", "")).split(",") if x.strip() != "")
    except ValueError:
        return None
    if unit <= 0 or not on or not off or len(on) != len(off):
        return None
    ins = Instruction(unit, sel, on, status="start")
    ins.off = off
    return ins


def instructions_for(cfg: dict | None = None) -> list[Instruction]:
    """ward's probed control first, else the tool's `start` entries."""
    own = config_instruction(cfg)
    return [own] if own else load_instructions()


def tool_state(tool_dir: Path = TOOL_DIR) -> str:
    """'working' (an instruction marked start), 'none' (files but nothing works), 'absent' (no files)."""
    allofthem = load_instructions(tool_dir, only_start=False)
    if not allofthem:
        return "absent"
    return "working" if any(i.status == "start" for i in allofthem) else "none"


class XU:
    """One UVC extension-unit control on a /dev/videoN node."""

    def __init__(self, device: str):
        self.fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)

    def close(self) -> None:
        try:
            os.close(self.fd)
        except OSError:
            pass

    def _query(self, unit: int, selector: int, code: int, data: bytes | bytearray) -> bytes:
        buf = (ctypes.c_uint8 * len(data))(*data)
        q = _Query(unit, selector, code, len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8)))
        fcntl.ioctl(self.fd, UVCIOC_CTRL_QUERY, q)
        return bytes(buf)

    def get(self, unit: int, selector: int, size: int, code: int = UVC_GET_CUR) -> bytes:
        return self._query(unit, selector, code, bytearray(size))

    def set(self, unit: int, selector: int, value: bytes) -> None:
        self._query(unit, selector, UVC_SET_CUR, value)


class Emitter:
    def __init__(self, device: str, instructions: list[Instruction] | None = None, xu: XU | None = None):
        self.device = device
        self.instructions = instructions_for() if instructions is None else instructions
        self.xu = xu
        self.off_values: dict[tuple[int, int], bytes] = {}
        self.available = False
        self.error = ""
        if not self.instructions:
            self.error = "no working emitter instruction (linux-enable-ir-emitter found none)"
            return
        try:
            if self.xu is None:
                self.xu = XU(device)
            for ins in self.instructions:
                n = len(ins.on)
                off = getattr(ins, "off", b"")            # a probed instruction knows its OFF bytes
                cur = self.xu.get(ins.unit, ins.selector, n)
                if not off:
                    off = cur if cur != ins.on else b""
                if not off:
                    for code in (UVC_GET_DEF, UVC_GET_MIN):
                        try:
                            cand = self.xu.get(ins.unit, ins.selector, n, code)
                        except OSError:
                            continue
                        if cand != ins.on:
                            off = cand; break
                if not off and ins.minimum and ins.minimum != ins.on:
                    off = ins.minimum
                if not off:
                    raise OSError(f"cannot determine an OFF state for unit {ins.unit} selector {ins.selector}")
                self.off_values[(ins.unit, ins.selector)] = off
            self.available = True
        except OSError as e:
            self.error = str(e)

    def on(self) -> bool:
        return self._apply({(i.unit, i.selector): i.on for i in self.instructions})

    def off(self) -> bool:
        return self._apply(self.off_values)

    def _apply(self, values: dict[tuple[int, int], bytes]) -> bool:
        if not self.available or self.xu is None:
            return False
        try:
            for (unit, selector), value in values.items():
                self.xu.set(unit, selector, value)
            return True
        except OSError as e:
            self.error = str(e); self.available = False
            return False

    def close(self) -> None:
        if self.xu is not None:
            self.xu.close()


@dataclass
class BenchResult:
    on_mean: float = 0.0
    off_mean: float = 0.0
    latency_ms: int = -1          # frames→ms until brightness crossed the midpoint after a switch
    cycles: int = 0
    usable: bool = False
    reason: str = ""
    trace: list = field(default_factory=list)

    @property
    def delta(self) -> float:
        return self.on_mean - self.off_mean


def toggle_bench(emitter: Emitter, read_gray, cycles: int = 3, frames_per_state: int = 8, fps: float = 30.0) -> BenchResult:
    """Switch off/on `cycles` times, reading `frames_per_state` frames each time through
    `read_gray()` (→ HxW uint8 or None). Latency = frames from the switch until the mean
    crosses the midpoint between the two steady levels."""
    res = BenchResult()
    if not emitter.available:
        res.reason = emitter.error or "emitter unavailable"; return res
    trace: list[tuple[str, float]] = []
    for _ in range(cycles):
        for state in ("off", "on"):
            if not (emitter.off() if state == "off" else emitter.on()):
                res.reason = emitter.error; return res
            for _ in range(frames_per_state):
                g = read_gray()
                trace.append((state, float(g.mean()) if g is not None else float("nan")))
    emitter.on()
    res.trace = trace; res.cycles = cycles
    # segments of consecutive frames in one state; the settled level is the second half of each
    segments: list[tuple[str, list[float]]] = []
    for state, v in trace:
        if segments and segments[-1][0] == state:
            segments[-1][1].append(v)
        else:
            segments.append((state, [v]))
    def settled(state: str) -> list[float]:
        out = []
        for st, vals in segments:
            if st == state:
                out.extend(x for x in vals[len(vals) // 2:] if x == x)
        return out
    ons, offs = settled("on"), settled("off")
    if not ons or not offs:
        res.reason = "no frames"; return res
    res.on_mean, res.off_mean = sum(ons) / len(ons), sum(offs) / len(offs)
    if res.delta < SHADOW_MIN_DELTA:
        res.reason = f"LEDs change the frame by only {res.delta:.1f} (need ≥ {SHADOW_MIN_DELTA:.0f}): emitter off, or a bright room"
        return res
    mid = (res.on_mean + res.off_mean) / 2
    lat = []
    for state, vals in segments[1:]:                              # every switch after the first segment
        want_high = state == "on"
        k = next((i for i, x in enumerate(vals) if x == x and (x >= mid) == want_high), None)
        lat.append(len(vals) if k is None else k)                 # never followed → as slow as the segment
    if lat:
        res.latency_ms = int(round(max(lat) / fps * 1000))
    res.usable = 0 <= res.latency_ms <= SHADOW_MAX_LATENCY_MS
    res.reason = ("usable for the shadow test" if res.usable else
                  f"toggle takes {res.latency_ms} ms (limit {SHADOW_MAX_LATENCY_MS})" if res.latency_ms >= 0 else "brightness never followed the switch")
    return res


def shadow_pair(emitter: Emitter, read_gray, settle_frames: int = 2):
    """Grab one emitter-OFF and one emitter-ON frame for liveness/shadow.py, leaving the LEDs on.
    Returns (off_gray, on_gray) or (None, None)."""
    if not emitter.available:
        return None, None
    try:
        if not emitter.off():
            return None, None
        off = None
        for _ in range(settle_frames + 1):
            off = read_gray()
        emitter.on()
        on = None
        for _ in range(settle_frames + 1):
            on = read_gray()
        return off, on
    finally:
        emitter.on()


# candidate extension units / selectors to try when nothing is known (Windows-Hello modules use
# vendor XUs 4–7 on the IR interface; selectors 1–12); only controls whose GET_LEN is ≤ 2 bytes
PROBE_UNITS = (4, 5, 6, 7, 3, 2)
PROBE_SELECTORS = range(1, 13)


def probe(device: str, read_face, say=None, units=PROBE_UNITS, selectors=PROBE_SELECTORS,
          cycles: int = 2, frames_per_state: int = 4, fps: float = 30.0) -> tuple[Instruction | None, list]:
    """Find the LED control by measurement, in any lighting: for every small XU control, alternate
    its minimum and its current value every few frames (faster than the firmware auto-exposure)
    and watch the FACE brightness (`read_face()` → face crop or None). The control whose minimum
    darkens the face by the largest step is the emitter; its current value is ON and its
    minimum OFF. Every control is restored. Returns (instruction or None, [(unit, sel, delta_ms, latency)])."""
    xu = XU(device)
    results = []
    best = None
    try:
        for unit in units:
            for sel in selectors:
                try:
                    cur = xu.get(unit, sel, 2, UVC_GET_CUR)
                except OSError:
                    try:
                        cur = xu.get(unit, sel, 1, UVC_GET_CUR)
                    except OSError:
                        continue
                n = len(cur)
                try:
                    mn = xu.get(unit, sel, n, UVC_GET_MIN)
                except OSError:
                    mn = bytes(n)
                if mn == cur:
                    continue
                ins = Instruction(unit, sel, cur, minimum=mn, status="start"); ins.off = mn
                em = Emitter(device, [ins], xu=xu)
                if not em.available:
                    continue
                r = toggle_bench(em, read_face, cycles=cycles, frames_per_state=frames_per_state, fps=fps)
                try:
                    xu.set(unit, sel, cur)
                except OSError:
                    pass
                results.append((unit, sel, r.delta, r.latency_ms))
                if say:
                    say(f"unit {unit} sel {sel} {list(cur)}→{list(mn)}: face {r.off_mean:.0f}/{r.on_mean:.0f} delta {r.delta:+.0f} latency {r.latency_ms} ms")
                if r.delta >= SHADOW_MIN_DELTA and (best is None or r.delta > best[1].delta):
                    best = (ins, r)
    finally:
        xu.close()
    return (best[0] if best else None), results
