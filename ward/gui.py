"""Camera windows (OpenCV HighGUI, no extra toolkit): free-move enrollment and the emitter check.

Both run as root (like `howdy test`) and open a window on the user's display; if HighGUI
cannot open a window (no DISPLAY, headless), callers fall back to the text flows.
"""
from __future__ import annotations

import os

import queue
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import display as disp
from .engine import brightness
from .engine import pose as posemod
from .engine.capture import Capture
from .engine.detect import Box, Models, fuse
from .engine.enroll import MIN_ENROLL_FACE_PX
from .engine.quality import measure
from .engine.store import Store

WIN = "ward"
SAMPLES_PER_CELL = 3
SAMPLES_PER_POSE = 6          # guided window: frames per pose model, spread over the pass (review A9: 3 → p95 0.34, 6 → 0.23)
CELL_DONE = 2                 # samples of the primary spectrum that mark a cell as filled
SAMPLE_EVERY_MS = 120         # embeddings are computed at most this often
MAX_TRACKS = 900              # landmark frames kept for the post-calibration (≈30 s)
COL_TXT = (235, 235, 235)


GuiUnavailable = disp.DisplayUnavailable


class _Profile:
    """Per-stage timing printed once a second when enrollment runs with --profile."""

    def __init__(self, enabled: bool, cap: Capture):
        self.enabled, self.cap = enabled, cap
        self.t = {}
        self.n = 0
        self.t0 = time.monotonic()
        self.seq0 = {sp: r.seq for sp, r in cap.readers.items()}
        self._mark = None
        if enabled:
            import os, subprocess
            prof = subprocess.run(["powerprofilesctl", "get"], capture_output=True, text=True).stdout.strip() or "?"
            print(f"[profile] QT_QPA_PLATFORM={os.environ.get('QT_QPA_PLATFORM', '(unset)')} DISPLAY={os.environ.get('DISPLAY', '')} "
                  f"WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY', '')} power-profile={prof} euid={os.geteuid()}", flush=True)
            if prof == "power-saver":
                print("[profile] WARNING: power-saver profile — CPU throttled; plug in or `powerprofilesctl set performance` for a fair measurement", flush=True)

    def start(self) -> None:
        self._mark = time.perf_counter()

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.t[name] = self.t.get(name, 0.0) + (now - self._mark) * 1000
        self._mark = now

    def frame(self) -> None:
        self.n += 1
        dt = time.monotonic() - self.t0
        if self.enabled and dt >= 1.0:
            cam = " ".join(f"{sp}:{(r.seq - self.seq0[sp]) / dt:.0f}fps" for sp, r in self.cap.readers.items())
            stages = " ".join(f"{k}={v / self.n:.1f}" for k, v in self.t.items())
            print(f"[profile] ui {self.n / dt:.1f} fps | camera {cam} | ms/frame: {stages}", flush=True)
            self.t, self.n, self.t0 = {}, 0, time.monotonic()
            self.seq0 = {sp: r.seq for sp, r in self.cap.readers.items()}


def _canvas_of(frame) -> np.ndarray:
    return frame.rgb.copy()


def _text(img, s, xy, scale=0.6, color=COL_TXT, thick=1):
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


# ------------------------------------------------------------ enrollment ----
@dataclass
class Cell:
    samples: dict[str, list[tuple[float, np.ndarray]]] = field(default_factory=dict)   # spectrum -> [(quality, emb)]
    poses: list[tuple[float, float]] = field(default_factory=list)
    n_primary: int = 0          # samples taken of the primary spectrum (drives "filled")

    def count(self, spectrum: str) -> int:
        return self.n_primary if spectrum == "primary" else len(self.samples.get(spectrum, []))

    @property
    def full(self) -> bool:
        return self.n_primary >= SAMPLES_PER_CELL

    def push(self, spectrum: str, q: float, emb: np.ndarray, aug: list | None = None) -> None:
        lst = self.samples.setdefault(spectrum, [])
        lst.append((q, emb, aug or []))
        lst.sort(key=lambda t: t[0], reverse=True)
        del lst[SAMPLES_PER_CELL:]


_LUT_BRIGHT = np.array([255 * (i / 255) ** 0.45 for i in range(256)], np.uint8)


def _light_variants(rgb: np.ndarray) -> list[np.ndarray]:
    """Synthetic illumination variants of a face frame: night (dark + sensor noise), bright,
    side-lit. Deterministic, cheap; embeddings of these enter the whitening scatter only."""
    rng = np.random.default_rng(int(rgb[::16, ::16].sum()) & 0xFFFF)
    night = np.clip(rgb.astype(np.float32) * 0.35 + rng.normal(0, 6, rgb.shape), 0, 255).astype(np.uint8)
    bright = cv2.LUT(rgb, _LUT_BRIGHT)
    ramp = np.linspace(0.4, 1.0, rgb.shape[1], dtype=np.float32)[None, :, None]
    side = np.clip(rgb * ramp, 0, 255).astype(np.uint8)
    return [night, bright, side]


def _sampler_main(req, res, detector_kind: str, max_height: int, recogniser: str = "dlib") -> None:
    """Worker PROCESS: dlib's Python bindings hold the GIL during detection and embedding, so a
    thread would freeze the preview (measured: 30 → 14 fps). A separate process keeps the UI
    at camera rate; frames travel over a pipe (~0.7 MB per sample, a few times per second)."""
    models = Models("yunet", max_height, recogniser=recogniser)   # fast detector here: this must keep up with the preview
    models.preload()
    while True:
        job = req.get()
        if job is None:
            break
        cell, pose, items = job
        results = []
        for item in items:
            try:                                          # one bad frame must not kill the worker (review A3)
                sp, rgb, hint, q, exact = item[:5]
                want_aug = item[5] if len(item) > 5 else True
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                box = Box(*hint)
                if not exact:
                    box = models.detect(gray, box)
                if box is None or box.w < MIN_ENROLL_FACE_PX:
                    continue
                emb = models.embed(rgb, box)
                # lighting variants of the same frame: not match targets, but they teach the
                # whitening which embedding directions are "just lighting" for this user
                aug = [models.embed(v, box) for v in _light_variants(rgb)] if want_aug else []
                results.append((sp, q, emb, aug))
            except Exception:
                continue
        res.put((cell, pose, results))


class _Sampler:
    def __init__(self, models: Models):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.req = ctx.Queue(maxsize=2)
        self.res = ctx.Queue()
        # never let interpreter exit block on a queue feeder thread if the worker died
        self.req.cancel_join_thread(); self.res.cancel_join_thread()
        self.proc = ctx.Process(target=_sampler_main, args=(self.req, self.res, models.detector_kind, models.max_height, models.recogniser_kind), daemon=True)
        self.proc.start()
        self.pending = 0

    @property
    def alive(self) -> bool:
        return self.proc.is_alive()

    @property
    def busy(self) -> bool:
        return self.pending >= 1 or not self.proc.is_alive()

    def submit(self, cell, pose, items) -> bool:
        try:
            self.req.put_nowait((cell, pose, items))
        except queue.Full:
            return False
        self.pending += 1
        return True

    def drain(self) -> list:
        out = []
        while True:
            try:
                out.append(self.res.get_nowait())
            except queue.Empty:
                break
        self.pending = max(0, self.pending - len(out))
        return out

    def close(self, timeout: float = 3.0) -> list:
        """Wait for in-flight jobs, collect their results, stop the process."""
        t_end = time.monotonic() + timeout
        out = []
        while self.pending > 0 and self.proc.is_alive() and time.monotonic() < t_end:
            got = self.drain()
            out += got
            if not got:
                time.sleep(0.02)
        try:
            self.req.put_nowait(None)
        except queue.Full:
            pass
        self.proc.join(timeout=1.0)
        if self.proc.is_alive():
            self.proc.terminate()
        return out


PANEL_H = 110


def to_display(pose, sgn: float):
    """Camera-convention pose → what the user sees. The preview is mirrored, so the user's left is
    the image's left; camera yaw is negative when the nose moves camera-left, i.e. when the user
    turns to THEIR right — hence display yaw = −camera yaw when mirrored. Applied exactly once,
    here; the coverage map, the target, the hints and the coverage view all live in display space."""
    if pose is None:
        return None
    return posemod.Pose(sgn * pose.yaw, pose.pitch, sgn * pose.roll)


def _coverage_panel(width: int, cov, current, target, sgn: float = 1.0) -> np.ndarray:
    """Heatmap of (yaw, pitch) coverage: the required ellipse outlined, covered areas green,
    uncovered required areas orange, the current pose as a dot, the next target as a ring."""
    m = np.minimum(cov.map / cov.FULL, 1.0)                          # rows = pitch (low→high), cols = yaw
    img = np.zeros((m.shape[0], m.shape[1], 3), np.uint8)
    img[..., 1] = (60 + 130 * m).astype(np.uint8)                     # green channel = coverage
    img[..., 0] = (40 + 30 * (1 - m)).astype(np.uint8)
    img[..., 2] = (40 + 30 * (1 - m)).astype(np.uint8)
    req = cov.required
    img[req & (m < 1.0)] = (np.array([[255, 140, 60]]) * (1 - m[req & (m < 1.0)])[:, None]
                            + np.array([[90, 190, 90]]) * m[req & (m < 1.0)][:, None]).astype(np.uint8)
    img[~req] = (img[~req] * 0.45).astype(np.uint8)
    img = cv2.flip(img, 0)                                            # pitch "up" at the top
    if sgn < 0:
        img = cv2.flip(img, 1)                                        # mirror: user's left on the left
    panel = cv2.resize(img, (width, PANEL_H), interpolation=cv2.INTER_LINEAR)

    def to_px(yaw, pitch):
        x = (yaw - posemod.YAW_EDGES[0]) / (posemod.YAW_EDGES[-1] - posemod.YAW_EDGES[0]) * width
        y = (posemod.PITCH_EDGES[-1] - pitch) / (posemod.PITCH_EDGES[-1] - posemod.PITCH_EDGES[0]) * PANEL_H
        return int(np.clip(x, 4, width - 4)), int(np.clip(y, 4, PANEL_H - 4))

    # ellipse outline of the (adaptive) required region, centre mirrored with everything else
    cx, cy = to_px(sgn * cov.centre[0], cov.centre[1])
    ax = int(cov.axes[0] / (posemod.YAW_EDGES[-1] - posemod.YAW_EDGES[0]) * width)
    ay = int(cov.axes[1] / (posemod.PITCH_EDGES[-1] - posemod.PITCH_EDGES[0]) * PANEL_H)
    cv2.ellipse(panel, (cx, cy), (ax, ay), 0, 0, 360, (230, 230, 230), 1, cv2.LINE_AA)
    _text(panel, "left", (6, PANEL_H // 2 + 4), 0.4, (200, 200, 200)); _text(panel, "right", (width - 42, PANEL_H // 2 + 4), 0.4, (200, 200, 200))
    _text(panel, "up", (width // 2 - 10, 14), 0.4, (200, 200, 200)); _text(panel, "down", (width // 2 - 18, PANEL_H - 6), 0.4, (200, 200, 200))
    if target is not None:
        tx, ty = to_px(*target)
        cv2.circle(panel, (tx, ty), 9, (0, 0, 0), 2, cv2.LINE_AA); cv2.circle(panel, (tx, ty), 9, COL_CUR, 1, cv2.LINE_AA)
    if current is not None:
        px, py = to_px(current.yaw, current.pitch)
        # dot with a short "roll" tick so head tilt is visible too
        cv2.circle(panel, (px, py), 6, (0, 0, 0), -1, cv2.LINE_AA); cv2.circle(panel, (px, py), 4, COL_CUR, -1, cv2.LINE_AA)
        r = np.deg2rad(current.roll)
        cv2.line(panel, (px, py), (int(px + 12 * np.cos(r)), int(py + 12 * np.sin(r))), COL_CUR, 2, cv2.LINE_AA)
    return panel


def _render_enroll(snap: dict, sgn: float) -> np.ndarray:
    """Fallback rendering (OpenCV HighGUI / plain GTK picture): the snapshot painted onto a canvas."""
    img = snap["frame"].copy()
    h, w = img.shape[:2]
    if snap["box"] is not None:
        x, y, bw, bh = snap["box"]
        cv2.rectangle(img, (x, y), (x + bw, y + bh), COL_OK if snap.get("sampled") else COL_CUR, 2)
    if snap["phase"] == "calibrating":
        _text(img, f"Look at the MIDDLE of the screen (not the camera)... calibrating {snap['calib_progress'] * 100:.0f}%", (10, 24), 0.62, COL_CUR)
    else:
        _text(img, snap["hint"], (10, 24), 0.62, (255, 120, 90) if snap["light"] == "dark" else COL_TXT)
    status = (f"coverage {snap['fraction'] * 100:3.0f}%  {snap['n_models']} poses  {snap['spectrum'].upper()} {snap['ir_mean']:.0f}"
              f"{(' face ' + snap['light']) if snap['light'] else ''}  {snap['fps']:.0f} fps  {snap['gate']}")
    if snap["pose"]:
        status += f"  yaw {snap['pose'][0]:+.0f} pitch {snap['pose'][1]:+.0f} roll {snap['pose'][2]:+.0f}"
    _text(img, status, (10, h - 12), 0.5, (200, 200, 200))
    _text(img, "Enter = finish   Esc = cancel", (w - 235, 24), 0.45, (180, 180, 180))
    cur = posemod.Pose(*snap["pose"]) if snap["pose"] else None
    return np.vstack([img, _coverage_panel(w, snap["coverage"], cur, snap["target"], sgn)])


def _hint(current, target, box=None, calibrating=False) -> str:
    if target is None:
        return "Coverage complete - press Enter to finish"
    if current is None:
        if box is not None and box.w < MIN_ENROLL_FACE_PX:
            return f"Come a bit closer - face is {box.w} px, need {MIN_ENROLL_FACE_PX}"
        return "Face the camera so it can find you"
    ty, tp = target
    parts = []
    if abs(ty - current.yaw) > 5:
        parts.append("turn " + ("left" if ty < current.yaw else "right"))
    if abs(tp - current.pitch) > 5:
        parts.append("look " + ("down" if tp < current.pitch else "up"))
    return ("Slowly " + " and ".join(parts) + " toward the blinking dot") if parts else "Hold still..."


def free_move_enroll(st: Store, cap: Capture, models: Models, cams: dict, mirror: bool = True, profile: bool = False) -> int:
    """Track the head, fill the pose grid, fuse per cell and per spectrum. Returns models added.

    Per-frame cost is kept to: one camera read + ROI face detection (~5 ms) + 68 landmarks
    (~5 ms) + drawing. Embeddings (the expensive part) are computed only when the current
    cell still needs samples, at most every SAMPLE_EVERY_MS, and the second camera is read
    only at those moments.
    """
    models.sp68; models.preload()                           # preload: no stall on the first face
    primary = "ir" if "ir" in cap.spectra else "rgb"
    cells: dict[tuple[int, int], Cell] = {}
    others = [sp for sp in cap.spectra if sp != primary]
    sampler = _Sampler(models)
    restarts = 0
    prev_pose = None

    def absorb(results) -> None:
        for cell, pose, items in results:
            c = cells.setdefault(cell, Cell())
            for item in items:
                sp, qq, emb = item[0], item[1], item[2]
                c.push(sp, qq, emb, item[3] if len(item) > 3 else None)
            c.poses.append(pose)
    prev_box: Box | None = None
    last_sample_t = 0.0
    finished = False
    fps_t, fps_n, fps = time.monotonic(), 0, 0.0
    prof = _Profile(profile, cap)
    state = {"finished": False, "cancel": False, "fps_t": fps_t, "fps_n": 0, "fps": 0.0,
             "prev_pose": None, "prev_box": None, "last_sample_t": 0.0,
             "sampler": sampler, "restarts": 0,
             "filter": posemod.PoseFilter(), "neutral": None, "neutral_samples": [], "neutral_t0": None,
             "tracks": [], "fast_poses": [], "shape3d": (np.array(st.calib["shape3d"]) if st.calib.get("shape3d") else None),
             "coverage": posemod.Coverage(), "t_complete": None,
             "lmfilter": posemod.LandmarkFilter(), "stability": posemod.Stability(),
             "neutral_ratio_samples": [], "neutral_ratios": None}
    # Mirror convention for everything the user sees: the preview is mirrored, so "left" on the
    # panel and in the hints must be the USER's left. Camera yaw is negative when the nose moves
    # camera-left, i.e. when the user turns to THEIR right → display yaw = −camera yaw.
    sgn = -1.0 if mirror else 1.0
    NEUTRAL_SAMPLES = 30          # ~1 s of face at 30 fps
    probe = cap.read_one(primary, flush=0)
    ph, pw = (probe.gray.shape[:2] if probe is not None else (360, 640))
    try:
        screen = disp.open_display(WIN, pw, ph, kind="enroll", sgn=1.0, canvas=(pw, ph + PANEL_H))   # snapshot is already display space
    except disp.DisplayUnavailable as e:
        sampler.close()
        raise GuiUnavailable(str(e))
    if profile:
        print(f"[profile] display backend: {screen.name} ({screen.backend})", flush=True)

    def step():
        sampler = state["sampler"]
        for k in screen.keys():
            if k == disp.KEY_ENTER:
                state["finished"] = True
            elif k == disp.KEY_ESC:
                state["cancel"] = True
        if state["finished"] or state["cancel"]:
            return None
        prof.start()
        fr = cap.read_one(primary, flush=0)
        prof.lap("wait")
        if fr is None:
            return np.zeros((ph, pw, 3), np.uint8)
        h, w = fr.gray.shape[:2]
        q = measure(fr.gray, models.max_height)
        gray_n = brightness.normalise(fr.gray, q.mean)
        box = None if (q.sharpness < 5.0 and q.mean < cams[f"{primary}_dark_threshold"]) else models.detect(gray_n, state["prev_box"])
        state["prev_box"] = box
        light = cap.meter(primary, fr.gray, box) if box is not None else None
        light_state = light.state(primary) if light else ""
        prof.lap("detect")
        current = None
        sampled = False
        prev_pose = state["prev_pose"]
        calibrating = state["neutral"] is None
        if box is not None and box.w >= MIN_ENROLL_FACE_PX and light_state != "dark":
            now = time.monotonic()
            sbox = state["lmfilter"].smooth_box(box, now)
            lm68 = state["lmfilter"](models.landmarks68(gray_n, sbox), now)
            raw = posemod.estimate(lm68, w, h, shape3d=state["shape3d"], neutral=state.get("neutral_ratios"))
            if raw is not None:
                if len(state["tracks"]) < MAX_TRACKS:
                    state["tracks"].append(lm68.astype(np.float32))
                    state["fast_poses"].append((raw.yaw, raw.pitch))
                filt = state["filter"](raw, now)
                if calibrating:
                    # Neutral pose = the user looking at the screen naturally. Generic path: the
                    # user's own landmark RATIOS become the zero point (the inversion then stays in
                    # its linear region for every face). Personal-model path: angle offsets.
                    state["neutral_samples"].append((filt.yaw, filt.pitch))
                    state["neutral_ratio_samples"].append(posemod.ratios(lm68, filt.roll))
                    if len(state["neutral_samples"]) >= NEUTRAL_SAMPLES:
                        ys = sorted(a for a, _ in state["neutral_samples"]); ps = sorted(b for _, b in state["neutral_samples"])
                        state["neutral"] = (ys[len(ys) // 2], ps[len(ps) // 2])
                        keys = set().union(*[set(r) for r in state["neutral_ratio_samples"]])
                        med = {k: float(np.median([r[k] for r in state["neutral_ratio_samples"] if k in r])) for k in keys}
                        state["neutral_ratios"] = posemod.Neutral(med)
                        st.calib["neutral_ratios"] = {k: round(v, 5) for k, v in med.items()}
                        st.calib["pose_offset"] = [round(state["neutral"][0], 1), round(state["neutral"][1], 1)]
                        state["filter"].reset(); state["stability"].reset()
                    current = posemod.Pose(0.0, 0.0, filt.roll)      # dot sits in the centre while calibrating
                elif state["shape3d"] is not None:
                    ny, npch = state["neutral"]
                    current = posemod.Pose(filt.yaw - ny, filt.pitch - npch, filt.roll)
                else:
                    current = filt                                     # already relative: estimated with the neutral ratios
                stable = state["stability"].push(current)
                state["prev_pose"] = current
                if not calibrating:
                    d = to_display(current, sgn)
                    state["coverage"].observe(d.yaw, d.pitch)
                cell = current.cell() or posemod.nearest_cell(current.yaw, current.pitch)
                cov = state["coverage"]
                dcur = to_display(current, sgn)                       # coverage lives in display space
                covered = cov.at(dcur.yaw, dcur.pitch) >= cov.FULL
                worker_ok = sampler.alive
                busy = sampler.busy if worker_ok else False
                gate = ("calibrating" if calibrating else "moving" if not stable else "covered here" if covered
                        else "worker busy" if busy else "")
                state["gate"] = gate
                if not gate and (now - state["last_sample_t"]) * 1000 >= SAMPLE_EVERY_MS:
                    c = cells.setdefault(cell, Cell())
                    if True:
                        items = [(primary, brightness.embed_rgb(fr.rgb, fr.gray, primary, q.mean), (box.x, box.y, box.w, box.h), q.score(), True)]
                        for sp in others:
                            f2 = cap.read_one(sp, flush=0)
                            if f2 is None:
                                continue
                            q2 = measure(f2.gray, models.max_height)
                            if q2.reason(cams[f"{sp}_dark_threshold"]):
                                continue
                            sx, sy = f2.gray.shape[1] / w, f2.gray.shape[0] / h
                            items.append((sp, brightness.embed_rgb(f2.rgb, f2.gray, sp, q2.mean), (int(box.x * sx), int(box.y * sy), int(box.w * sx), int(box.h * sy)), q2.score(), False))
                        if worker_ok:
                            ok = sampler.submit(cell, (current.yaw, current.pitch), items)
                        else:
                            # worker process unavailable: sample inline (slower preview, but it works)
                            if not state.get("inline_warned"):
                                print("[ward] sampling worker unavailable — computing embeddings inline (preview will be slower)", flush=True)
                                state["inline_warned"] = True
                            res = []
                            for sp, rgb, hint, qq, exact in items:
                                gg = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                                b = Box(*hint) if exact else models.detect(gg, Box(*hint))
                                if b is not None and b.w >= MIN_ENROLL_FACE_PX:
                                    res.append((sp, qq, models.embed(rgb, b)))
                            absorb([(cell, (current.yaw, current.pitch), res)])
                            ok = True
                        if ok:
                            c.n_primary += 1
                            cov.add(dcur.yaw, dcur.pitch)
                            state["last_sample_t"] = now
                            sampled = True
        else:
            state["prev_pose"] = None
            state["gate"] = "no face" if box is None else ("too dark" if light_state == "dark" else "too far")
            state["filter"].reset(); state["lmfilter"].reset(); state["stability"].reset()
        prof.lap("pose+sample")
        absorb(sampler.drain())
        if not sampler.alive and state["restarts"] < 2:
            state["restarts"] += 1
            print(f"[ward] sampling worker died (exit {sampler.proc.exitcode}); restarting", flush=True)
            state["sampler"] = _Sampler(models)

        cov = state["coverage"]
        disp_current = to_display(current, sgn)
        target = None if calibrating else cov.target(disp_current.yaw if disp_current else None, disp_current.pitch if disp_current else None)
        frac = cov.fraction()
        disp_target = target                                            # already display space
        if frac >= 0.97 and state["t_complete"] is None:
            state["t_complete"] = time.monotonic()

        state["fps_n"] += 1; state["frames_total"] = state.get("frames_total", 0) + 1
        if time.monotonic() - state["fps_t"] >= 1.0:
            state["fps"] = state["fps_n"] / (time.monotonic() - state["fps_t"]); state["fps_n"] = 0; state["fps_t"] = time.monotonic()
        img = _canvas_of(fr)
        if mirror:
            img = cv2.flip(img, 1)
        n_models = sum(1 for c in cells.values() if c.count("primary") >= CELL_DONE)
        n_cal = len(state["neutral_samples"])
        snap = {
            "frame": img,
            "box": ((w - box.x - box.w) if mirror else box.x, box.y, box.w, box.h) if box is not None else None,
            "pose": (disp_current.yaw, disp_current.pitch, disp_current.roll) if disp_current else None,
            "target": disp_target,
            "coverage": cov,
            "fraction": frac,
            "hint": ("Face too dark - is the IR emitter on? (turn on a light to continue)" if light_state == "dark"
                     else _hint(disp_current, disp_target, box)),
            "phase": "done" if state["t_complete"] else ("calibrating" if calibrating else "enrolling"),
            "calib_progress": min(1.0, n_cal / NEUTRAL_SAMPLES),
            "light": light_state,
            "fps": state["fps"],
            "n_models": n_models,
            "gate": state.get("gate", "") or "sampling",
            "ir_mean": q.mean,
            "sampled": sampled,
            "spectrum": primary,
        }
        out = snap if screen.snapshot else _render_enroll(snap, 1.0)   # everything in snap is display space
        prof.lap("draw")
        prof.frame()
        if state["t_complete"] and time.monotonic() - state["t_complete"] > 3:
            state["finished"] = True
        return out

    try:
        screen.run(step)
    finally:
        absorb(state["sampler"].close())
    finished = state["finished"]

    # ---- post-calibration: personal 3D model from the recorded landmark tracks (DESIGN §8.6)
    if finished and len(state["tracks"]) >= 40:
        from .engine import sfm
        tracks = np.array(state["tracks"], dtype=np.float64)
        try:
            cal = sfm.calibrate(tracks, pw, ph, state["fast_poses"])
            st.calib["shape3d"] = [[round(float(v), 2) for v in row] for row in cal.shape3d]
            st.calib["pose_report"] = cal.report
            np.savez_compressed(str(st.path.with_suffix(".tracks.npz")), tracks=tracks.astype(np.float32),
                                fast=np.array(state["fast_poses"], dtype=np.float32), size=np.array([pw, ph]))
        except sfm.CalibrationError as e:
            st.calib["pose_report"] = {"error": str(e)}

    if not finished and not cells:
        return 0
    added = 0
    extra: dict[str, list] = {}
    for cell, c in cells.items():
        if c.count("primary") < CELL_DONE:
            continue
        label = posemod.cell_name(cell)
        st.remove_label(label)
        mean_pose = (float(np.mean([p[0] for p in c.poses])), float(np.mean([p[1] for p in c.poses])))
        for sp, lst in c.samples.items():
            if not lst:
                continue
            st.add(label, sp, fuse([e for _, e, _ in lst]), float(np.mean([qq for qq, _, _ in lst])), mean_pose,
                   samples=[e for _, e, _ in lst])
            for _, _, aug in lst:
                extra.setdefault(sp, []).extend((label, a) for a in aug)
            added += 1
    if added:
        for sp in st.spectra():
            st.fit_whitening(sp, extra.get(sp))
        st.save()
    return added


# ------------------------------------------------------------ emitter ----
def emitter_verdict(ir_mean: float, ir_face: bool, rgb_mean: float | None) -> tuple[str, str]:
    """(state, message). Inference: RGB sees the room light, IR sees the emitter."""
    if rgb_mean is None:
        return "unknown", "no RGB camera to compare against - judge by eye: is your face lit?"
    if rgb_mean > 55:
        return "inconclusive", "room is lit (RGB %.0f) - turn the lights OFF to test the emitter" % rgb_mean
    if ir_face or ir_mean >= 45:
        return "ok", "EMITTER WORKS: room dark (RGB %.0f) but IR sees you (%.0f)" % (rgb_mean, ir_mean)
    return "dead", "EMITTER NOT FIRING: room dark and IR is dark too (%.0f)" % ir_mean


def _render_emitter(snap: dict) -> np.ndarray:
    left = snap["ir"].copy()
    if snap["box"] is not None:
        x, y, w, h = snap["box"]
        cv2.rectangle(left, (x, y), (x + w, y + h), COL_OK, 2)
    _text(left, "IR  mean %.0f" % snap["ir_mean"], (10, 24))
    panels = [left]
    if snap["rgb"] is not None:
        right = cv2.resize(snap["rgb"], (left.shape[1], left.shape[0]))
        _text(right, "RGB  mean %.0f" % snap["rgb_mean"], (10, 24))
        panels.append(right)
    canvas = np.hstack(panels)
    col = {"ok": COL_OK, "dead": (230, 60, 60), "inconclusive": COL_REQ}.get(snap["state"], COL_TXT)
    _text(canvas, snap["msg"], (10, canvas.shape[0] - 14), 0.55, col)
    _text(canvas, "q = quit", (canvas.shape[1] - 80, 24), 0.45, (180, 180, 180))
    return canvas


def emitter_check(cap: Capture, models: Models, gui: bool = True, seconds: float = 0) -> tuple[str, str]:
    result = {"state": "unknown", "msg": ""}
    t_end = time.monotonic() + seconds if seconds else None
    screen = None
    if gui:
        try:
            screen = disp.open_display(WIN, 640, 360, kind="emitter", canvas=(1280, 360))
        except disp.DisplayUnavailable:
            screen = None

    def evaluate():
        fs = cap.read_set()
        if "ir" not in fs:
            return None
        ir = fs["ir"]; rgb = fs.get("rgb")
        box = models.detect(ir.gray)
        result["state"], result["msg"] = emitter_verdict(float(ir.gray.mean()), box is not None, float(rgb.gray.mean()) if rgb else None)
        return ir, rgb, box

    def step():
        if any(k in (disp.KEY_Q, disp.KEY_ESC) for k in screen.keys()):
            return None
        ev = evaluate()
        if ev is None:
            return np.zeros((360, 1280, 3), np.uint8)
        ir, rgb, box = ev
        snap = {"ir": _canvas_of(ir), "rgb": _canvas_of(rgb) if rgb is not None else None,
                "box": (box.x, box.y, box.w, box.h) if box is not None else None,
                "ir_mean": float(ir.gray.mean()), "rgb_mean": float(rgb.gray.mean()) if rgb is not None else None,
                "state": result["state"], "msg": result["msg"]}
        return snap if screen.snapshot else _render_emitter(snap)

    if screen is not None:
        screen.run(step)
    else:
        while True:
            evaluate()
            if t_end is None or time.monotonic() > t_end:
                break
    return result["state"], result["msg"]


# ------------------------------------------------------------ calibration ----
def _ir_device(cap) -> str:
    from .engine import config as cfgmod
    paths, _ = cfgmod.camera_paths(cfgmod.load())
    return paths.get("ir", "")


class _LiveSample:
    """What calibration.Collector needs from one frame set (the lab's Sample shape, computed here)."""
    __slots__ = ("raw", "wdist", "light", "label", "yaw", "pitch", "roll", "face_w")

    def __init__(self):
        self.raw, self.wdist, self.light, self.label = {}, {}, {}, {}
        self.yaw = self.pitch = self.roll = None
        self.face_w = None


def guided_setup(st: Store, cap: Capture, models: Models, cams: dict, rec: dict, mirror: bool = True,
                 need: int | None = None, timeout: float | None = None,
                 identity: str = "", condition: str = "", poses=None, style: str = "guided"):
    """ONE guided window that both ENROLLS and CALIBRATES (roadmap: fused setup). For each pose the
    window guides the head to a target (chase-the-dot with a derived arrow, so left/right is always
    right in the mirrored preview), captures embeddings in the `_Sampler` PROCESS, and keeps them
    smooth. Enrollment uses the best per-pose frames; the REMAINING (held-out) frames measure real
    distances so the gate is honest in the same sweep. Returns a dict summary, or None if cancelled.
    """
    from .engine import calibration as cal
    from .engine.detect import fuse
    from collections import defaultdict, deque
    sgn = -1.0 if mirror else 1.0
    poses = list(poses) if poses else cal.SETUP_POSES
    # segmented-coverage-ring visuals (style="ring"): the angular poses OTHER than the frontal hub
    # are ring segments, placed by their display target; the hub (centre) and the lean/light tail are
    # not. Computed once; the per-frame snapshot says which segments are done and which is active.
    ring_mode = style == "ring"
    seg_idx = [i for i, p in enumerate(poses)
               if p[1] in cal.DIRECTIONAL_CUES and tuple(_target_display(p[3], sgn)) != (0.0, 0.0)]
    ring_segs = [list(_target_display(poses[i][3], sgn)) for i in seg_idx]
    centre_i = next((i for i, p in enumerate(poses) if p[1] == "centre"), None)
    primary = "ir" if "ir" in cap.spectra else "rgb"
    others = [sp for sp in cap.spectra if sp != primary]
    models.sp68
    em = None
    if "ir" in cap.spectra:
        try:
            from .engine import emitter as emitmod, config as cfgmod
            ins = emitmod.instructions_for(cfgmod.load())
            if ins:
                em = emitmod.Emitter(_ir_device(cap), ins)
                if em.available:
                    em.on()
        except Exception:
            em = None
    sampler = _Sampler(models)
    probe = cap.read_one(primary, flush=0)
    ph, pw = (probe.gray.shape[:2] if probe is not None else (360, 640))
    title = "ward setup" + (f" · {identity}" if identity else "") + (f" · {condition}" if condition else "")
    try:
        screen = disp.open_display(title, pw, ph, kind="calibrate", sgn=1.0)
    except disp.DisplayUnavailable as e:
        sampler.close()
        if em is not None:
            em.close()
        raise GuiUnavailable(str(e))
    if not getattr(screen, "snapshot", False):       # canvas fallback (no libadwaita): this window needs snapshots
        sampler.close()
        if em is not None:
            em.close()
        raise GuiUnavailable("the setup window needs GTK 4 + libadwaita (python3-gi, gir1.2-adw-1); use --guided for the text flow")
    shape3d = np.array(st.calib["shape3d"]) if st.calib.get("shape3d") else None
    neutral = posemod.Neutral(st.calib["neutral_ratios"]) if st.calib.get("neutral_ratios") else None
    offset = tuple(st.calib.get("pose_offset") or (0.0, 0.0))

    # per pose: all captured (quality, emb, aug) per spectrum, and the baseline face width
    caps: dict = {i: defaultdict(list) for i in range(len(poses))}
    need_align = len(cap.spectra) == 2 and not condition and not st.calib.get("s")
    align_pairs: list = []
    widths: list = []
    st_data = {"phase": "intro", "cancel": False, "done": None, "i": 0, "hits": 0, "started": None,
               "last_sample_t": 0.0, "lmfilter": posemod.LandmarkFilter(), "filter": posemod.PoseFilter(),
               "prev_box": None, "fps_t": time.monotonic(), "fps_n": 0, "fps": 0.0, "emit_t": 0.0,
               "counted_t": 0.0, "ready_since": None, "intro_t0": None}
    NEEDN = need or cal.NEED_FRAMES
    NEED_S = min(cal.RING_NEED_SAMPLES if ring_mode else cal.NEED_SAMPLES, NEEDN)   # ring pools the gate over more poses, so each holds fewer; tests pass a small `need`
    TIMEOUT = timeout or cal.PASS_TIMEOUT_S

    def satisfied(pose_disp, cue, face_w):
        if cue == "light":
            return True
        if cue == "closer":
            return face_w is not None and widths and face_w >= float(np.median(widths)) * 1.12
        if cue == "away":
            return face_w is not None and widths and face_w <= float(np.median(widths)) * 0.9
        if pose_disp is None:
            return None
        return cal.pose_ok(pose_disp, poses[st_data["i"]][3])

    def advance():
        if st_data["i"] == 0 and widths:
            pass
        st_data["i"] += 1; st_data["hits"] = 0; st_data["started"] = time.monotonic(); st_data["active"] = 0.0

    def step():
        nonlocal sampler
        for k in screen.keys():
            if k == KEY_GLASSES and st_data["phase"] == "done" and st_data.get("offer_glasses"):
                start_condition("glasses"); continue
            if k == disp.KEY_ESC:
                if st_data["phase"] == "done":
                    return None                              # closing the summary keeps the finished enrolment
                st_data["cancel"] = True
            elif k == disp.KEY_ENTER:
                if st_data["phase"] == "intro":
                    st_data["phase"] = "run"; st_data["started"] = time.monotonic(); st_data["active"] = 0.0
                elif st_data["phase"] == "done":
                    return None
                elif st_data["phase"] == "run":
                    advance()
                    if st_data["i"] >= len(poses):
                        finish()
        if st_data["cancel"]:
            return None
        now = time.monotonic()
        dt = min(0.25, now - st_data.get("last_step_t", now)); st_data["last_step_t"] = now
        if st_data["phase"] == "done":
            left = (DONE_AUTOCLOSE_S * (2 if st_data.get("offer_glasses") else 1)) - (now - st_data["done_t"])
            if st_data["done"].get("ok") and left <= 0:
                return None                                  # auto-finish: the summary is printed in the terminal too
            fr = cap.read_one(primary, flush=0)
            snap = dict(st_data["done"]); snap["closing_in"] = left if st_data["done"].get("ok") else None
            if fr is not None:
                snap["frame"] = fr.rgb[:, ::-1].copy() if mirror else fr.rgb
            return snap
        if getattr(cap, "lost", ""):                         # the camera is gone: end with an honest card
            st_data["failed"] = cap.lost; st_data["failed_kind"] = "camera"
            finish()
            return dict(st_data["done"])
        fr = cap.read_one(primary, flush=0)
        if fr is None:
            stalled = now - st_data.get("last_frame_t", now) > 2.0
            return {"phase": "run" if st_data["phase"] == "run" else st_data["phase"], "index": st_data["i"],
                    "total": len(poses), "camera_stalled": stalled}
        st_data["last_frame_t"] = now
        st_data["fps_n"] += 1
        if now - st_data["fps_t"] >= 1.0:
            st_data["fps"] = st_data["fps_n"] / (now - st_data["fps_t"]); st_data["fps_n"], st_data["fps_t"] = 0, now
        if em is not None and em.available and primary == "ir" and (now - st_data["emit_t"]) > 0.8:
            em.on(); st_data["emit_t"] = now
        h, w = fr.gray.shape[:2]
        q = measure(fr.gray, models.max_height)
        gray_n = brightness.normalise(fr.gray, q.mean)
        box = None if (q.sharpness < 5.0 and q.mean < cams[f"{primary}_dark_threshold"]) else models.detect(gray_n, st_data["prev_box"])
        st_data["prev_box"] = box
        light = cap.meter(primary, fr.gray, box) if box is not None else None
        light_state = light.state(primary) if light else ""
        img = fr.rgb[:, ::-1].copy() if mirror else fr.rgb
        b = ((w - box.x - box.w) if mirror else box.x, box.y, box.w, box.h) if box is not None else None
        base = {"frame": img, "box": b, "light": light_state, "fps": st_data["fps"]}

        if st_data["phase"] == "intro":
            ready = box is not None and light_state != "dark"
            intro_wait = 6.0 if st_data.get("main") else 5.0     # glasses sweep: time to put them on; main: time to read the intro
            st_data["intro_t0"] = st_data["intro_t0"] or now
            # IR↔RGB alignment while the user settles in (was a separate terminal step before the window)
            if ready and need_align and not st_data.get("main") and primary == "ir" and len(align_pairs) < 10:
                fo = cap.read_one("rgb", flush=0)
                br = models.detect(fo.gray) if fo is not None else None
                if br is not None:
                    from .engine.enroll import pair_offset
                    align_pairs.append(pair_offset(box, br, fr.gray.shape, fo.gray.shape))
            if ready and not st_data.get("main"):
                st_data["ready_since"] = st_data["ready_since"] or now
                if now - st_data["ready_since"] >= intro_wait:
                    st_data["phase"] = "run"; st_data["started"] = now; st_data["active"] = 0.0
            else:
                # the glasses sweep never starts by itself: frames without glasses would be
                # labelled 'glasses' if the user is still putting them on (review A8)
                st_data["ready_since"] = None
            base.update({"phase": "intro", "index": 0, "total": len(poses), "ready": ready,
                         "intro_text": CONDITION_INTRO.get(condition, "") if st_data.get("main") else MAIN_INTRO,
                         "manual_start": bool(st_data.get("main")),
                         "waited": now - st_data["intro_t0"],
                         "countdown": max(0.0, intro_wait - (now - st_data["ready_since"])) if st_data["ready_since"] else None})
            return base

        # pose (smoothed) at camera rate
        pose_disp = None; face_w = None
        if box is not None and box.w >= MIN_ENROLL_FACE_PX and light_state != "dark":
            face_w = int(box.w)
            sbox = st_data["lmfilter"].smooth_box(box, now)
            lm68 = st_data["lmfilter"](models.landmarks68(gray_n, sbox), now)
            raw = posemod.estimate(lm68, w, h, shape3d=shape3d, neutral=neutral)
            if raw is not None:
                filt = st_data["filter"](raw, now)
                if shape3d is not None:
                    filt = posemod.Pose(filt.yaw - offset[0], filt.pitch - offset[1], filt.roll)
                pose_disp = (sgn * filt.yaw, filt.pitch)
            if st_data["i"] == 0 and not st_data.get("main"):
                widths.append(face_w)
            # capture embeddings for this pose in the worker process
            if (now - st_data["last_sample_t"]) * 1000 >= SAMPLE_EVERY_MS and not sampler.busy and sampler.alive:
                # no synthetic lighting variants: 3 extra embeddings per sample (~255 ms) quartered the
                # sample rate and changed no decision (review A11); whitening learns from real samples
                items = [(primary, brightness.embed_rgb(fr.rgb, fr.gray, primary, q.mean), (box.x, box.y, box.w, box.h), q.score(), True, False)]
                lights = {primary: light_state}
                for sp in others:
                    fo = cap.read_one(sp, flush=0)
                    if fo is not None:
                        qo = measure(fo.gray, models.max_height)   # rank each camera by ITS OWN quality
                        lights[sp] = "dark" if qo.mean < cams[f"{sp}_dark_threshold"] else ""
                        # lighting variants only for the primary: 4 embeddings per camera per sample
                        # halved the pass's sample rate, and whitening also learns from real samples
                        items.append((sp, brightness.embed_rgb(fo.rgb, fo.gray, sp, qo.mean), (box.x, box.y, box.w, box.h), qo.score(), False, False))
                # the pose and light travel WITH the sample: tagged when it was taken, not when it drains
                if sampler.submit(st_data["i"], (pose_disp, lights), items):
                    st_data["last_sample_t"] = now
        if not sampler.alive:                                # the worker died (OOM, dlib crash): respawn once
            if st_data.get("respawns", 0) < 1:
                st_data["respawns"] = st_data.get("respawns", 0) + 1
                try:
                    sampler.close(timeout=0.2)
                except Exception:
                    pass
                sampler = _Sampler(models)
            else:
                st_data["failed"] = "the face-processing worker stopped"; st_data["failed_kind"] = "worker"
                finish()
                return dict(st_data["done"])
        for pose_i, tag, results in sampler.drain():
            pose_at, lights = tag
            for item in results:
                sp, qq, emb = item[0], item[1], item[2]
                aug = item[3] if len(item) > 3 else []
                caps[pose_i][sp].append((qq, emb, aug, lights.get(sp, ""), pose_at))
        label, cue, text, target = poses[st_data["i"]]
        ok = satisfied(pose_disp, cue, face_w)
        if st_data["started"] is None:
            st_data["started"] = now
        if ok:
            st_data["hits"] += 1; st_data["counted_t"] = now
        # the pass clock runs only while a usable face is in view (review A5): stepping away, a
        # too-small or dark face never costs a pose; a wall-clock cap still ends a stuck pass
        if face_w is not None:
            st_data["active"] = st_data.get("active", 0.0) + dt
        elapsed = st_data.get("active", 0.0)
        wall = now - st_data["started"]
        small = box is not None and box.w < MIN_ENROLL_FACE_PX
        # a pass is done when it has enough USABLE embeddings (pose-true, lit), not camera frames:
        # the sampler yields one every ~0.3–0.7 s, so 20 on-target frames (~0.7 s) left most
        # directional poses with < 3 samples once transition frames stopped counting
        got = max([len(cal.usable_rows(rows, target, cue)) for rows in caps[st_data["i"]].values()] or [0])
        if cue == "light":                                # time to actually change the light, then samples
            progress = min(1.0, elapsed / cal.LIGHT_PASS_MIN_S, got / NEED_S)
            done = elapsed >= cal.LIGHT_PASS_MIN_S and got >= NEED_S
        else:
            progress = min(1.0, st_data["hits"] / NEEDN, got / NEED_S)
            done = st_data["hits"] >= NEEDN and got >= NEED_S
        base.update({"phase": "run", "index": st_data["i"], "total": len(poses), "key": cue, "text": text,
                     "cue": cue, "pose": pose_disp, "target": _target_display(target, sgn), "progress": progress,
                     "ok": ok, "hits": st_data["hits"], "need": NEEDN, "elapsed": elapsed,
                     "counted": (now - st_data["counted_t"]) < 0.25, "small": small})
        if ring_mode:
            cur = st_data["i"]
            base.update({"style": "ring", "ring_segs": ring_segs,
                         "ring_done": [seg_idx[k] < cur for k in range(len(seg_idx))],
                         "ring_active": seg_idx.index(cur) if cur in seg_idx else None,
                         "hub_done": centre_i is not None and cur > centre_i})
        if done or elapsed >= TIMEOUT or wall >= 3 * TIMEOUT:
            base["pass_done"] = True
            advance()
            if st_data["i"] >= len(poses):
                finish()
                base.update(st_data["done"])
        return base

    def start_condition(name: str) -> None:
        """Second sweep in the SAME window (review C1): the user puts their glasses on and does the
        short condition poses; models carry `condition`, the owner's measured gate is untouched."""
        nonlocal poses, caps, condition, sampler
        st_data["main"] = {"done": st_data["done"], "result": st_data.get("result"), "added": st_data.get("added", 0)}
        condition, poses = name, list(cal.CONDITION_POSES)
        caps = {i: defaultdict(list) for i in range(len(poses))}
        try:
            sampler.close(timeout=0.2)                       # the main sweep's worker (finish() closed it; be sure)
        except Exception:
            pass
        sampler = _Sampler(models)
        st_data.update({"respawns": 0, "phase": "intro", "i": 0, "hits": 0, "started": None, "ready_since": None,
                        "intro_t0": None, "done": None, "offer_glasses": False, "prev_box": None,
                        "lmfilter": posemod.LandmarkFilter(), "filter": posemod.PoseFilter()})

    def finish():
        from .engine import calibration as cal
        for pose_i, tag, results in sampler.close():    # samples still in flight belong to the last pass
            pose_at, lights = tag
            for item in results:
                caps[pose_i][item[0]].append((item[1], item[2], item[3] if len(item) > 3 else [], lights.get(item[0], ""), pose_at))
        if align_pairs and not st_data.get("main"):
            from .engine.enroll import summarise_pairs
            st.calib.update(summarise_pairs(align_pairs))
        added, dist_by, _extra = cal.build_from_captures(st, caps, poses, SAMPLES_PER_POSE, fuse,
                                                         identity=identity, condition=condition)
        # a pose counts when ANY camera produced its model: in bright daylight the IR face can meter
        # dark while RGB carries — that is a good enrolment, not a failed one
        missed = [poses[i][2] for i in range(len(poses))
                  if all(cal.select_rows(rows, poses[i][3], poses[i][1], SAMPLES_PER_POSE) is None
                         for rows in caps[i].values()) or not caps[i]]
        main = st_data.get("main")
        if main is not None:                                 # the condition sweep: the main result stands
            stopped = st_data.pop("failed", "")              # a failure here never costs the finished main enrolment
            st_data["failed_kind"] = ""
            if stopped:
                st.models = [m for m in st.models if m.condition != condition or m.label not in {p[0] for p in poses}] \
                    if added else st.models
                added = 0
            st_data.update({"result": main["result"], "added": main["added"] + added, "done_t": time.monotonic(), "phase": "done"})
            extra = (f"{condition.capitalize()} sweep stopped ({stopped}): not added." if stopped else
                     f"With {condition}: {added} models" + (f" (no model for: {', '.join(missed)})" if missed else "") + ".")
            st_data["done"] = dict(main["done"], summary=main["done"]["summary"] + "\n" + extra, offer_glasses=False,
                                   headline=f"Enrolled {main['added'] + added} models · {condition} included")
            return
        res = cal.derive_from_holdout(dist_by)
        st_data["result"] = res
        st_data["added"] = added
        st_data["done_t"] = time.monotonic()
        failed = st_data.get("failed", "")
        got_poses = len(poses) - len(missed)
        # a sweep that produced only a pose or two is not an enrolment: with --fresh, saving it
        # would replace a good enrolment with a weak one — require at least half the poses
        ok = added > 0 and not failed and got_poses * 2 >= len(poses)
        if added > 0 and not ok and not failed:
            failed = f"only {got_poses} of {len(poses)} poses produced a model"
            st_data["failed"] = failed; st_data["failed_kind"] = "poses"
        st_data["done"] = {"phase": "done", "index": len(poses), "total": len(poses), "ok": ok,
                           "headline": ({"camera": "Camera lost — nothing saved", "worker": "Processing stopped — nothing saved",
                                         "poses": "Not enough poses — nothing saved"}.get(st_data.get("failed_kind", ""), "Nothing enrolled") if not ok
                                        else f"Enrolled {added} models · gate set from your real distances" if res.ok() else f"Enrolled {added} models"),
                           "summary": (f"{failed[:1].upper() + failed[1:]}. Your previous enrolment is unchanged — try again in better light, facing the camera." if failed and not ok else
                                       "No pose produced a model — your previous enrolment is unchanged." if not ok else
                                       _setup_summary(res, added, missed, ir="ir" in cap.spectra, condition=condition))}
        st_data["phase"] = "done"
        st_data["offer_glasses"] = not condition and ok
        st_data["done"]["offer_glasses"] = st_data["offer_glasses"]

    try:
        screen.run(step)
    finally:
        sampler.close()
        if em is not None:
            try:
                em.on(); em.close()
            except Exception:
                pass
    if st_data["cancel"] and st_data["phase"] != "done":
        main = st_data.get("main")
        if main is None:
            return None
        # cancelled during the optional glasses sweep: the finished main enrolment still stands
        return {"result": main["result"], "added": main["added"], "failed": ""}
    return {"result": st_data.get("result"), "added": st_data.get("added", 0), "failed": st_data.get("failed", "")}


def ring_setup(st: Store, cap: Capture, models: Models, cams: dict, rec: dict, mirror: bool = True,
               need: int | None = None, timeout: float | None = None, identity: str = "", condition: str = ""):
    """`ward enroll --style ring`: the same enrol+calibrate window as `guided_setup`, driven through
    the segmented coverage ring (`calibration.RING_POSES`) with the Face-ID-style overlay. A thin
    wrapper so the default path and all its tests are untouched — the capture → sampler →
    build_from_captures backend is shared."""
    from .engine import calibration as cal
    return guided_setup(st, cap, models, cams, rec, mirror, need, timeout, identity, condition,
                        poses=cal.RING_POSES, style="ring")


def _target_display(target, sgn):
    return (target[0], target[1])   # SETUP_POSES targets are already in display space


DONE_AUTOCLOSE_S = 12.0
KEY_GLASSES = "glasses"                # the done card's "I wear glasses" button
CONDITION_INTRO = {"glasses": "Put your glasses on, then press Start", "headset": "Put your headset on, then press Start"}
MAIN_INTRO = ("Ward will learn your face.\n"
              "Follow the arrow to turn your head to a few positions and hold each for a moment "
              "— about 15 seconds.\nYour password always works; this just adds your face.")


def _setup_summary(res, added, missed=(), ir: bool = False, condition: str = ""):
    lines = ["Your face is set up — you can now unlock with your face."]
    lines.append(f"Enrolled {added} pose models" + (f" for the condition '{condition}'." if condition else "."))
    if missed:
        lines.append("No model for: " + ", ".join(missed) + " — re-run later to add them (the timeout moved on).")
    for sp in sorted(res.gate):
        lines.append(f"{sp.upper()}  {res.n[sp]} checks · median {res.p50[sp]:.2f} · p95 {res.p95[sp]:.2f} → gate {res.gate[sp]}"
                     + (f" · whitening {res.w_threshold[sp]}" if sp in res.w_threshold else ""))
    lines += [f"⚠ {w}" for w in res.warnings]
    if ir and not condition:
        lines.append("Works in the dark too: at night the camera's own infrared light shows the same face.")
    return "\n".join(lines)


def calibration_session(st, cap: Capture, models: Models, cams: dict, rec: dict, mirror: bool = True,
                        need: int | None = None, timeout: float | None = None):
    """The calibration session in a window (appui.CalibrateWindow).

    Built like enrollment so the window stays at camera rate: this loop only reads frames,
    runs YuNet and the 68 landmarks (a few ms) and draws; the embeddings — dlib holds the GIL for
    ~30 ms each — run in the `_Sampler` PROCESS at most every SAMPLE_EVERY_MS. Distances are
    computed here from the returned embeddings (fusion of the last `fusion_frames` per camera,
    raw and whitened against the store). Returns the Measures, or None when cancelled."""
    from .engine import calibration as cal
    from .engine.detect import fuse
    from collections import deque
    sgn = -1.0 if mirror else 1.0
    col = cal.Collector(need=need or cal.NEED_FRAMES, timeout=timeout or cal.PASS_TIMEOUT_S, sgn=sgn)
    primary = "ir" if "ir" in cap.spectra else "rgb"
    others = [sp for sp in cap.spectra if sp != primary]
    models.sp68
    em = None
    if "ir" in cap.spectra:
        try:
            from .engine import emitter as emitmod, config as cfgmod
            ins = emitmod.instructions_for(cfgmod.load())
            if ins:
                em = emitmod.Emitter(_ir_device(cap), ins)
                if em.available:
                    em.on()                                  # recognition needs the LEDs; a probe may have left them off
                if os.environ.get("WARD_DEBUG"):
                    print(f"[calibrate] emitter available={em.available} err={em.error!r}", flush=True)
        except Exception as e:
            if os.environ.get("WARD_DEBUG"):
                print(f"[calibrate] emitter setup failed: {e!r}", flush=True)
            em = None
    sampler = _Sampler(models)
    probe = cap.read_one(primary, flush=0)
    ph, pw = (probe.gray.shape[:2] if probe is not None else (360, 640))
    try:
        screen = disp.open_display("ward calibration", pw, ph, kind="calibrate", sgn=1.0)
    except disp.DisplayUnavailable as e:
        sampler.close()
        raise GuiUnavailable(str(e))
    if not getattr(screen, "snapshot", False):
        sampler.close()
        raise GuiUnavailable("the calibration window needs GTK 4 + libadwaita; use --text")
    shape3d = np.array(st.calib["shape3d"]) if st.calib.get("shape3d") else None
    neutral = posemod.Neutral(st.calib["neutral_ratios"]) if st.calib.get("neutral_ratios") else None
    offset = tuple(st.calib.get("pose_offset") or (0.0, 0.0))
    n_fuse = int(rec.get("fusion_frames", 4))
    recent: dict[str, deque] = {sp: deque(maxlen=n_fuse) for sp in cap.spectra}
    state = {"phase": "intro", "cancel": False, "pass_done_at": None, "done": None, "last_sample_t": 0.0,
             "lmfilter": posemod.LandmarkFilter(), "filter": posemod.PoseFilter(), "prev_box": None,
             "fps_t": time.monotonic(), "fps_n": 0, "fps": 0.0, "counted_t": 0.0, "pending_light": {}}

    def snapshot_base(fr, box, light_state):
        img = fr.rgb[:, ::-1].copy() if mirror else fr.rgb
        w = fr.gray.shape[1]
        b = ((w - box.x - box.w) if mirror else box.x, box.y, box.w, box.h) if box is not None else None
        return {"frame": img, "box": b, "light": light_state, "fps": state["fps"]}

    def distances(sample: _LiveSample) -> None:
        for sp, dq in recent.items():
            if not dq or not st.by_spectrum(sp):
                continue
            fused = fuse(list(dq))
            d, label = st.min_distance(fused, sp)
            wd, _thr = st.wdist(fused, sp)
            sample.raw[sp], sample.label[sp], sample.wdist[sp] = d, label or "", wd
            sample.light[sp] = state["pending_light"].get(sp, "")
        # fused embeddings for the whitening scatter
        state["fused"] = {sp: fuse(list(dq)) for sp, dq in recent.items() if dq}

    def step():
        for k in screen.keys():
            if os.environ.get("WARD_DEBUG"):
                print(f"[calibrate] key {k} in phase {state['phase']}", flush=True)
            if k == disp.KEY_ESC:
                state["cancel"] = True
            elif k == disp.KEY_ENTER:
                if state["phase"] == "intro":
                    state["phase"] = "measuring"; col.start_pass()
                elif state["phase"] == "done":
                    return None
                elif state["phase"] == "measuring":
                    col.skip()
                    if col.finished:
                        finish()
        if state["cancel"]:
            return None
        fr = cap.read_one(primary, flush=0)
        if fr is None:
            return {"phase": state["phase"], "index": col.i, "total": len(col.prompts)}
        state["fps_n"] += 1; state["frames_total"] = state.get("frames_total", 0) + 1
        if time.monotonic() - state["fps_t"] >= 1.0:
            state["fps"] = state["fps_n"] / (time.monotonic() - state["fps_t"]); state["fps_n"], state["fps_t"] = 0, time.monotonic()
        h, w = fr.gray.shape[:2]
        q = measure(fr.gray, models.max_height)
        gray_n = brightness.normalise(fr.gray, q.mean)
        box = None if (q.sharpness < 5.0 and q.mean < cams[f"{primary}_dark_threshold"]) else models.detect(gray_n, state["prev_box"])
        state["prev_box"] = box
        light = cap.meter(primary, fr.gray, box) if box is not None else None
        light_state = light.state(primary) if light else ""
        snap = snapshot_base(fr, box, light_state)
        now = time.monotonic()
        if os.environ.get("WARD_DEBUG") and now - state.get("dbg_t", 0) >= 1.0:
            state["dbg_t"] = now
            print(f"[calibrate] phase={state['phase']} fps={state['fps']:.0f} box={'yes' if box is not None else 'no'} light={light_state!r} "
                  f"q.mean={q.mean:.0f} ready_for={now - state['ready_since']:.1f}s" if state.get("ready_since") else
                  f"[calibrate] phase={state['phase']} fps={state['fps']:.0f} box={'yes' if box is not None else 'no'} light={light_state!r} q.mean={q.mean:.0f}", flush=True)
        # keep the IR LEDs on the whole session: in a dark room the emitter is the only light and
        # some modules silently drop the control — re-assert it periodically so it never goes black
        if em is not None and em.available and primary == "ir" and (now - state.get("emit_t", 0)) > 0.8:
            em.on(); state["emit_t"] = now
        if state["phase"] == "intro":
            ready = box is not None and light_state != "dark"
            if ready:
                state.setdefault("ready_since", now)
                if now - state["ready_since"] >= INTRO_AUTOSTART_S:      # the face has been there long enough: go
                    state["phase"] = "measuring"; col.start_pass()
            else:
                state.pop("ready_since", None)
            state.setdefault("intro_t0", now)
            snap.update({"phase": "intro", "index": 0, "total": len(col.prompts), "ready": ready,
                         "waited": now - state["intro_t0"],
                         "countdown": max(0.0, INTRO_AUTOSTART_S - (now - state["ready_since"])) if ready else None})
            return snap
        if state["phase"] == "done":
            snap.update(state["done"]); return snap
        # pose at camera rate (smoothed like enrollment), embeddings only every SAMPLE_EVERY_MS
        sample = _LiveSample()
        if box is not None and box.w >= MIN_ENROLL_FACE_PX and light_state != "dark":
            sbox = state["lmfilter"].smooth_box(box, now)
            lm68 = state["lmfilter"](models.landmarks68(gray_n, sbox), now)
            raw = posemod.estimate(lm68, w, h, shape3d=shape3d, neutral=neutral)
            if raw is not None:
                filt = state["filter"](raw, now)
                if shape3d is not None:
                    filt = posemod.Pose(filt.yaw - offset[0], filt.pitch - offset[1], filt.roll)
                sample.yaw, sample.pitch, sample.roll = filt.yaw, filt.pitch, filt.roll
            sample.face_w = int(box.w)
            if (now - state["last_sample_t"]) * 1000 >= SAMPLE_EVERY_MS and not sampler.busy and sampler.alive:
                items = [(primary, brightness.embed_rgb(fr.rgb, fr.gray, primary, q.mean), (box.x, box.y, box.w, box.h), q.score(), True, False)]
                state["pending_light"][primary] = light_state
                for sp in others:
                    fo = cap.read_one(sp, flush=0)
                    if fo is not None:
                        items.append((sp, brightness.embed_rgb(fo.rgb, fo.gray, sp), (box.x, box.y, box.w, box.h), q.score(), False, False))
                if sampler.submit(None, None, items):
                    state["last_sample_t"] = now
        # results from the process → recent embeddings → distances
        for _cell, _pose, results in sampler.drain():
            for item in results:
                sp, _q, emb = item[0], item[1], item[2]
                recent[sp].append(emb)
                if sp != primary:
                    state["pending_light"][sp] = "ok"
        if recent[primary]:
            distances(sample)
        stt = col.feed(sample if (sample.raw or sample.yaw is not None) else None, state.get("fused"))
        if stt.get("ok") and stt.get("hits") and (sample.raw or sample.yaw is not None):
            state["counted_t"] = now
        dists = " · ".join(f"{sp} {sample.raw[sp]:.2f}" + (f" (w {sample.wdist[sp]:.1f})" if sample.wdist.get(sp) is not None else "")
                           for sp in ("ir", "rgb") if sample.raw.get(sp) is not None)
        snap.update({"phase": "measuring", "pose": col.display_pose(sample), "index": stt.get("index", col.i), "total": len(col.prompts),
                     "key": stt.get("key", ""), "text": stt.get("text", ""), "cue": stt.get("cue", "centre"),
                     "progress": stt.get("progress", 0.0), "ok": stt.get("ok"), "dists": dists,
                     "hits": stt.get("hits", 0), "need": stt.get("need", col.need), "elapsed": stt.get("elapsed", 0.0),
                     "counted": (now - state["counted_t"]) < 0.25, "sampling": sampler.busy})
        if stt.get("pass_done"):
            snap["pass_done"] = True
        if stt.get("finished"):
            finish()
            snap.update(state["done"])
        return snap

    def finish():
        res = cal.summarise(col.measures)
        lines = []
        for sp in sorted(res.gate):
            lines.append(f"{sp.upper()}  {res.n[sp]} measurements · median {res.p50[sp]:.2f} · p95 {res.p95[sp]:.2f} → gate {res.gate[sp]}"
                         + (f"\n      whitening threshold {res.w_threshold[sp]}" if sp in res.w_threshold else ""))
        lines += [f"⚠ {wn}" for wn in res.warnings]
        state["done"] = {"phase": "done", "index": len(col.prompts), "total": len(col.prompts),
                         "headline": "Thresholds set from your measurements." if res.ok() else "Not enough measurements — nothing changed.",
                         "summary": "\n".join(lines)}
        state["phase"] = "done"

    t_run = time.monotonic(); state["frames_total"] = 0
    try:
        screen.run(step)
    finally:
        sampler.close()
        if em is not None:
            try:
                em.on(); em.close()
            except Exception:
                pass
    calibration_session.last_fps = state["frames_total"] / max(1e-6, time.monotonic() - t_run)
    if state["cancel"]:
        return None
    return col.measures


calibration_session.last_fps = 0.0
INTRO_AUTOSTART_S = 2.5
