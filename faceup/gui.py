"""Camera windows (OpenCV HighGUI, no extra toolkit): free-move enrollment and the emitter check.

Both run as root (like `howdy test`) and open a window on the user's display; if HighGUI
cannot open a window (no DISPLAY, headless), callers fall back to the text flows.
"""
from __future__ import annotations

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

WIN = "faceup"
SAMPLES_PER_CELL = 3
CELL_DONE = 2                 # samples of the primary spectrum that mark a cell as filled
STABLE_DEG = 4.0              # pose must move less than this between frames to be sampled
SAMPLE_EVERY_MS = 120         # embeddings are computed at most this often
MAX_TRACKS = 900              # landmark frames kept for the post-calibration (≈30 s)
COL_OK = (90, 190, 90); COL_REQ = (255, 140, 60); COL_OPT = (110, 110, 110); COL_CUR = (60, 220, 255)
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

    def push(self, spectrum: str, q: float, emb: np.ndarray) -> None:
        lst = self.samples.setdefault(spectrum, [])
        lst.append((q, emb))
        lst.sort(key=lambda t: t[0], reverse=True)
        del lst[SAMPLES_PER_CELL:]


def _sampler_main(req, res, detector_kind: str, max_height: int) -> None:
    """Worker PROCESS: dlib's Python bindings hold the GIL during detection and embedding, so a
    thread would freeze the preview (measured: 30 → 14 fps). A separate process keeps the UI
    at camera rate; frames travel over a pipe (~0.7 MB per sample, a few times per second)."""
    models = Models("yunet", max_height)        # fast detector here: this must keep up with the preview
    models.sp5; models.rec
    while True:
        job = req.get()
        if job is None:
            break
        cell, pose, items = job
        results = []
        for sp, rgb, hint, q, exact in items:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            box = Box(*hint)
            if not exact:
                box = models.detect(gray, box)
            if box is None or box.w < MIN_ENROLL_FACE_PX:
                continue
            results.append((sp, q, models.embed(rgb, box)))
        res.put((cell, pose, results))


class _Sampler:
    def __init__(self, models: Models):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.req = ctx.Queue(maxsize=2)
        self.res = ctx.Queue()
        # never let interpreter exit block on a queue feeder thread if the worker died
        self.req.cancel_join_thread(); self.res.cancel_join_thread()
        self.proc = ctx.Process(target=_sampler_main, args=(self.req, self.res, models.detector_kind, models.max_height), daemon=True)
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

    # ellipse outline of the required region (centre mirrored with everything else)
    cx, cy = to_px(sgn * posemod.REQ_CENTRE[0], posemod.REQ_CENTRE[1])
    ax = int(posemod.REQ_AXES[0] / (posemod.YAW_EDGES[-1] - posemod.YAW_EDGES[0]) * width)
    ay = int(posemod.REQ_AXES[1] / (posemod.PITCH_EDGES[-1] - posemod.PITCH_EDGES[0]) * PANEL_H)
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
    return ("Slowly " + " and ".join(parts) + " toward the ring") if parts else "Hold still..."


def free_move_enroll(st: Store, cap: Capture, models: Models, cams: dict, mirror: bool = True, profile: bool = False) -> int:
    """Track the head, fill the pose grid, fuse per cell and per spectrum. Returns models added.

    Per-frame cost is kept to: one camera read + ROI face detection (~5 ms) + 68 landmarks
    (~5 ms) + drawing. Embeddings (the expensive part) are computed only when the current
    cell still needs samples, at most every SAMPLE_EVERY_MS, and the second camera is read
    only at those moments.
    """
    models.sp68; models.sp5; models.rec                     # preload: no stall on the first face
    primary = "ir" if "ir" in cap.spectra else "rgb"
    cells: dict[tuple[int, int], Cell] = {}
    others = [sp for sp in cap.spectra if sp != primary]
    sampler = _Sampler(models)
    restarts = 0
    prev_pose = None

    def absorb(results) -> None:
        for cell, pose, items in results:
            c = cells.setdefault(cell, Cell())
            for sp, qq, emb in items:
                c.push(sp, qq, emb)
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
             "lmfilter": posemod.LandmarkFilter(), "stability": posemod.Stability()}
    # Mirror convention for everything the user sees: the preview is mirrored, so "left" on the
    # panel and in the hints must be the USER's left. Camera yaw is negative when the nose moves
    # camera-left, i.e. when the user turns to THEIR right → display yaw = −camera yaw.
    sgn = -1.0 if mirror else 1.0
    NEUTRAL_SAMPLES = 30          # ~1 s of face at 30 fps
    probe = cap.read_one(primary, flush=0)
    ph, pw = (probe.gray.shape[:2] if probe is not None else (360, 640))
    try:
        screen = disp.open_display(WIN, pw, ph + PANEL_H)
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
            raw = posemod.estimate(lm68, w, h, shape3d=state["shape3d"])
            if raw is not None:
                if len(state["tracks"]) < MAX_TRACKS:
                    state["tracks"].append(lm68.astype(np.float32))
                    state["fast_poses"].append((raw.yaw, raw.pitch))
                filt = state["filter"](raw, now)
                if calibrating:
                    # neutral pose: the user looking at the screen naturally. The camera sits above
                    # the eyes, so "level" reads as pitch +10..+20 — the grid is made relative to this.
                    state["neutral_samples"].append((filt.yaw, filt.pitch))
                    if len(state["neutral_samples"]) >= NEUTRAL_SAMPLES:
                        ys = sorted(a for a, _ in state["neutral_samples"]); ps = sorted(b for _, b in state["neutral_samples"])
                        state["neutral"] = (ys[len(ys) // 2], ps[len(ps) // 2])
                        st.calib["pose_offset"] = [round(state["neutral"][0], 1), round(state["neutral"][1], 1)]
                    current = posemod.Pose(0.0, 0.0, filt.roll)      # dot sits in the centre while calibrating
                else:
                    ny, npch = state["neutral"]
                    current = posemod.Pose(filt.yaw - ny, filt.pitch - npch, filt.roll)
                stable = state["stability"].push(current)
                state["prev_pose"] = current
                cell = current.cell() or posemod.nearest_cell(current.yaw, current.pitch)
                cov = state["coverage"]
                covered = cov.at(current.yaw, current.pitch) >= cov.FULL
                worker_ok = sampler.alive
                busy = sampler.busy if worker_ok else False
                gate = ("calibrating" if calibrating else "moving" if not stable else "covered here" if covered
                        else "worker busy" if busy else "")
                state["gate"] = gate
                if not gate and (now - state["last_sample_t"]) * 1000 >= SAMPLE_EVERY_MS:
                    c = cells.setdefault(cell, Cell())
                    if True:
                        items = [(primary, fr.rgb, (box.x, box.y, box.w, box.h), q.score(), True)]
                        for sp in others:
                            f2 = cap.read_one(sp, flush=0)
                            if f2 is None:
                                continue
                            q2 = measure(f2.gray, models.max_height)
                            if q2.reason(cams[f"{sp}_dark_threshold"]):
                                continue
                            sx, sy = f2.gray.shape[1] / w, f2.gray.shape[0] / h
                            items.append((sp, f2.rgb, (int(box.x * sx), int(box.y * sy), int(box.w * sx), int(box.h * sy)), q2.score(), False))
                        if worker_ok:
                            ok = sampler.submit(cell, (current.yaw, current.pitch), items)
                        else:
                            # worker process unavailable: sample inline (slower preview, but it works)
                            if not state.get("inline_warned"):
                                print("[faceup] sampling worker unavailable — computing embeddings inline (preview will be slower)", flush=True)
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
                            cov.add(current.yaw, current.pitch)       # THE line that was missing: coverage never grew
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
            print(f"[faceup] sampling worker died (exit {sampler.proc.exitcode}); restarting", flush=True)
            state["sampler"] = _Sampler(models)

        cov = state["coverage"]
        target = None if calibrating else cov.target(current.yaw if current else None, current.pitch if current else None)
        frac = cov.fraction()
        disp_current = posemod.Pose(sgn * current.yaw, current.pitch, sgn * current.roll) if current else None
        disp_target = (sgn * target[0], target[1]) if target else None
        if frac >= 0.97 and state["t_complete"] is None:
            state["t_complete"] = time.monotonic()

        state["fps_n"] += 1
        if time.monotonic() - state["fps_t"] >= 1.0:
            state["fps"] = state["fps_n"] / (time.monotonic() - state["fps_t"]); state["fps_n"] = 0; state["fps_t"] = time.monotonic()
        img = _canvas_of(fr)
        if mirror:
            img = cv2.flip(img, 1)
        if box is not None:
            bx = (w - box.x - box.w) if mirror else box.x
            cv2.rectangle(img, (bx, box.y), (bx + box.w, box.y + box.h), COL_OK if sampled else COL_CUR, 2)
        if light_state == "dark":
            _text(img, "Face too dark - is the IR emitter on? (turn on a light to continue)", (10, 24), 0.62, (255, 120, 90))
        elif calibrating:
            n = len(state["neutral_samples"])
            _text(img, f"Look at the centre of the screen naturally... calibrating {min(n, NEUTRAL_SAMPLES)}/{NEUTRAL_SAMPLES}", (10, 24), 0.62, COL_CUR)
        else:
            _text(img, _hint(disp_current, disp_target, box), (10, 24), 0.62)
        n_models = sum(1 for c in cells.values() if c.count("primary") >= CELL_DONE)
        gate = state.get("gate", "")
        status = (f"coverage {frac * 100:3.0f}% | {n_models} poses | {primary.upper()} {q.mean:.0f}{(' face ' + light_state) if light_state else ''}"
                  f" | {state['fps']:.0f} fps" + (f" | {gate}" if gate else " | sampling"))
        if current is not None:
            status += f" | yaw {sgn * current.yaw:+.0f} pitch {current.pitch:+.0f} roll {sgn * current.roll:+.0f}"
        _text(img, status, (10, h - 12), 0.5, (200, 200, 200))
        _text(img, "Enter = finish   Esc = cancel", (w - 235, 24), 0.45, (180, 180, 180))
        canvas = np.vstack([img, _coverage_panel(w, cov, disp_current, disp_target, sgn)])
        prof.lap("draw")
        prof.frame()
        if state["t_complete"] and time.monotonic() - state["t_complete"] > 3:
            state["finished"] = True
        return canvas

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
    for cell, c in cells.items():
        if c.count("primary") < CELL_DONE:
            continue
        label = posemod.cell_name(cell)
        st.remove_label(label)
        mean_pose = (float(np.mean([p[0] for p in c.poses])), float(np.mean([p[1] for p in c.poses])))
        for sp, lst in c.samples.items():
            if not lst:
                continue
            st.add(label, sp, fuse([e for _, e in lst]), float(np.mean([qq for qq, _ in lst])), mean_pose)
            added += 1
    if added:
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


def emitter_check(cap: Capture, models: Models, gui: bool = True, seconds: float = 0) -> tuple[str, str]:
    result = {"state": "unknown", "msg": ""}
    t_end = time.monotonic() + seconds if seconds else None
    screen = None
    if gui:
        try:
            screen = disp.open_display(WIN, 1280, 360)
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
        left = _canvas_of(ir)
        if box is not None:
            cv2.rectangle(left, (box.x, box.y), (box.x + box.w, box.y + box.h), COL_OK, 2)
        _text(left, "IR  mean %.0f" % ir.gray.mean(), (10, 24))
        panels = [left]
        if rgb is not None:
            right = cv2.resize(_canvas_of(rgb), (left.shape[1], left.shape[0]))
            _text(right, "RGB  mean %.0f" % rgb.gray.mean(), (10, 24))
            panels.append(right)
        canvas = np.hstack(panels)
        col = {"ok": COL_OK, "dead": (230, 60, 60), "inconclusive": COL_REQ}.get(result["state"], COL_TXT)
        _text(canvas, result["msg"], (10, canvas.shape[0] - 14), 0.55, col)
        _text(canvas, "q = quit", (canvas.shape[1] - 80, 24), 0.45, (180, 180, 180))
        return canvas

    if screen is not None:
        screen.run(step)
    else:
        while True:
            evaluate()
            if t_end is None or time.monotonic() > t_end:
                break
    return result["state"], result["msg"]
