"""Camera windows (OpenCV HighGUI, no extra toolkit): free-move enrollment and the emitter check.

Both run as root (like `howdy test`) and open a window on the user's display; if HighGUI
cannot open a window (no DISPLAY, headless), callers fall back to the text flows.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

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
COL_OK = (90, 190, 90); COL_REQ = (60, 140, 255); COL_OPT = (110, 110, 110); COL_CUR = (255, 220, 60)
COL_TXT = (235, 235, 235)


class GuiUnavailable(Exception):
    pass


def _open_window() -> None:
    try:
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(WIN, np.zeros((10, 10, 3), np.uint8))
        cv2.waitKey(1)
    except cv2.error as e:   # pragma: no cover
        raise GuiUnavailable(str(e))


def _bgr(frame) -> np.ndarray:
    return cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)


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


def _grid_panel(width: int, cells: dict, current, target) -> np.ndarray:
    """5×3 pose grid drawn as a panel: green = filled, orange = required & empty, grey = optional."""
    cw, ch = width // 5, 34
    panel = np.full((ch * 3 + 8, width, 3), 28, np.uint8)
    for (yi, pi) in posemod.ALL_CELLS:
        x0, y0 = yi * cw + 3, (2 - pi) * ch + 4          # pitch "up" drawn at the top
        filled = cells.get((yi, pi)) is not None and cells[(yi, pi)].count("primary") >= CELL_DONE
        col = COL_OK if filled else (COL_REQ if (yi, pi) in posemod.REQUIRED else COL_OPT)
        cv2.rectangle(panel, (x0, y0), (x0 + cw - 6, y0 + ch - 6), col, -1 if filled else 1)
        if (yi, pi) == target:
            cv2.rectangle(panel, (x0 - 1, y0 - 1), (x0 + cw - 5, y0 + ch - 5), COL_CUR, 2)
        _text(panel, posemod.cell_name((yi, pi)), (x0 + 5, y0 + 22), 0.45, COL_TXT if not filled else (20, 40, 20))
    if current is not None:
        cx = int((posemod.YAW_EDGES[0] - current.yaw) / (posemod.YAW_EDGES[0] - posemod.YAW_EDGES[-1]) * width)
        cy = int((posemod.PITCH_EDGES[-1] - current.pitch) / (posemod.PITCH_EDGES[-1] - posemod.PITCH_EDGES[0]) * ch * 3) + 4
        cx = max(6, min(width - 6, cx)); cy = max(6, min(ch * 3, cy))
        cv2.circle(panel, (cx, cy), 7, (0, 0, 0), -1); cv2.circle(panel, (cx, cy), 5, COL_CUR, -1)
    return panel


def _pick_target(cells: dict, current) -> tuple[int, int] | None:
    todo = [c for c in posemod.REQUIRED if not (cells.get(c) and cells[c].count("primary") >= CELL_DONE)]
    if not todo:
        todo = [c for c in posemod.ALL_CELLS if not (cells.get(c) and cells[c].count("primary") >= CELL_DONE)]
    if not todo:
        return None
    if current is None or current.cell() is None:
        return (2, 1) if (2, 1) in todo else sorted(todo)[0]
    cy, cp = current.cell()
    return min(todo, key=lambda c: abs(c[0] - cy) + abs(c[1] - cp))


def _hint(current, target) -> str:
    if target is None:
        return "All cells captured - press Enter to finish"
    if current is None:
        return "Face the camera so it can find you"
    c = current.cell()
    if c == target:
        return "Hold still..."
    ty, tp = posemod.cell_centre(target)
    parts = []
    if abs(ty - current.yaw) > 6:
        parts.append("turn a little " + ("left" if ty < current.yaw else "right"))
    if abs(tp - current.pitch) > 6:
        parts.append("look a little " + ("down" if tp < current.pitch else "up"))
    return "Move the dot into the highlighted cell: " + ", ".join(parts) if parts else "Almost - hold still"


def free_move_enroll(st: Store, cap: Capture, models: Models, cams: dict, mirror: bool = True) -> int:
    """Track the head, fill the pose grid, fuse per cell and per spectrum. Returns models added.

    Per-frame cost is kept to: one camera read + ROI face detection (~5 ms) + 68 landmarks
    (~5 ms) + drawing. Embeddings (the expensive part) are computed only when the current
    cell still needs samples, at most every SAMPLE_EVERY_MS, and the second camera is read
    only at those moments.
    """
    _open_window()
    models.sp68; models.sp5; models.rec                     # preload: no stall on the first face
    primary = "ir" if "ir" in cap.spectra else "rgb"
    others = [sp for sp in cap.spectra if sp != primary]
    cells: dict[tuple[int, int], Cell] = {}
    prev_pose = None
    prev_box: Box | None = None
    last_sample_t = 0.0
    finished = False
    t_all_required = None
    fps_t, fps_n, fps = time.monotonic(), 0, 0.0
    try:
        while True:
            fr = cap.read_one(primary, flush=0)
            if fr is None:
                continue
            h, w = fr.gray.shape[:2]
            q = measure(fr.gray, models.max_height)
            box = None if q.reason(cams[f"{primary}_dark_threshold"]) else models.detect(fr.gray, prev_box)
            prev_box = box
            current = None
            sampled = False
            if box is not None and box.w >= MIN_ENROLL_FACE_PX:
                p = posemod.estimate(models.landmarks68(fr.gray, box), w, h)
                if p is not None:
                    current = posemod.smooth(prev_pose, p)
                    stable = prev_pose is not None and abs(current.yaw - prev_pose.yaw) < STABLE_DEG \
                        and abs(current.pitch - prev_pose.pitch) < STABLE_DEG
                    prev_pose = current
                    cell = current.cell()
                    now = time.monotonic()
                    if cell is not None and stable and (now - last_sample_t) * 1000 >= SAMPLE_EVERY_MS:
                        c = cells.setdefault(cell, Cell())
                        if not c.full:
                            c.push(primary, q.score(), models.embed(fr.rgb, box))
                            c.n_primary += 1
                            for sp in others:
                                f2 = cap.read_one(sp, flush=1)
                                if f2 is None:
                                    continue
                                q2 = measure(f2.gray, models.max_height)
                                if q2.reason(cams[f"{sp}_dark_threshold"]):
                                    continue
                                # same module, same field of view: the primary box is a good ROI hint
                                sx, sy = f2.gray.shape[1] / w, f2.gray.shape[0] / h
                                hint = Box(int(box.x * sx), int(box.y * sy), int(box.w * sx), int(box.h * sy))
                                b2 = models.detect(f2.gray, hint)
                                if b2 is not None and b2.w >= MIN_ENROLL_FACE_PX:
                                    c.push(sp, q2.score(), models.embed(f2.rgb, b2))
                            c.poses.append((current.yaw, current.pitch))
                            last_sample_t = now
                            sampled = True
            else:
                prev_pose = None

            target = _pick_target(cells, current)
            required_done = all(cells.get(c) and cells[c].count("primary") >= CELL_DONE for c in posemod.REQUIRED)
            if required_done and t_all_required is None:
                t_all_required = time.monotonic()

            # ---- draw
            fps_n += 1
            if time.monotonic() - fps_t >= 1.0:
                fps, fps_n, fps_t = fps_n / (time.monotonic() - fps_t), 0, time.monotonic()
            img = _bgr(fr)
            if mirror:
                img = cv2.flip(img, 1)
            if box is not None:
                bx = (w - box.x - box.w) if mirror else box.x
                cv2.rectangle(img, (bx, box.y), (bx + box.w, box.y + box.h), COL_OK if sampled else COL_CUR, 2)
            _text(img, _hint(current, target), (10, 24), 0.62)
            filled = sum(1 for c in cells.values() if c.count("primary") >= CELL_DONE)
            req = sum(1 for c in posemod.REQUIRED if cells.get(c) and cells[c].count("primary") >= CELL_DONE)
            status = f"{req}/{len(posemod.REQUIRED)} required | {filled}/15 cells | {primary.upper()} {q.mean:.0f} | {fps:.0f} fps"
            if current is not None:
                status += f" | yaw {current.yaw:+.0f} pitch {current.pitch:+.0f}"
            _text(img, status, (10, h - 12), 0.5, (200, 200, 200))
            _text(img, "Enter = finish   Esc = cancel", (w - 235, 24), 0.45, (180, 180, 180))
            cv2.imshow(WIN, np.vstack([img, _grid_panel(w, cells, current, target)]))
            k = cv2.waitKey(1) & 0xFF
            if k in (13, 10):
                finished = True
            elif k == 27:
                break
            if finished or (required_done and filled >= 15) or (t_all_required and time.monotonic() - t_all_required > 12):
                finished = True
                break
    finally:
        cv2.destroyAllWindows(); cv2.waitKey(1)

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
    state, msg = "unknown", ""
    t_end = time.monotonic() + seconds if seconds else None
    if gui:
        try:
            _open_window()
        except GuiUnavailable:
            gui = False
    try:
        while True:
            fs = cap.read_set()
            if "ir" not in fs:
                continue
            ir = fs["ir"]; rgb = fs.get("rgb")
            box = models.detect(ir.gray)
            state, msg = emitter_verdict(float(ir.gray.mean()), box is not None, float(rgb.gray.mean()) if rgb else None)
            if not gui:
                if t_end is None or time.monotonic() > t_end:
                    break
                continue
            left = _bgr(ir)
            if box is not None:
                cv2.rectangle(left, (box.x, box.y), (box.x + box.w, box.y + box.h), COL_OK, 2)
            _text(left, "IR  mean %.0f" % ir.gray.mean(), (10, 24))
            panels = [left]
            if rgb is not None:
                right = cv2.resize(_bgr(rgb), (left.shape[1], left.shape[0]))
                _text(right, "RGB  mean %.0f" % rgb.gray.mean(), (10, 24))
                panels.append(right)
            canvas = np.hstack(panels)
            col = {"ok": COL_OK, "dead": (60, 60, 230), "inconclusive": COL_REQ}.get(state, COL_TXT)
            _text(canvas, msg, (10, canvas.shape[0] - 14), 0.55, col)
            _text(canvas, "q = quit", (canvas.shape[1] - 80, 24), 0.45, (180, 180, 180))
            cv2.imshow(WIN, canvas)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        if gui:
            cv2.destroyAllWindows(); cv2.waitKey(1)
    return state, msg
