"""Head pose (yaw / pitch / roll) from dlib's 68 landmarks via PnP against a generic 3D face.

Also defines the pose GRID used by free-move enrollment: 5 yaw × 3 pitch cells. The seven
cells on the centre cross are required (they are where a face sits while using a laptop),
the corners are optional. Adjacent filled cells are interpolated at match time (store.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Generic 3D face model (mm), landmark indices: nose tip 30, chin 8, eye outer corners 36/45, mouth corners 48/54
_MODEL_3D = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0),
], dtype=np.float64)
_LM_IDX = [30, 8, 36, 45, 48, 54]

YAW_EDGES = [-45.0, -33.0, -21.0, -9.0, 9.0, 21.0, 33.0, 45.0]   # 7 storage bins (12°, centre 18°)
# Pitch is relative to the calibrated neutral pose and deliberately asymmetric: at a laptop you
# look far DOWN (keyboard ≈ −25°) but only a little UP before looking above the screen, and the
# PnP estimate compresses pitch. down < −10°, level −10…+6°, up > +6°.
PITCH_EDGES = [-32.0, -10.0, 6.0, 22.0]                # 3 cells: down, level, up
YAW_NAMES = ["L3", "L2", "L1", "C", "R1", "R2", "R3"]
PITCH_NAMES = ["down", "level", "up"]
NY, NP = len(YAW_NAMES), len(PITCH_NAMES)
ALL_CELLS = [(y, p) for p in range(NP) for y in range(NY)]
# Coverage requirement is continuous (see Coverage): an ellipse in (yaw, pitch) around the
# neutral pose. Corners such as far-left + up are not required — physically awkward at a
# laptop and covered by interpolation anyway.
REQ_CENTRE = (0.0, -4.0)        # degrees; slightly below level: laptop use is keyboard-biased
REQ_AXES = (30.0, 14.0)         # semi-axes yaw, pitch — modest on purpose: reachable at a laptop


def required(yaw: float, pitch: float) -> bool:
    return ((yaw - REQ_CENTRE[0]) / REQ_AXES[0]) ** 2 + ((pitch - REQ_CENTRE[1]) / REQ_AXES[1]) ** 2 <= 1.0


# storage cells that intersect the required ellipse (for doctor/report purposes)
REQUIRED = {c for c in ALL_CELLS
            if required((YAW_EDGES[c[0]] + YAW_EDGES[c[0] + 1]) / 2, (PITCH_EDGES[c[1]] + PITCH_EDGES[c[1] + 1]) / 2)}


@dataclass
class Pose:
    yaw: float      # degrees, negative = head turned to the camera's left
    pitch: float    # degrees, negative = looking down
    roll: float

    def cell(self) -> tuple[int, int] | None:
        yi = _bin(self.yaw, YAW_EDGES)
        pi = _bin(self.pitch, PITCH_EDGES)
        return (yi, pi) if yi is not None and pi is not None else None


def _bin(v: float, edges: list[float]) -> int | None:
    for i in range(len(edges) - 1):
        if edges[i] <= v < edges[i + 1]:
            return i
    return None


def nearest_cell(yaw: float, pitch: float) -> tuple[int, int]:
    """Storage bin for a pose outside the grid: clip to the edge bins."""
    y = min(max(yaw, YAW_EDGES[0]), YAW_EDGES[-1] - 1e-3)
    p = min(max(pitch, PITCH_EDGES[0]), PITCH_EDGES[-1] - 1e-3)
    return (_bin(y, YAW_EDGES), _bin(p, PITCH_EDGES))


def cell_name(cell: tuple[int, int]) -> str:
    return f"{YAW_NAMES[cell[0]]}/{PITCH_NAMES[cell[1]]}"


class Coverage:
    """Continuous coverage of the (yaw, pitch) plane: each sample paints a Gaussian footprint
    (σ = SIGMA degrees). Guidance sends the user toward the least-covered required direction and
    sampling happens wherever coverage is still low.

    The required ellipse is ADAPTIVE: it starts at REQ_CENTRE/REQ_AXES and, once the user has
    shown a range (≥ MIN_OBS frames), re-centres on the middle of the poses actually reached and
    scales its axes to that span (never below MIN_AXES). A requirement is thus always reachable:
    a neutral captured while glancing at the camera, or an estimator that compresses "up", can
    no longer leave part of the ellipse out of reach."""
    STEP = 2.0
    SIGMA = 7.0
    FULL = 1.0            # local value at which a spot counts as covered
    MIN_OBS = 90          # frames with a face before the requirement may adapt
    ADAPTIVE = False      # a target that moves is disorienting; keep the ellipse fixed unless enabled
    MIN_AXES = (22.0, 12.0)
    SPAN_FACTOR = 0.55    # semi-axis = SPAN_FACTOR × observed (p3..p97) span

    def __init__(self):
        self.yaws = np.arange(YAW_EDGES[0], YAW_EDGES[-1] + 0.1, self.STEP)
        self.pitches = np.arange(PITCH_EDGES[0], PITCH_EDGES[-1] + 0.1, self.STEP)
        self.map = np.zeros((len(self.pitches), len(self.yaws)))
        self.yy, self.pp = np.meshgrid(self.yaws, self.pitches)
        self.centre = tuple(REQ_CENTRE)
        self.axes = tuple(REQ_AXES)
        self.obs: list[tuple[float, float]] = []
        self.adapted = False
        self._rebuild()

    def _rebuild(self) -> None:
        cy, cp = self.centre; ay, ap = self.axes
        self.required = ((self.yy - cy) / ay) ** 2 + ((self.pp - cp) / ap) ** 2 <= 1.0

    def is_required(self, yaw: float, pitch: float) -> bool:
        cy, cp = self.centre; ay, ap = self.axes
        return ((yaw - cy) / ay) ** 2 + ((pitch - cp) / ap) ** 2 <= 1.0

    def observe(self, yaw: float, pitch: float) -> None:
        """Every frame with a face (after calibration): learn the user's reachable range.
        The ellipse adapts ONCE (at MIN_OBS frames) and then stays fixed — a target that keeps
        moving is disorienting; `adapted` tells the UI when it happened."""
        self.obs.append((yaw, pitch))
        if self.ADAPTIVE and len(self.obs) == self.MIN_OBS and not self.adapted:
            self.adapted = True
            ys = np.array([o[0] for o in self.obs]); ps = np.array([o[1] for o in self.obs])
            y_lo, y_hi = np.percentile(ys, [3, 97]); p_lo, p_hi = np.percentile(ps, [3, 97])
            centre = ((y_lo + y_hi) / 2, (p_lo + p_hi) / 2)
            axes = (min(REQ_AXES[0], max(self.MIN_AXES[0], self.SPAN_FACTOR * (y_hi - y_lo))),
                    min(REQ_AXES[1], max(self.MIN_AXES[1], self.SPAN_FACTOR * (p_hi - p_lo))))
            centre = (float(np.clip(centre[0], YAW_EDGES[0] + axes[0], YAW_EDGES[-1] - axes[0])),
                      float(np.clip(centre[1], PITCH_EDGES[0] + axes[1], PITCH_EDGES[-1] - axes[1])))
            self.centre, self.axes = centre, axes
            self._rebuild()

    def add(self, yaw: float, pitch: float, weight: float = 0.5) -> None:
        dy = (self.yaws[None, :] - yaw) / self.SIGMA
        dp = (self.pitches[:, None] - pitch) / self.SIGMA
        self.map += weight * np.exp(-0.5 * (dy ** 2 + dp ** 2))

    def at(self, yaw: float, pitch: float) -> float:
        i = int(np.clip(round((pitch - self.pitches[0]) / self.STEP), 0, len(self.pitches) - 1))
        j = int(np.clip(round((yaw - self.yaws[0]) / self.STEP), 0, len(self.yaws) - 1))
        return float(self.map[i, j])

    def fraction(self) -> float:
        """Share of the required ellipse that is covered."""
        req = self.required
        return float(np.mean(np.minimum(self.map[req] / self.FULL, 1.0))) if req.any() else 1.0

    def target(self, yaw: float | None, pitch: float | None) -> tuple[float, float] | None:
        """Least-covered required direction, biased toward the current pose so hints are small moves."""
        cov = np.where(self.required, self.map, np.inf)
        if not np.isfinite(cov).any() or cov[np.isfinite(cov)].min() >= self.FULL:
            return None
        if yaw is None or pitch is None:
            i, j = np.unravel_index(np.argmin(cov), cov.shape)
            return float(self.yaws[j]), float(self.pitches[i])
        dist = np.sqrt(((self.yy - yaw) / self.axes[0]) ** 2 + ((self.pp - pitch) / self.axes[1]) ** 2)
        score = np.where(self.required & (self.map < self.FULL), self.map + 0.6 * dist, np.inf)
        i, j = np.unravel_index(np.argmin(score), score.shape)
        return float(self.yaws[j]), float(self.pitches[i])


def neighbours(cell: tuple[int, int]) -> list[tuple[int, int]]:
    y, p = cell
    out = []
    for dy, dp in ((1, 0), (0, 1)):          # right and up only → each pair counted once
        n = (y + dy, p + dp)
        if 0 <= n[0] < NY and 0 <= n[1] < NP:
            out.append(n)
    return out


# Vertical geometry of the generic model used by the pitch proxy (mm): eye line y=+170 z=-135,
# nose tip y=0 z=0, mouth line y=-150 z=-125.
_EYE_Y, _EYE_Z, _MOUTH_Y, _MOUTH_Z = 170.0, -135.0, -150.0, -125.0


def roll_from_eyes(landmarks68: np.ndarray) -> float:
    """Head roll from the eye line (degrees, positive = right eye lower in the image)."""
    le = landmarks68[36:42].mean(axis=0); re = landmarks68[42:48].mean(axis=0)
    return float(np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0])))


# Generic-model geometry behind each cue (mm): depth of the reference line below the nose tip
# and its span. r = (depth/span)·tan(angle) under orthography.
_CUES = {
    "yaw_eyes":  (135.0, 450.0),    # eye line z=-135, outer-corner span 450
    "yaw_mouth": (125.0, 300.0),    # mouth corners z=-125, span 300
    "yaw_wings": (50.0, 90.0),      # nose wings z=-50, span 90 — local, pitch-insensitive
}
_PITCH_R0 = 170.0 / 320.0           # nose-tip position between eye and mouth lines at neutral (generic)


def ratios(landmarks68: np.ndarray, roll_deg: float) -> dict[str, float]:
    """Raw geometric ratios from the landmarks (pitch de-rotated by roll; yaw cues in image axes —
    see yaw notes). All chin-free, distance-independent."""
    lm = landmarks68
    out: dict[str, float] = {}
    le = lm[36:42].mean(axis=0); re = lm[42:48].mean(axis=0); nose = lm[30]
    ml, mr = lm[48], lm[54]
    wl, wr = lm[31], lm[35]
    for key, (a, b) in (("yaw_eyes", (le, re)), ("yaw_mouth", (ml, mr)), ("yaw_wings", (wl, wr))):
        span = b[0] - a[0]
        if abs(span) > 1e-3:
            out[key] = float((nose[0] - (a[0] + b[0]) / 2) / span)
    th = np.deg2rad(-roll_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    e, m, n = (R @ (v - le) for v in ((le + re) / 2, (ml + mr + lm[51] + lm[57]) / 4, nose))
    span = m[1] - e[1]
    if abs(span) > 1e-3:
        out["pitch"] = float((n[1] - e[1]) / span)
    return out


@dataclass
class Neutral:
    """The user's own ratios at their neutral pose (median over the calibration frames)."""
    r: dict[str, float]

    def as_offsets(self) -> tuple[float, float]:
        """Equivalent angle offsets for display/report: what the generic mapping would have said."""
        p = angles_from_ratios(self.r)
        return (p.yaw, p.pitch) if p else (0.0, 0.0)


def _pitch_from_r(r: float) -> float:
    dy, dz = _EYE_Y - _MOUTH_Y, _EYE_Z - _MOUTH_Z
    return float(np.degrees(np.arctan2(_EYE_Y - r * dy, r * dz - _EYE_Z)))


def angles_from_ratios(r: dict[str, float], neutral: "Neutral | None" = None) -> Pose | None:
    """Invert the ratios into angles. With `neutral`, each ratio is first shifted so the user's
    neutral maps onto the generic model's neutral — the inversion then works in its linear
    region for every user instead of after a large angle offset. Roll must be added by the caller."""
    ests = []
    for key, (depth, span) in _CUES.items():
        if key not in r:
            continue
        val = r[key]
        if neutral is not None and key in neutral.r:
            val = val - neutral.r[key]              # generic neutral yaw ratio is 0 (symmetric face)
        ests.append(np.degrees(np.arctan(val / (depth / span))))
    if not ests or "pitch" not in r:
        return None
    yaw = float(np.mean(ests)) * YAW_GAIN
    pr = r["pitch"]
    if neutral is not None and "pitch" in neutral.r:
        pr = pr - neutral.r["pitch"] + _PITCH_R0
    return Pose(yaw, _pitch_from_r(pr), 0.0)


YAW_GAIN = 1.2


def estimate(landmarks68: np.ndarray, frame_w: int, frame_h: int, shape3d: np.ndarray | None = None,
             neutral: "Neutral | None" = None) -> Pose | None:
    """Head pose. With `shape3d` (the user's personal rigid-landmark model from sfm.calibrate) all
    three angles come from a 13-point PnP against the user's own geometry; without it, yaw/roll
    come from the generic 6-point PnP and pitch from the chin-free nose-ratio proxy."""
    f = float(frame_w)
    K = np.array([[f, 0, frame_w / 2.0], [0, f, frame_h / 2.0], [0, 0, 1]], dtype=np.float64)
    if shape3d is not None:
        from .sfm import RIGID_IDX
        pts = landmarks68[RIGID_IDX].astype(np.float64)
        ok, rvec, _ = cv2.solvePnP(np.asarray(shape3d, dtype=np.float64), pts, K, np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = (float(a) for a in cv2.RQDecomp3x3(R)[0])
        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180
        return Pose(yaw, -pitch, roll)
    # generic path: no PnP at all — every angle from chin-free landmark geometry, in ratio space
    roll = roll_from_eyes(landmarks68)
    p = angles_from_ratios(ratios(landmarks68, roll), neutral)
    if p is None:
        return None
    return Pose(p.yaw, p.pitch, roll)


def smooth(prev: Pose | None, new: Pose, alpha: float = 0.7) -> Pose:
    if prev is None:
        return new
    return Pose(prev.yaw + alpha * (new.yaw - prev.yaw), prev.pitch + alpha * (new.pitch - prev.pitch),
                prev.roll + alpha * (new.roll - prev.roll))


class OneEuro:
    """One Euro filter (Casiez et al. 2012): adaptive low-pass — smooth when the signal is
    still, responsive when it moves. Tuned for head angles in degrees at ~30 Hz."""

    def __init__(self, min_cutoff: float = 0.6, beta: float = 0.05, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * 3.141592653589793 * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self.x_prev is None or self.t_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-3)
        self.t_prev = t
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat


class PoseFilter:
    """One Euro per angle; call reset() when the face is lost so it doesn't drag old state in."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.f = [OneEuro(), OneEuro(), OneEuro()]

    def __call__(self, p: Pose, t: float) -> Pose:
        return Pose(self.f[0](p.yaw, t), self.f[1](p.pitch, t), self.f[2](p.roll, t))


class LandmarkFilter:
    """One Euro filter on every landmark coordinate (and the face box) BEFORE the pose is
    computed: the ratio estimators pass landmark jitter straight through, so smoothing the
    input is far more effective than smoothing the angles. reset() when the face is lost."""

    def __init__(self, n: int = 68, min_cutoff: float = 1.2, beta: float = 0.01):
        self.n, self.mc, self.beta = n, min_cutoff, beta
        self.reset()

    def reset(self) -> None:
        self.f = [[OneEuro(self.mc, self.beta), OneEuro(self.mc, self.beta)] for _ in range(self.n)]
        self.box = [OneEuro(1.0, 0.01) for _ in range(4)]

    def __call__(self, lm: np.ndarray, t: float) -> np.ndarray:
        out = np.empty_like(lm)
        for i in range(self.n):
            out[i, 0] = self.f[i][0](float(lm[i, 0]), t)
            out[i, 1] = self.f[i][1](float(lm[i, 1]), t)
        return out

    def smooth_box(self, box, t: float):
        vals = [self.box[i](float(v), t) for i, v in enumerate((box.x, box.y, box.w, box.h))]
        return type(box)(int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3]))


class Stability:
    """Pose is 'stable' when the last N filtered poses stay within `deg` in yaw and pitch —
    a windowed criterion instead of frame-to-frame, which flicker never satisfies."""

    def __init__(self, n: int = 6, deg: float = 7.0):
        self.n, self.deg = n, deg
        self.hist: list[tuple[float, float]] = []

    def reset(self) -> None:
        self.hist.clear()

    def push(self, p: Pose) -> bool:
        self.hist.append((p.yaw, p.pitch))
        del self.hist[:-self.n]
        if len(self.hist) < self.n:
            return False
        ys = [h[0] for h in self.hist]; ps = [h[1] for h in self.hist]
        return (max(ys) - min(ys)) <= self.deg and (max(ps) - min(ps)) <= self.deg
