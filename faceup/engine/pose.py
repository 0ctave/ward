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
CENTRE = (NY // 2, 1)
# Coverage requirement is continuous (see Coverage): an ellipse in (yaw, pitch) around the
# neutral pose. Corners such as far-left + up are not required — physically awkward at a
# laptop and covered by interpolation anyway.
REQ_CENTRE = (0.0, -4.0)        # degrees; slightly below level: laptop use is keyboard-biased
REQ_AXES = (34.0, 19.0)         # semi-axes yaw, pitch


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


def cell_name(cell: tuple[int, int]) -> str:
    return f"{YAW_NAMES[cell[0]]}/{PITCH_NAMES[cell[1]]}"


class Coverage:
    """Continuous coverage of the (yaw, pitch) plane: each sample paints a Gaussian footprint
    (σ = SIGMA degrees). Replaces "cell full/empty" for guidance: the user is sent toward the
    least-covered required direction, and sampling happens wherever coverage is still low."""
    STEP = 2.0
    SIGMA = 7.0
    FULL = 1.0            # local value at which a spot counts as covered

    def __init__(self):
        self.yaws = np.arange(YAW_EDGES[0], YAW_EDGES[-1] + 0.1, self.STEP)
        self.pitches = np.arange(PITCH_EDGES[0], PITCH_EDGES[-1] + 0.1, self.STEP)
        self.map = np.zeros((len(self.pitches), len(self.yaws)))
        yy, pp = np.meshgrid(self.yaws, self.pitches)
        self.required = np.vectorize(required)(yy, pp)

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
        yy, pp = np.meshgrid(self.yaws, self.pitches)
        dist = np.sqrt(((yy - yaw) / REQ_AXES[0]) ** 2 + ((pp - pitch) / REQ_AXES[1]) ** 2)
        score = np.where(self.required & (self.map < self.FULL), self.map + 0.6 * dist, np.inf)
        i, j = np.unravel_index(np.argmin(score), score.shape)
        return float(self.yaws[j]), float(self.pitches[i])


def cell_centre(cell: tuple[int, int]) -> tuple[float, float]:
    y = (YAW_EDGES[cell[0]] + YAW_EDGES[cell[0] + 1]) / 2
    p = (PITCH_EDGES[cell[1]] + PITCH_EDGES[cell[1] + 1]) / 2
    return y, p


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


def pitch_from_ratio(landmarks68: np.ndarray, roll_deg: float) -> float | None:
    """Pitch from where the nose tip sits between the eye line and the mouth line.

    The nose protrudes ~135 mm in front of the eye/mouth plane, so head pitch moves its projected
    tip up/down relative to eyes and mouth: r = (eyes_y − nose_y) / (eyes_y − mouth_y). Unlike PnP
    this ignores the chin (dlib's noisiest landmark), is independent of distance and of where the
    face sits in the frame, and averages 12 eye + 4 mouth points. Inverted in closed form:
        r·(ΔY·c + ΔZ·s) = EYE_Y·c − EYE_Z·s   with c=cos θ, s=sin θ  →  tan θ = (EYE_Y − r·ΔY)/(r·ΔZ − EYE_Z)
    """
    lm = landmarks68
    eyes = np.vstack([lm[36:42], lm[42:48]]).mean(axis=0)
    mouth = lm[[48, 54, 51, 57]].mean(axis=0)
    nose = lm[30]
    # remove roll so "vertical" is the face's own vertical
    th = np.deg2rad(-roll_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    e, m, n = (R @ (v - eyes) for v in (eyes, mouth, nose))
    span = m[1] - e[1]
    if abs(span) < 1e-3:
        return None
    r = (n[1] - e[1]) / span
    dy, dz = _EYE_Y - _MOUTH_Y, _EYE_Z - _MOUTH_Z
    return float(np.degrees(np.arctan2(_EYE_Y - r * dy, r * dz - _EYE_Z)))


def roll_from_eyes(landmarks68: np.ndarray) -> float:
    """Head roll from the eye line (degrees, positive = right eye lower in the image)."""
    le = landmarks68[36:42].mean(axis=0); re = landmarks68[42:48].mean(axis=0)
    return float(np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0])))


def yaw_from_ratio(landmarks68: np.ndarray, roll_deg: float) -> float | None:
    """Yaw from where the nose tip sits horizontally between the eyes and between the mouth corners.

    The nose protrudes in front of both lines (eyes: z=−135, mouth: z=−125 in the generic model),
    so turning the head shifts its projected tip sideways:  r_eyes = (nose_x − eyes_mid_x)/eye_span
    = (135/450)·tan ψ,  r_mouth = (nose_x − mouth_mid_x)/mouth_span = (125/300)·tan ψ.
    Chin-free, distance-independent; the two references are averaged. Measured in image axes on
    purpose: the projected eye line tilts under combined yaw+pitch (apparent roll), and
    de-rotating by it *worsens* yaw (synthetic sweep: rms 5.2° → 4.1° without de-rotation). A
    gain of YAW_GAIN corrects the orthographic formula's compression under perspective
    (rms 1.6°, max 4° over ±35° yaw × −20…+12° pitch)."""
    lm = landmarks68
    le = lm[36:42].mean(axis=0); re = lm[42:48].mean(axis=0)
    ml = lm[48]; mr = lm[54]; nose = lm[30]
    ests = []
    eye_span = re[0] - le[0]
    if abs(eye_span) > 1e-3:
        r = (nose[0] - (le[0] + re[0]) / 2) / eye_span
        ests.append(np.degrees(np.arctan(r / (135.0 / 450.0))))
    mouth_span = mr[0] - ml[0]
    if abs(mouth_span) > 1e-3:
        r = (nose[0] - (ml[0] + mr[0]) / 2) / mouth_span
        ests.append(np.degrees(np.arctan(r / (125.0 / 300.0))))
    return float(np.mean(ests)) * YAW_GAIN if ests else None


YAW_GAIN = 1.2


def estimate(landmarks68: np.ndarray, frame_w: int, frame_h: int, shape3d: np.ndarray | None = None) -> Pose | None:
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
    # generic path: no PnP at all — every angle from chin-free landmark geometry
    roll = roll_from_eyes(landmarks68)
    yaw = yaw_from_ratio(landmarks68, roll)
    pitch = pitch_from_ratio(landmarks68, roll)
    if yaw is None or pitch is None:
        return None
    return Pose(yaw, pitch, roll)


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
