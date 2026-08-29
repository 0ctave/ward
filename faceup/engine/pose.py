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

YAW_EDGES = [-45.0, -24.0, -9.0, 9.0, 24.0, 45.0]      # 5 cells: far-left, left, centre, right, far-right
# Pitch is relative to the calibrated neutral pose and deliberately asymmetric: at a laptop you
# look far DOWN (keyboard ≈ −25°) but only a little UP before looking above the screen, and the
# PnP estimate compresses pitch. down < −10°, level −10…+6°, up > +6°.
PITCH_EDGES = [-32.0, -10.0, 6.0, 22.0]                # 3 cells: down, level, up
YAW_NAMES = ["L2", "L1", "C", "R1", "R2"]
PITCH_NAMES = ["down", "level", "up"]
REQUIRED = {(2, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (2, 2)}   # centre cross (yaw_i, pitch_i)
ALL_CELLS = [(y, p) for p in range(3) for y in range(5)]


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


def cell_centre(cell: tuple[int, int]) -> tuple[float, float]:
    y = (YAW_EDGES[cell[0]] + YAW_EDGES[cell[0] + 1]) / 2
    p = (PITCH_EDGES[cell[1]] + PITCH_EDGES[cell[1] + 1]) / 2
    return y, p


def neighbours(cell: tuple[int, int]) -> list[tuple[int, int]]:
    y, p = cell
    out = []
    for dy, dp in ((1, 0), (0, 1)):          # right and up only → each pair counted once
        n = (y + dy, p + dp)
        if 0 <= n[0] < 5 and 0 <= n[1] < 3:
            out.append(n)
    return out


def estimate(landmarks68: np.ndarray, frame_w: int, frame_h: int) -> Pose | None:
    pts = landmarks68[_LM_IDX].astype(np.float64)
    f = float(frame_w)
    K = np.array([[f, 0, frame_w / 2.0], [0, f, frame_h / 2.0], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(_MODEL_3D, pts, K, np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    # Decompose to Euler angles (degrees). RQDecomp3x3 returns (pitch, yaw, roll) for this convention.
    angles = cv2.RQDecomp3x3(R)[0]
    pitch, yaw, roll = (float(a) for a in angles)
    # normalise pitch into (-90, 90]: the PnP solution often comes out as 180 - x
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180
    pitch = -pitch          # make "looking down" negative
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
