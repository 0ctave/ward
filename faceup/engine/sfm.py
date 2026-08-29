"""Personal 3D face model from enrollment landmarks (post-calibration, DESIGN.md §8.6).

The real-time head pose uses a *generic* 3D head, so it carries a per-user bias (nose length,
eye–mouth spacing…). Enrollment records 68 landmarks for every processed frame across many
poses; here a batch structure-from-motion solve over the RIGID landmarks recovers the user's
own 3D geometry and the true rotation of every frame:

  1. Tomasi–Kanade factorization (scaled orthography): the centred 2F×P measurement matrix has
     rank 3 → SVD → affine motion M̂ (2F×3) and shape Ŝ (3×P).
  2. Metric upgrade: find Q with M̂Q having per-frame rows i_f ⟂ j_f, |i_f| = |j_f| (a linear
     least-squares problem in L = QQᵀ), then M = M̂Q, S = Q⁻¹Ŝ.
  3. Align S to the generic model's frame with a similarity Procrustes fit on the shared
     landmarks (reflection allowed — orthography cannot tell a face from its mirror image, the
     generic model settles it).

The result is a P×3 model in the same frame and conventions as the generic one, so the
real-time estimator can simply run PnP against it (pose.estimate(..., shape3d=S)).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Landmarks that move as a rigid body with the skull (no jaw, chin, eyelids, lips, brows).
RIGID_IDX = [36, 39, 42, 45, 27, 28, 29, 30, 31, 33, 35, 48, 54]
# Their coordinates in the generic model where known (mm) — used to align the recovered shape.
_GENERIC = {36: (-225.0, 170.0, -135.0), 45: (225.0, 170.0, -135.0), 30: (0.0, 0.0, 0.0),
            48: (-150.0, -150.0, -125.0), 54: (150.0, -150.0, -125.0)}
MIN_FRAMES = 40
MIN_YAW_SPREAD = 25.0     # degrees of yaw range needed for a well-conditioned solve
MIN_PITCH_SPREAD = 12.0


class CalibrationError(Exception):
    pass


@dataclass
class Calibration:
    shape3d: np.ndarray            # len(RIGID_IDX) × 3, generic frame, mm
    frames: int
    rms_px: float                  # reprojection residual of the factorization
    report: dict                   # accuracy of the fast estimator vs the personal model


def factorize(tracks: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """tracks: F×P×2 pixel positions of the rigid landmarks.
    Returns (S 3×P in an arbitrary metric frame, M 2F×3 scaled rotations, rms residual px)."""
    F, P, _ = tracks.shape
    W = np.vstack([tracks[:, :, 0], tracks[:, :, 1]]).astype(np.float64)   # 2F×P: all u rows then all v rows
    W -= W.mean(axis=1, keepdims=True)
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    M_hat = U[:, :3] * np.sqrt(s[:3])
    S_hat = (np.sqrt(s[:3])[:, None]) * Vt[:3]
    # metric constraints on L = QQᵀ (symmetric, 6 unknowns): iᵀLi = jᵀLj, iᵀLj = 0, |i_0|² = 1
    def row(a, b):
        return np.array([a[0] * b[0], a[0] * b[1] + a[1] * b[0], a[0] * b[2] + a[2] * b[0],
                         a[1] * b[1], a[1] * b[2] + a[2] * b[1], a[2] * b[2]])
    A, y = [], []
    for f in range(F):
        i, j = M_hat[f], M_hat[F + f]
        A.append(row(i, i) - row(j, j)); y.append(0.0)
        A.append(row(i, j)); y.append(0.0)
    A.append(row(M_hat[0], M_hat[0])); y.append(1.0)
    l, *_ = np.linalg.lstsq(np.array(A), np.array(y), rcond=None)
    L = np.array([[l[0], l[1], l[2]], [l[1], l[3], l[4]], [l[2], l[4], l[5]]])
    w, V = np.linalg.eigh(L)
    w = np.clip(w, 1e-9, None)               # L must be positive definite; clamp numerical noise
    Q = V @ np.diag(np.sqrt(w))
    M = M_hat @ Q
    S = np.linalg.solve(Q, S_hat)
    rms = float(np.sqrt(np.mean((M @ S - W) ** 2)))
    return S, M, rms


def procrustes(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity transform (scale, R, t) minimising |s·R·src + t − dst|; reflection allowed."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    A, B = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(A.T @ B)
    R = (U @ Vt).T
    s = sig.sum() / (A ** 2).sum()
    t = mu_d - s * (R @ mu_s)
    return s, R, t


def personal_model(tracks: np.ndarray) -> tuple[np.ndarray, float]:
    """Rigid-landmark 3D model (len(RIGID_IDX)×3) in the generic frame, plus factorization rms."""
    if tracks.shape[0] < MIN_FRAMES:
        raise CalibrationError(f"need ≥ {MIN_FRAMES} frames with a face, have {tracks.shape[0]}")
    S, _, rms = factorize(tracks)
    S = S.T                                    # P×3
    idx = [RIGID_IDX.index(k) for k in _GENERIC]
    s, R, t = procrustes(S[idx], np.array([_GENERIC[k] for k in _GENERIC]))
    return (s * (R @ S.T)).T + t, rms


def pose_spread(poses: list[tuple[float, float]]) -> tuple[float, float]:
    if not poses:
        return 0.0, 0.0
    ys = [p[0] for p in poses]; ps = [p[1] for p in poses]
    return max(ys) - min(ys), max(ps) - min(ps)


def calibrate(tracks68: np.ndarray, frame_w: int, frame_h: int, fast_poses: list[tuple[float, float]] | None = None) -> Calibration:
    """tracks68: F×68×2. Builds the personal model and, if the fast estimator's (yaw, pitch) per
    frame is given, reports its error against poses re-estimated with the personal model."""
    from . import pose as posemod
    rigid = tracks68[:, RIGID_IDX, :]
    shape, rms = personal_model(rigid)
    report: dict = {"frames": int(tracks68.shape[0]), "rms_px": round(rms, 2)}
    if fast_poses is not None and len(fast_poses) == tracks68.shape[0]:
        err_y, err_p = [], []
        for f in range(tracks68.shape[0]):
            p = posemod.estimate(tracks68[f], frame_w, frame_h, shape3d=shape)
            if p is None:
                continue
            err_y.append(fast_poses[f][0] - p.yaw); err_p.append(fast_poses[f][1] - p.pitch)
        if err_y:
            ey, ep = np.array(err_y), np.array(err_p)
            report.update(yaw_bias=round(float(ey.mean()), 1), yaw_rms=round(float(np.sqrt((ey ** 2).mean())), 1),
                          pitch_bias=round(float(ep.mean()), 1), pitch_rms=round(float(np.sqrt((ep ** 2).mean())), 1))
    return Calibration(shape, int(tracks68.shape[0]), rms, report)
