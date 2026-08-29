"""Personal 3D model recovery from synthetic enrollment tracks (structure from motion)."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("FACEUP_ETC", tempfile.mkdtemp(prefix="faceup-test-"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from faceup.engine import pose, sfm  # noqa: E402

# A plausible personal face (mm) for the 13 rigid landmarks, in the generic frame but with
# individual proportions: longer nose, narrower eyes than the generic model.
TRUE = {
    36: (-200, 165, -140), 39: (-80, 160, -120), 42: (80, 160, -120), 45: (200, 165, -140),
    27: (0, 150, -70), 28: (0, 105, -45), 29: (0, 55, -20), 30: (0, 0, 0),
    31: (-45, -25, -50), 33: (0, -35, -40), 35: (45, -25, -50),
    48: (-140, -150, -130), 54: (140, -150, -130),
}
W, H = 640, 360


def rotation(yaw, pitch, roll):
    ry, rp, rr = np.deg2rad(yaw), np.deg2rad(-pitch), np.deg2rad(roll)
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(rp), -np.sin(rp)], [0, np.sin(rp), np.cos(rp)]])
    Rz = np.array([[np.cos(rr), -np.sin(rr), 0], [np.sin(rr), np.cos(rr), 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def synth_tracks(n=120, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    S = np.array([TRUE[k] for k in sfm.RIGID_IDX], dtype=np.float64)
    K = np.array([[W, 0, W / 2], [0, W, H / 2], [0, 0, 1]], dtype=np.float64)
    tracks, angles = [], []
    for _ in range(n):
        yaw, pitch, roll = rng.uniform(-35, 35), rng.uniform(-22, 18), rng.uniform(-8, 8)
        rvec, _ = cv2.Rodrigues(rotation(yaw, pitch, roll))
        tvec = np.array([[rng.uniform(-80, 80)], [rng.uniform(-60, 60)], [rng.uniform(1100, 1700)]])
        pts, _ = cv2.projectPoints(S, rvec, tvec, K, np.zeros(4))
        pts = pts[:, 0] + rng.normal(0, noise, (len(S), 2))
        lm = np.zeros((68, 2))
        lm[sfm.RIGID_IDX] = pts
        tracks.append(lm); angles.append((yaw, pitch, roll))
    return np.array(tracks), angles, S


class SfmTests(unittest.TestCase):
    def test_recovers_personal_shape_and_poses(self):
        tracks, angles, S_true = synth_tracks()
        cal = sfm.calibrate(tracks, W, H)
        self.assertEqual(cal.shape3d.shape, (len(sfm.RIGID_IDX), 3))
        self.assertLess(cal.rms_px, 2.5)
        # shape: compare after the best similarity alignment (SfM has a global scale/rotation freedom
        # that the generic-frame alignment only fixes up to the generic model's own proportions)
        s, R, t = sfm.procrustes(cal.shape3d, S_true)
        aligned = (s * (R @ cal.shape3d.T)).T + t
        err = np.sqrt(np.mean(np.sum((aligned - S_true) ** 2, axis=1)))
        self.assertLess(err, 12.0, f"shape rms error {err:.1f} mm")   # ~1 px noise at 1.4 m ≈ 3 mm
        # poses re-estimated with the personal model: better than the generic estimator
        e_personal, e_generic = [], []
        for lm, (yaw, pitch, roll) in zip(tracks, angles):
            pp = pose.estimate(lm, W, H, shape3d=cal.shape3d)
            e_personal.append((pp.yaw - yaw, pp.pitch - pitch))
        e_personal = np.array(e_personal)
        rms_yaw, rms_pitch = np.sqrt((e_personal ** 2).mean(axis=0))
        self.assertLess(rms_yaw, 3.0, f"yaw rms {rms_yaw:.2f}")
        self.assertLess(rms_pitch, 3.0, f"pitch rms {rms_pitch:.2f}")

    def test_needs_enough_frames(self):
        tracks, _, _ = synth_tracks(n=10)
        with self.assertRaises(sfm.CalibrationError):
            sfm.calibrate(tracks, W, H)


if __name__ == "__main__":
    unittest.main()
