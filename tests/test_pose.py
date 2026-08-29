"""Pose grid, PnP head-pose estimate on synthetic landmarks, store interpolation."""
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

from faceup.engine import pose, store  # noqa: E402


def synth_landmarks(yaw_deg: float, pitch_deg: float, w=640, h=360) -> np.ndarray:
    """Project the generic 3D face at a known rotation into a 68-point array (only the 6 used points matter)."""
    ry = np.deg2rad(yaw_deg); rp = np.deg2rad(-pitch_deg)     # estimate() flips pitch so 'down' is negative
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(rp), -np.sin(rp)], [0, np.sin(rp), np.cos(rp)]])
    R = Ry @ Rx
    rvec, _ = cv2.Rodrigues(R)
    tvec = np.array([[0.0], [0.0], [1400.0]])
    K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    pts, _ = cv2.projectPoints(pose._MODEL_3D, rvec, tvec, K, np.zeros(4))
    lm = np.zeros((68, 2))
    for i, idx in enumerate(pose._LM_IDX):
        lm[idx] = pts[i, 0]
    # eye sets = their outer corners, mouth top/bottom = mouth centre (all on the model's eye/mouth lines)
    lm[36:42] = lm[36]; lm[42:48] = lm[45]
    lm[51] = lm[57] = (lm[48] + lm[54]) / 2
    return lm


class PoseTests(unittest.TestCase):
    def test_bins_and_names(self):
        self.assertEqual(pose.Pose(0, 0, 0).cell(), (2, 1))
        self.assertEqual(pose.Pose(-30, 0, 0).cell(), (0, 1))
        self.assertEqual(pose.Pose(15, -20, 0).cell(), (3, 0))
        self.assertIsNone(pose.Pose(60, 0, 0).cell())
        self.assertEqual(pose.cell_name((2, 1)), "C/level")
        self.assertIn((2, 1), pose.REQUIRED)
        self.assertEqual(len(pose.ALL_CELLS), 15)

    def test_neighbours_each_pair_once(self):
        pairs = {(c, n) for c in pose.ALL_CELLS for n in pose.neighbours(c)}
        self.assertEqual(len(pairs), 4 * 3 + 5 * 2)   # horizontal + vertical adjacencies
        self.assertNotIn(((1, 1), (0, 1)), pairs)

    def test_estimate_recovers_known_rotation(self):
        for yaw, pitch in ((0, 0), (20, 0), (-25, 0), (0, -15), (15, 12)):
            p = pose.estimate(synth_landmarks(yaw, pitch), 640, 360)
            self.assertIsNotNone(p)
            self.assertAlmostEqual(p.yaw, yaw, delta=4, msg=f"yaw for {(yaw, pitch)} → {p}")
            self.assertAlmostEqual(p.pitch, pitch, delta=4, msg=f"pitch for {(yaw, pitch)} → {p}")


class InterpolationTests(unittest.TestCase):
    def test_virtual_models_between_adjacent_cells(self):
        st = store.load("interp")
        a = np.zeros(128); a[0] = 1.0
        b = np.zeros(128); b[1] = 1.0
        st.add("C/level", "ir", a, 0.9, pose=(0.0, 0.0))
        st.add("R1/level", "ir", b, 0.8, pose=(15.0, 0.0))
        st.add("L2/up", "ir", -a, 0.8, pose=(-30.0, 20.0))         # not adjacent to either
        v = st.virtuals("ir")
        self.assertEqual(len(v), 1)
        self.assertTrue(v[0].virtual)
        mid = (a + b) / np.linalg.norm(a + b)
        d, label = st.min_distance(mid, "ir")
        self.assertAlmostEqual(d, 0.0, places=6)
        self.assertTrue(label.startswith("~"))
        st.save()
        st2 = store.load("interp")
        self.assertEqual(len(st2.models), 3)                         # virtuals are never persisted
        self.assertEqual(st2.models[0].pose, (0.0, 0.0))



class EmitterVerdictTests(unittest.TestCase):
    def test_inference_from_rgb_vs_ir(self):
        from faceup.gui import emitter_verdict
        self.assertEqual(emitter_verdict(80, True, 120)[0], "inconclusive")   # lights on: can't tell
        self.assertEqual(emitter_verdict(70, True, 20)[0], "ok")              # dark room, IR sees a face
        self.assertEqual(emitter_verdict(12, False, 20)[0], "dead")           # dark room, IR dark too
        self.assertEqual(emitter_verdict(70, True, None)[0], "unknown")       # no RGB to compare



class YuNetTests(unittest.TestCase):
    def test_bundled_model_loads_and_is_default(self):
        from faceup.engine import detect
        self.assertTrue(detect.YUNET_MODEL.exists(), detect.YUNET_MODEL)
        self.assertTrue(detect.yunet_available())
        m = detect.Models()
        self.assertEqual(m.detector_kind, "yunet")
        blank = np.full((360, 640), 120, np.uint8)
        self.assertIsNone(m.detect(blank))            # no face → None, and the net initialises fine



class OneEuroTests(unittest.TestCase):
    def test_smooths_noise_but_follows_motion(self):
        import random
        rng = random.Random(3)
        f = pose.OneEuro()
        # still head with ±1.5° landmark noise → output jitter well under the noise
        out = [f(10 + rng.uniform(-1.5, 1.5), i / 30) for i in range(90)]
        tail = out[30:]
        self.assertLess(max(tail) - min(tail), 1.3)      # ±1.5° input noise → ≤1.3° output p2p
        # fast turn 10° → 40° over 0.5 s: must follow within a few degrees by the end
        f2 = pose.OneEuro()
        for i in range(15):
            y = f2(10 + 30 * i / 14, i / 30)
        self.assertGreater(y, 33)


if __name__ == "__main__":
    unittest.main()
