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
    wings, _ = cv2.projectPoints(np.array([(-45.0, -25.0, -50.0), (45.0, -25.0, -50.0)]), rvec, tvec, K, np.zeros(4))
    lm[31], lm[35] = wings[0, 0], wings[1, 0]
    # eye sets = their outer corners, mouth top/bottom = mouth centre (all on the model's eye/mouth lines)
    lm[36:42] = lm[36]; lm[42:48] = lm[45]
    lm[51] = lm[57] = (lm[48] + lm[54]) / 2
    return lm


class PoseTests(unittest.TestCase):
    def test_bins_and_names(self):
        self.assertEqual(pose.Pose(0, 0, 0).cell(), (3, 1))
        self.assertEqual(pose.Pose(-30, 0, 0).cell(), (1, 1))
        self.assertEqual(pose.Pose(15, -20, 0).cell(), (4, 0))
        self.assertIsNone(pose.Pose(60, 0, 0).cell())
        self.assertEqual(pose.cell_name((3, 1)), "C/level")
        self.assertIn((3, 1), pose.REQUIRED)
        self.assertNotIn((0, 2), pose.REQUIRED)          # far-left + up: not required
        self.assertEqual(len(pose.ALL_CELLS), 21)
        self.assertTrue(pose.required(0, 0)); self.assertTrue(pose.required(-20, -12)); self.assertFalse(pose.required(-40, 15))

    def test_neighbours_each_pair_once(self):
        pairs = {(c, n) for c in pose.ALL_CELLS for n in pose.neighbours(c)}
        self.assertEqual(len(pairs), 6 * 3 + 7 * 2)   # horizontal + vertical adjacencies
        self.assertNotIn(((1, 1), (0, 1)), pairs)

    def test_estimate_recovers_known_rotation(self):
        for yaw, pitch in ((0, 0), (20, 0), (-25, 0), (0, -15), (15, 12), (-35, 0), (35, -10)):
            p = pose.estimate(synth_landmarks(yaw, pitch), 640, 360)
            self.assertIsNotNone(p)
            self.assertAlmostEqual(p.yaw, yaw, delta=5, msg=f"yaw for {(yaw, pitch)} → {p}")
            self.assertAlmostEqual(p.pitch, pitch, delta=4, msg=f"pitch for {(yaw, pitch)} → {p}")
        self.assertAlmostEqual(pose.estimate(synth_landmarks(0, 0), 640, 360).roll, 0.0, delta=1)

    def test_ratio_space_neutral_handles_other_proportions(self):
        """A face unlike the generic model: with its neutral ratios calibrated, angles come out
        right across the range — including the top rows that a post-hoc angle offset compresses."""
        W, H = 640, 360
        K = np.array([[W, 0, W / 2], [0, W, H / 2], [0, 0, 1]], dtype=np.float64)
        other = {36: (-190, 150, -150), 45: (210, 160, -150), 30: (0, 10, 0), 48: (-130, -120, -140),
                 54: (150, -125, -140), 31: (-40, -20, -60), 35: (50, -22, -60), 8: (0, -300, -80)}

        def lm_at(yaw, pitch):
            ry, rp = np.deg2rad(yaw), np.deg2rad(-pitch)
            Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
            Rx = np.array([[1, 0, 0], [0, np.cos(rp), -np.sin(rp)], [0, np.sin(rp), np.cos(rp)]])
            rvec, _ = cv2.Rodrigues(Ry @ Rx)
            pts, _ = cv2.projectPoints(np.array([other[i] for i in other], float), rvec, np.array([[0.0], [0.0], [1400.0]]), K, np.zeros(4))
            lm = np.zeros((68, 2))
            for i, pt in zip(other, pts[:, 0]):
                lm[i] = pt
            lm[36:42] = lm[36]; lm[42:48] = lm[45]; lm[51] = lm[57] = (lm[48] + lm[54]) / 2
            return lm

        neutral = pose.Neutral(pose.ratios(lm_at(0, 0), 0.0))
        p0 = pose.estimate(lm_at(0, 0), W, H, neutral=neutral)
        self.assertAlmostEqual(p0.yaw, 0, delta=1); self.assertAlmostEqual(p0.pitch, 0, delta=1)   # neutral maps to 0
        for yaw, pitch in ((0, 18), (-25, 15), (30, -15), (0, -20), (35, 10)):
            p = pose.estimate(lm_at(yaw, pitch), W, H, neutral=neutral)
            # a longer nose moves more per degree: without a personal model the gain is off by up to
            # ~25 % on this face, so tolerance grows with the angle (the direction and the top rows are right)
            self.assertAlmostEqual(p.yaw, yaw, delta=3 + 0.3 * abs(yaw), msg=f"yaw at {(yaw, pitch)} → {p}")
            self.assertAlmostEqual(p.pitch, pitch, delta=3 + 0.3 * abs(pitch), msg=f"pitch at {(yaw, pitch)} → {p}")
        # without calibration this face reads far from neutral even when it is neutral
        p0 = pose.estimate(lm_at(0, 0), W, H)
        self.assertGreater(abs(p0.pitch) + abs(p0.yaw), 4)

    def test_display_space_is_applied_exactly_once(self):
        from faceup import gui
        cam = pose.Pose(-20.0, -5.0, 3.0)                    # user turned to THEIR right (nose camera-left)
        d = gui.to_display(cam, -1.0)
        self.assertEqual((d.yaw, d.pitch, d.roll), (20.0, -5.0, -3.0))   # → right of centre on the panel
        self.assertEqual(gui.to_display(cam, 1.0).yaw, -20.0)            # unmirrored preview: unchanged
        src = open(gui.__file__).read()
        self.assertIn('kind="enroll", sgn=1.0', src)                     # the view must not flip again
        self.assertIn("_render_enroll(snap, 1.0)", src)

    def test_coverage_requirement_follows_the_user(self):
        c = pose.Coverage()
        c.ADAPTIVE = True
        self.assertEqual(c.centre, tuple(pose.REQ_CENTRE))
        fixed = pose.Coverage()
        for _ in range(200):
            fixed.observe(10.0, -25.0)
        self.assertEqual(fixed.centre, tuple(pose.REQ_CENTRE))          # default: the ellipse never moves
        # a user whose readings are biased down: reaches −30…+4 pitch, −25…+25 yaw
        rng = np.random.default_rng(0)
        for _ in range(200):
            c.observe(float(rng.uniform(-25, 25)), float(rng.uniform(-30, 4)))
        self.assertTrue(c.adapted)
        self.assertLess(c.centre[1], -8)                     # ellipse moved down to the user's range
        centre_after = c.centre
        for _ in range(100):
            c.observe(20.0, 10.0)
        self.assertEqual(c.centre, centre_after)               # …and then stays put
        self.assertLessEqual(c.axes[1], pose.REQ_AXES[1])      # and never larger than the default
        self.assertGreaterEqual(c.axes[1], c.MIN_AXES[1])
        self.assertTrue(c.is_required(0, -25))                 # bottom of the shown range is required…
        self.assertFalse(c.is_required(0, 14))                 # …the unreachable top is not

    def test_nearest_cell_clips(self):
        self.assertEqual(pose.nearest_cell(60, 0), (pose.NY - 1, 1))
        self.assertEqual(pose.nearest_cell(-60, -50), (0, 0))
        self.assertEqual(pose.nearest_cell(0, 0), pose.Pose(0, 0, 0).cell())

    def test_gui_uses_only_existing_pose_api(self):
        import re
        from faceup import gui
        src = open(gui.__file__).read()
        for name in set(re.findall(r"posemod\.([A-Za-z_]+)", src)):
            self.assertTrue(hasattr(pose, name), f"gui.py references pose.{name} which does not exist")

    def test_coverage_map(self):
        c = pose.Coverage()
        self.assertLess(c.fraction(), 0.05)
        for _ in range(3):
            c.add(0, 0)
        self.assertGreater(c.at(0, 0), c.FULL)
        self.assertGreater(c.fraction(), 0.05)
        t = c.target(0, 0)
        self.assertIsNotNone(t)
        self.assertTrue(pose.required(*t))
        # paint the whole required ellipse → complete
        for y in range(-40, 41, 6):
            for p in range(-30, 21, 6):
                if pose.required(y, p):
                    c.add(y, p, 2.0)
        self.assertGreater(c.fraction(), 0.95)
        self.assertIsNone(c.target(0, 0))


class InterpolationTests(unittest.TestCase):
    def test_virtual_models_between_adjacent_cells(self):
        st = store.load("interp")
        a = np.zeros(128); a[0] = 1.0
        b = np.zeros(128); b[1] = 1.0
        st.add("C/level", "ir", a, 0.9, pose=(0.0, 0.0))
        st.add("R1/level", "ir", b, 0.8, pose=(15.0, 0.0))
        st.add("L2/up", "ir", -a, 0.8, pose=(-30.0, 15.0))         # not adjacent to either
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

    def test_model_per_opencv_version_and_pinned(self):
        import hashlib
        from unittest import mock
        from faceup.engine import detect
        pins = {detect.YUNET_2022: "50ef07f702a31741ca46a4c0d947773b64143b9362780237bf0d427d6c79bab7",
                detect.YUNET_2023: "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"}
        for p, h in pins.items():
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), h, p.name)
        for ver, first in (("4.5.4", detect.YUNET_2022), ("4.6.0", detect.YUNET_2022),
                           ("4.13.0", detect.YUNET_2023), ("5.0.0", detect.YUNET_2023)):
            with mock.patch.object(detect.cv2, "__version__", ver):
                self.assertEqual(detect.yunet_candidates()[0], first, ver)

    def test_unrunnable_graph_retries_the_other_one(self):
        from faceup.engine import detect
        m = detect.Models()
        good = m._yunet_models[0]
        m._yunet_models = [detect._MODELS_DIR / "LICENSE-Apache-2.0.txt", good]   # not a graph → cv2.error
        self.assertIsNone(m.detect(np.full((360, 640), 120, np.uint8)))
        self.assertEqual(m._yunet_models, [good]); self.assertEqual(m.detector_kind, "yunet")



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
