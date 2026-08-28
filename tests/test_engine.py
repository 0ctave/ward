"""Unit tests that need no camera and no face: run with  python3 -m unittest discover tests"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("FACEUP_ETC", tempfile.mkdtemp(prefix="faceup-test-"))
os.environ.setdefault("FACEUP_LOG", os.path.join(os.environ["FACEUP_ETC"], "log"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from faceup.engine import config, pam, quality, store  # noqa: E402
from faceup.engine.detect import Box  # noqa: E402
from faceup.engine.liveness import Result, cross_spectrum, motion, shadow  # noqa: E402


def textured(h=360, w=640, seed=1):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w), dtype=np.uint8)
    return cv2.GaussianBlur(img, (0, 0), 1.2)


class ConfigTests(unittest.TestCase):
    def test_set_key_types_and_validation(self):
        cfg = config.load()
        self.assertEqual(config.set_key(cfg, "recognition.certainty", "0.5"), 0.5)
        self.assertEqual(config.set_key(cfg, "liveness.motion", "false"), False)
        self.assertEqual(config.set_key(cfg, "timing.window_ms", "900"), 900)
        with self.assertRaises(ValueError):
            config.set_key(cfg, "liveness.bogus", "1")
        with self.assertRaises(ValueError):
            config.set_key(cfg, "timing.window_ms", "fast")

    def test_roundtrip(self):
        cfg = config.load()
        cfg["cameras"]["ir"] = "/dev/video9"
        config.save(cfg)
        self.assertEqual(config.load()["cameras"]["ir"], "/dev/video9")


class StoreTests(unittest.TestCase):
    def test_roundtrip_and_distance(self):
        st = store.load("tester")
        e1 = np.ones(128) / np.sqrt(128)
        e2 = -e1
        st.add("straight", "ir", e1, 0.9)
        st.add("straight", "rgb", e2, 0.8)
        st.calib = {"dx": 0.01, "dy": 0.0, "s": 1.05}
        st.save()
        self.assertEqual(oct(os.stat(st.path).st_mode & 0o777), "0o600")
        st2 = store.load("tester")
        self.assertEqual({m.spectrum for m in st2.models}, {"ir", "rgb"})
        d, label = st2.min_distance(e1, "ir")
        self.assertAlmostEqual(d, 0.0, places=5)
        self.assertEqual(label, "straight")
        d2, _ = st2.min_distance(e1, "rgb")
        self.assertAlmostEqual(d2, 2.0, places=5)
        self.assertEqual(st2.calib["s"], 1.05)

    def test_refuses_bad_user(self):
        with self.assertRaises(store.StoreError):
            store.load("../etc")


class QualityTests(unittest.TestCase):
    def test_dark_and_blur(self):
        dark = np.full((360, 640), 10, np.uint8)
        self.assertEqual(quality.measure(dark).reason(25), "dark")
        flat = np.full((360, 640), 120, np.uint8)
        self.assertEqual(quality.measure(flat).reason(25), "blurred")
        self.assertIsNone(quality.measure(textured()).reason(25))


class CrossSpectrumTests(unittest.TestCase):
    calib = {"dx": 0.0, "dy": 0.0, "s": 1.0}
    size = (640, 360)

    def test_screen_replay_fails(self):
        v = cross_spectrum.vote(None, Box(100, 80, 150, 150), False, False, self.calib, self.size, self.size)
        self.assertEqual(v.result, Result.FAIL)

    def test_ir_print_fails(self):
        v = cross_spectrum.vote(Box(100, 80, 150, 150), None, False, False, self.calib, self.size, self.size)
        self.assertEqual(v.result, Result.FAIL)

    def test_night_abstains(self):
        v = cross_spectrum.vote(Box(100, 80, 150, 150), None, False, True, self.calib, self.size, self.size)
        self.assertEqual(v.result, Result.ABSTAIN)

    def test_aligned_passes_and_calibration_applies(self):
        v = cross_spectrum.vote(Box(100, 80, 150, 150), Box(104, 78, 150, 150), False, False, self.calib, self.size, self.size)
        self.assertEqual(v.result, Result.PASS)
        # RGB camera at 1280x720: same normalised place must still pass
        v = cross_spectrum.vote(Box(100, 80, 150, 150), Box(200, 160, 300, 300), False, False, self.calib, self.size, (1280, 720))
        self.assertEqual(v.result, Result.PASS)
        # far apart → fail
        v = cross_spectrum.vote(Box(100, 80, 150, 150), Box(400, 80, 150, 150), False, False, self.calib, self.size, self.size)
        self.assertEqual(v.result, Result.FAIL)


class MotionTests(unittest.TestCase):
    box = Box(200, 80, 220, 220)

    def test_still_abstains(self):
        g = textured()
        self.assertEqual(motion.vote(g, g.copy(), self.box).result, Result.ABSTAIN)

    def test_planar_motion_fails(self):
        g0 = textured()
        H = np.array([[1.01, 0.004, 3.0], [-0.003, 1.008, 2.0], [0.00001, 0.00002, 1.0]])
        g1 = cv2.warpPerspective(g0, H, (g0.shape[1], g0.shape[0]))
        v = motion.vote(g0, g1, self.box)
        self.assertEqual(v.result, Result.FAIL, v.reason)

    def test_non_rigid_motion_passes(self):
        g0 = textured()
        h, w = g0.shape
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        # a bulge moving differently from its surroundings (nose-vs-ears parallax caricature)
        cx, cy = 310, 190
        r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        bump = np.exp(-(r / 60.0) ** 2)
        map_x = xs - 4.0 * bump - 2.0
        map_y = ys + 3.0 * bump - 1.5
        g1 = cv2.remap(g0, map_x, map_y, cv2.INTER_LINEAR)
        v = motion.vote(g0, g1, self.box)
        self.assertEqual(v.result, Result.PASS, v.reason)


class ShadowTests(unittest.TestCase):
    box = Box(200, 80, 200, 200)

    def test_uniform_change_fails_and_gradient_passes(self):
        off = np.full((360, 640), 60, np.uint8)
        on_uniform = np.full((360, 640), 100, np.uint8)
        self.assertEqual(shadow.vote(off, on_uniform, self.box).result, Result.FAIL)
        ys = np.mgrid[0:360, 0:640][0]
        on_grad = np.clip(60 + (ys - 80) * 0.6, 0, 255).astype(np.uint8)   # strong top→bottom falloff
        self.assertEqual(shadow.vote(off, on_grad, self.box).result, Result.PASS)
        self.assertEqual(shadow.vote(off, off, self.box).result, Result.ABSTAIN)


class PamTests(unittest.TestCase):
    SAMPLE = ("# comment\nauth\t[success=3 default=ignore]\tpam_howdy.so\n"
              "auth\t[success=2 default=ignore]\tpam_unix.so nullok try_first_pass\n"
              "auth\trequisite\tpam_deny.so\n")

    def test_insert_before_first_real_module_and_remove(self):
        out = pam.render_enabled(self.SAMPLE)
        lines = out.splitlines()
        self.assertIn("faceup-auth", lines[1])
        self.assertTrue(lines[2].endswith("pam_howdy.so"))
        self.assertEqual(pam.render_enabled(out), out)         # idempotent
        self.assertEqual(pam.render_disabled(out), self.SAMPLE)


if __name__ == "__main__":
    unittest.main()
