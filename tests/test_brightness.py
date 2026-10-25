"""Brightness management: face metering, normalisation, exposure steering (fake camera)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ward.engine import brightness  # noqa: E402
from ward.engine.detect import Box  # noqa: E402


class MeterTests(unittest.TestCase):
    def test_face_not_frame(self):
        frame = np.full((360, 640), 200, np.uint8)          # bright room
        frame[100:260, 240:400] = 20                         # dark face
        light = brightness.meter(frame, Box(240, 100, 160, 160))
        self.assertLess(light.mean, 30)
        self.assertEqual(light.state("ir"), "dark")
        self.assertEqual(brightness.FaceLight(110, 60, 180).state("ir"), "ok")
        self.assertEqual(brightness.FaceLight(60, 30, 100).state("ir"), "dim")
        self.assertEqual(brightness.FaceLight(240, 200, 255).state("rgb"), "blown")

    def test_normalise_identity_in_range_and_lifts_dark(self):
        ok = np.random.default_rng(0).integers(80, 160, (360, 640), dtype=np.uint8)
        self.assertIs(brightness.normalise(ok), ok)
        dark = (np.random.default_rng(1).integers(0, 40, (360, 640))).astype(np.uint8)
        out = brightness.normalise(dark)
        self.assertGreater(out.mean(), dark.mean() * 2)
        rgb = cv2.cvtColor(dark, cv2.COLOR_GRAY2RGB)
        self.assertGreater(brightness.normalise_rgb(rgb, dark).mean(), rgb.mean() * 2)


class FakeCap:
    """Camera with manual exposure support, like a typical UVC webcam."""
    def __init__(self, manual=True):
        self.props = {cv2.CAP_PROP_AUTO_EXPOSURE: 3.0, cv2.CAP_PROP_EXPOSURE: 250.0, cv2.CAP_PROP_GAIN: -1.0}
        self.manual = manual
    def get(self, p):
        return self.props.get(p, -1.0)
    def set(self, p, v):
        if p == cv2.CAP_PROP_EXPOSURE and not self.manual:
            return False
        self.props[p] = float(v)
        return True


class FirmwareCap(FakeCap):
    """Like the reference IR module: no exposure controls at all."""
    def __init__(self):
        self.props = {cv2.CAP_PROP_AUTO_EXPOSURE: -1.0, cv2.CAP_PROP_EXPOSURE: -1.0, cv2.CAP_PROP_GAIN: -1.0}
        self.manual = False


class ExposureTests(unittest.TestCase):
    def test_probe_leaves_auto_and_steers_when_needed(self):
        cap = FakeCap()
        e = brightness.Exposure(cap, "rgb")
        self.assertTrue(e.manual)
        self.assertEqual(cap.props[cv2.CAP_PROP_AUTO_EXPOSURE], 3.0)     # restored to auto after probing
        self.assertEqual(cap.props[cv2.CAP_PROP_EXPOSURE], 250.0)
        # face in band: nothing happens
        self.assertFalse(e.steer(brightness.FaceLight(120, 80, 170)))
        # face too dark: takes over and raises exposure (bounded step)
        e.last_step = 0.0
        self.assertTrue(e.steer(brightness.FaceLight(40, 10, 80)))
        self.assertEqual(cap.props[cv2.CAP_PROP_AUTO_EXPOSURE], 1.0)
        self.assertGreater(cap.props[cv2.CAP_PROP_EXPOSURE], 250.0)
        self.assertLessEqual(cap.props[cv2.CAP_PROP_EXPOSURE], 250.0 * 1.6 + 1)
        e.restore()
        self.assertEqual(cap.props[cv2.CAP_PROP_AUTO_EXPOSURE], 3.0)

    def test_firmware_auto_exposure_is_left_alone(self):
        cap = FirmwareCap()
        e = brightness.Exposure(cap, "ir")
        self.assertFalse(e.manual)
        self.assertFalse(e.steer(brightness.FaceLight(20, 5, 40)))
        self.assertIn("firmware", e.describe())


if __name__ == "__main__":
    unittest.main()
