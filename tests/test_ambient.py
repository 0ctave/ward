"""Dark-room camera rule: the ambient light sensor picks IR-only, and the Capture watchdog drops
RGB and reopens IR when RGB starves the IR stream (measured on the reference module)."""
from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from faceup.engine import ambient, capture  # noqa: E402


def fake_iio(root: Path, readings: list[tuple[float, float]]) -> None:
    for i, (raw, scale) in enumerate(readings):
        d = root / f"iio:device{i}"
        d.mkdir(parents=True)
        (d / "name").write_text("als\n")
        (d / "in_illuminance_raw").write_text(f"{raw:g}\n")
        (d / "in_illuminance_scale").write_text(f"{scale}\n")
        (d / "in_illuminance_offset").write_text("0\n")


class AmbientTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="faceup-iio-"))
        self.env = mock.patch.dict(os.environ, {"FACEUP_IIO": str(self.tmp)})
        self.env.start()
        self.mod = mock.patch.object(ambient, "same_module", return_value=True)   # a Windows-Hello module
        self.mod.start()

    def tearDown(self):
        self.env.stop(); self.mod.stop()

    def test_no_sensor(self):
        self.assertIsNone(ambient.lux())
        devs = {"ir": "/dev/video2", "rgb": "/dev/video0"}
        self.assertEqual(ambient.for_light(devs), (devs, ""))

    def test_brightest_sensor_wins(self):
        fake_iio(self.tmp, [(0, 0.01), (7800, 0.01)])        # the reference laptop: two ALS nodes
        self.assertAlmostEqual(ambient.lux(), 78.0)
        devs = {"ir": "/dev/video2", "rgb": "/dev/video0"}
        self.assertEqual(ambient.for_light(devs)[0], devs)

    def test_dark_room_opens_ir_only(self):
        fake_iio(self.tmp, [(0, 0.01), (0, 0.01)])
        devs, note = ambient.for_light({"ir": "/dev/video2", "rgb": "/dev/video0"})
        self.assertEqual(devs, {"ir": "/dev/video2"})
        self.assertIn("IR only", note)

    def test_rgb_only_machine_keeps_rgb_in_the_dark(self):
        fake_iio(self.tmp, [(0, 0.01)])
        devs = {"rgb": "/dev/video0"}
        self.assertEqual(ambient.for_light(devs)[0], devs)

    def test_separate_cameras_keep_rgb_in_the_dark(self):
        fake_iio(self.tmp, [(0, 0.01)])
        devs = {"ir": "/dev/video2", "rgb": "/dev/video4"}
        with mock.patch.object(ambient, "same_module", return_value=False):   # e.g. an external webcam
            self.assertEqual(ambient.for_light(devs)[0], devs)

    def test_raw_count_without_scale_is_not_lux(self):
        d = self.tmp / "iio:device0"; d.mkdir()
        (d / "in_illuminance_raw").write_text("3\n")                      # no scale: unknown units
        self.assertIsNone(ambient.lux())

    def test_indexed_input_channel(self):
        d = self.tmp / "iio:device0"; d.mkdir()
        (d / "in_illuminance0_input").write_text("2.5\n")
        self.assertAlmostEqual(ambient.lux(), 2.5)

    def test_starved_frame(self):
        flat = np.full((360, 640), 24, np.uint8)
        self.assertTrue(ambient.starved(flat))
        face = np.random.default_rng(0).integers(0, 200, (360, 640), dtype=np.uint8)
        self.assertFalse(ambient.starved(face))


class FakeCap:
    """A cv2.VideoCapture stand-in delivering a fixed frame."""

    def __init__(self, img):
        self.img, self.released = img, False

    def read(self):
        time.sleep(0.005)
        return (not self.released), (None if self.released else self.img.copy())

    def set(self, *a):
        return True

    def get(self, *a):
        return 0.0

    def release(self):
        self.released = True


class FrameFormats(unittest.TestCase):
    def test_16bit_and_single_channel_ir_frames(self):
        y16 = (np.arange(360 * 640, dtype=np.uint16).reshape(360, 640) % 1024)
        fr = capture._to_frame("ir", y16)
        self.assertEqual(fr.gray.dtype, np.uint8); self.assertEqual(fr.rgb.shape, (360, 640, 3))
        self.assertGreater(int(fr.gray.max()), 200)
        fr = capture._to_frame("ir", np.zeros((360, 640, 1), np.uint8))
        self.assertEqual(fr.gray.shape, (360, 640))


class WatchdogTest(unittest.TestCase):
    def test_starved_ir_drops_rgb_and_reopens_ir(self):
        flat_ir = np.full((360, 640), 24, np.uint8)
        live_ir = np.random.default_rng(1).integers(0, 200, (360, 640), dtype=np.uint8)
        rgb = np.zeros((360, 640, 3), np.uint8)
        ir_cap, rgb_cap = FakeCap(flat_ir), FakeCap(rgb)
        reopened = FakeCap(live_ir)
        with mock.patch.object(capture, "_open_one", return_value=reopened) as op, \
                mock.patch.object(capture.brightness, "Exposure", mock.MagicMock()):
            cap = capture.Capture({"ir": "/dev/video2", "rgb": "/dev/video0"}, _opened={"ir": ir_cap, "rgb": rgb_cap})
            try:
                cap.warm_up(3)
                for _ in range(20):
                    if cap.spectra == ["ir"]:
                        break
                    cap.read_set()
                self.assertEqual(cap.spectra, ["ir"])
                op.assert_called_once_with("ir", "/dev/video2")
                self.assertTrue(rgb_cap.released and ir_cap.released)
                fr = cap.read_one("ir", flush=0)
                self.assertFalse(ambient.starved(fr.gray))
                self.assertTrue(any("IR only" in n for n in cap.notes))
                self.assertIsNone(cap.read_one("rgb"))
            finally:
                cap.close()

    def test_failed_reopen_is_reported_not_raised_from_read_one(self):
        flat_ir = np.full((360, 640), 24, np.uint8)
        with mock.patch.object(capture, "_open_one", side_effect=capture.CaptureError("busy")), \
                mock.patch.object(capture.brightness, "Exposure", mock.MagicMock()):
            cap = capture.Capture({"ir": "/dev/video2", "rgb": "/dev/video0"},
                                  _opened={"ir": FakeCap(flat_ir), "rgb": FakeCap(np.zeros((360, 640, 3), np.uint8))})
            try:
                cap.warm_up(3)
                for _ in range(10):
                    cap.read_one("ir", flush=0)                 # the enrolment window's path: never raises
                self.assertTrue(cap.lost)
                with self.assertRaises(capture.CaptureError):
                    cap.read_set()                              # the auth path ends the attempt cleanly
            finally:
                cap.close()

    def test_flat_ir_beside_a_bright_rgb_keeps_both(self):
        # an IR camera whose LED is off next to a working RGB camera: not starvation, keep RGB
        flat_ir = np.full((360, 640), 24, np.uint8)
        bright = np.full((360, 640, 3), 120, np.uint8)
        with mock.patch.object(capture, "_open_one") as op, \
                mock.patch.object(capture.brightness, "Exposure", mock.MagicMock()):
            cap = capture.Capture({"ir": "/dev/video2", "rgb": "/dev/video0"}, _opened={"ir": FakeCap(flat_ir), "rgb": FakeCap(bright)})
            try:
                cap.warm_up(3)
                for _ in range(20):
                    cap.read_set()
                op.assert_not_called()
                self.assertEqual(sorted(cap.spectra), ["ir", "rgb"])
            finally:
                cap.close()

    def test_ir_only_is_never_touched(self):
        flat_ir = np.full((360, 640), 24, np.uint8)                 # e.g. a covered lens, IR alone
        with mock.patch.object(capture, "_open_one") as op, \
                mock.patch.object(capture.brightness, "Exposure", mock.MagicMock()):
            cap = capture.Capture({"ir": "/dev/video2"}, _opened={"ir": FakeCap(flat_ir)})
            try:
                cap.warm_up(3)
                for _ in range(10):
                    cap.read_one("ir", flush=0)
                op.assert_not_called()
                self.assertEqual(cap.spectra, ["ir"])
            finally:
                cap.close()


if __name__ == "__main__":
    unittest.main()
