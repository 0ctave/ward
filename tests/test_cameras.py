"""Stable camera identification: /dev numbers change, USB ids don't."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faceup import cameras  # noqa: E402


def cam(path, fmts, iface, vid="30c9", pid="0057"):
    return cameras.Camera(path, "Integrated Camera", {f: [(640, 360)] for f in fmts}, vid, pid, "01.00.00", iface, "5-1")


class ResolveTests(unittest.TestCase):
    before = [cam("/dev/video0", ["MJPG", "YUYV"], "00"), cam("/dev/video2", ["GREY"], "02")]
    after = [cam("/dev/video1", ["MJPG", "YUYV"], "00"), cam("/dev/video4", ["GREY"], "02")]

    def test_identity_and_role(self):
        self.assertEqual(self.before[1].identity, "usb:30c9:0057:02")
        self.assertEqual(self.before[1].role, "ir")
        self.assertEqual(self.before[0].role, "rgb")

    def test_identity_survives_renumbering(self):
        self.assertEqual(cameras.resolve("usb:30c9:0057:02", "ir", self.before)[0], "/dev/video2")
        self.assertEqual(cameras.resolve("usb:30c9:0057:02", "ir", self.after)[0], "/dev/video4")
        self.assertEqual(cameras.resolve("usb:30c9:0057:00", "rgb", self.after)[0], "/dev/video1")

    def test_legacy_path_self_heals_with_a_note(self):
        path, note = cameras.resolve("/dev/video2", "ir", self.after)
        self.assertEqual(path, "/dev/video4")
        self.assertIn("moved", note)
        path, note = cameras.resolve("/dev/video2", "ir", self.before)
        self.assertEqual((path, note), ("/dev/video2", ""))

    def test_role_mismatch_and_missing(self):
        self.assertEqual(cameras.resolve("usb:30c9:0057:00", "ir", self.after)[0], "")
        self.assertEqual(cameras.resolve("/dev/video2", "ir", [self.after[0]])[0], "")
        self.assertEqual(cameras.resolve("auto", "ir", self.after)[0], "/dev/video4")
        self.assertEqual(cameras.resolve("", "ir", self.after), ("", ""))


class GeneralHardwareTests(unittest.TestCase):
    def test_identical_webcams_by_serial_then_port(self):
        a = cameras.Camera("/dev/video0", "Webcam C920", {"MJPG": [(1920, 1080)]}, "046d", "082d", "AB12CD34", "00", "1-2")
        b = cameras.Camera("/dev/video2", "Webcam C920", {"MJPG": [(1920, 1080)]}, "046d", "082d", "EF56GH78", "00", "1-3")
        self.assertNotEqual(a.identity, b.identity)                       # real serials → distinct ids
        self.assertEqual(cameras.resolve(b.identity, "rgb", [a, b])[0], "/dev/video2")
        # generic serials → same id; resolution warns and @port pins one
        c = cameras.Camera("/dev/video0", "Cam", {"MJPG": [(1280, 720)]}, "1234", "5678", "01.00.00", "00", "1-2")
        d = cameras.Camera("/dev/video2", "Cam", {"MJPG": [(1280, 720)]}, "1234", "5678", "01.00.00", "00", "1-3")
        self.assertEqual(c.identity, d.identity)
        path, note = cameras.resolve(c.identity, "rgb", [c, d])
        self.assertIn("identical", note)
        self.assertEqual(cameras.resolve(c.identity + "@1-3", "rgb", [c, d]), ("/dev/video2", ""))
        self.assertEqual(cameras.spec_for(d, [c, d]), d.identity + "@1-3")
        self.assertEqual(cameras.spec_for(a, [a, b]), a.identity)

    def test_non_usb_identity(self):
        p = cameras.Camera("/dev/video0", "ipu6 capture", {"NV12": [(1280, 720)]}, "8086", "a75d", "", "", "0000:00:05.0", "pci", "intel-ipu6")
        self.assertEqual(p.identity, "pci:0000:00:05.0:ipu6 capture")
        self.assertEqual(cameras.resolve(p.identity, "rgb", [p])[0], "/dev/video0")

    def test_ir_detection_variants(self):
        self.assertTrue(cameras.Camera("/dev/video2", "Cam", {"Y8  ": [(640, 480)]}).is_ir)
        self.assertTrue(cameras.Camera("/dev/video2", "Cam", {"GREY": [(640, 360)], "YUYV": [(640, 360)]}).is_ir)   # grey + no MJPG
        self.assertTrue(cameras.Camera("/dev/video2", "Integrated IR Camera", {"YUYV": [(640, 360)]}).is_ir)   # name hint
        self.assertFalse(cameras.Camera("/dev/video0", "Integrated Camera", {"YUYV": [(640, 360)], "MJPG": [(1280, 720)]}).is_ir)
        # force_roles: trust the config even when the stream looks like colour
        yuyv_ir = cameras.Camera("/dev/video2", "Camera", {"YUYV": [(640, 360)]}, "1bcf", "28c4", "", "02", "3-1")
        self.assertEqual(cameras.resolve(yuyv_ir.identity, "ir", [yuyv_ir])[0], "")
        self.assertEqual(cameras.resolve(yuyv_ir.identity, "ir", [yuyv_ir], force=True)[0], "/dev/video2")

    def test_pick_pair_prefers_same_module(self):
        builtin_rgb = cam("/dev/video0", ["MJPG"], "00")
        builtin_ir = cam("/dev/video2", ["GREY"], "02")
        external = cameras.Camera("/dev/video4", "Webcam", {"MJPG": [(3840, 2160)]}, "046d", "085e", "SER", "00", "1-4")
        ir, rgb = cameras.pick_pair([external, builtin_rgb, builtin_ir])
        self.assertEqual((ir.path, rgb.path), ("/dev/video2", "/dev/video0"))     # not the 4K external one
        ir, rgb = cameras.pick_pair([external, builtin_rgb])
        self.assertEqual((ir, rgb.path), (None, "/dev/video4"))                    # no IR → best colour camera


if __name__ == "__main__":
    unittest.main()


class ModuleLayoutTests(unittest.TestCase):
    def test_terminal_ui_and_app_ui_are_separate_modules(self):
        from faceup import ui
        for name in ("ok", "warn", "fail", "banner", "section", "confirm", "ask"):
            self.assertTrue(hasattr(ui, name), f"faceup.ui lost {name}")
        try:
            from faceup import appui
        except (ImportError, ValueError) as e:       # no GTK 4/libadwaita/pycairo (minimal containers)
            self.skipTest(f"GTK stack incomplete: {e}")
        self.assertTrue(hasattr(appui, "EnrollWindow") and hasattr(appui, "EmitterWindow"))
