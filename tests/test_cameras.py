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


if __name__ == "__main__":
    unittest.main()
