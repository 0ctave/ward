"""The app's data layer (health checks) and its non-GTK helpers. The window itself needs a
display; what it renders comes from `health.collect()`, which is what is tested here."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("FACEUP_ETC", tempfile.mkdtemp(prefix="faceup-test-"))
os.environ.setdefault("FACEUP_LOG", os.path.join(os.environ["FACEUP_ETC"], "log"))

from faceup import app, health


class Helpers(unittest.TestCase):
    def test_strip_ansi_and_command(self):
        self.assertEqual(app.strip_ansi("\x1b[32m✔\x1b[0m ok \x1b[1mbold\x1b[0m"), "✔ ok bold")
        cmd = app.faceup_cmd(["enroll", "--identity", "Alice"])
        self.assertEqual(cmd[-4:], ["faceup", "enroll", "--identity", "Alice"]); self.assertEqual(cmd[1], "-m")
        env = app.child_env()
        self.assertIn("PYTHONPATH", env); self.assertEqual(env.get("NO_COLOR"), "1")
        self.assertIn("glasses", app.CONDITIONS)


class Health(unittest.TestCase):
    def test_collect_is_structured_and_never_raises(self):
        rep = health.collect(user="nobody-here", probe_cameras=False)
        self.assertIn(rep["profile"], ("A", "B", "C", "-"))
        keys = [s.key for s in rep["sections"]]
        for k in ("cameras", "engine", "faces", "access", "attempts", "compat"):
            self.assertIn(k, keys)
        for s in rep["sections"]:
            self.assertTrue(s.title)
            for it in s.items:
                self.assertIn(it.level, ("ok", "warn", "fail", "info")); self.assertTrue(it.text)
                if it.fix is not None:
                    self.assertIsInstance(it.fix, list); self.assertTrue(all(isinstance(x, str) for x in it.fix))
        self.assertIsInstance(health.problems(rep), int)
        faces = next(s for s in rep["sections"] if s.key == "faces")
        self.assertTrue(any(it.fix == ["enroll"] or "root" in it.text for it in faces.items))   # no store → enrol, or root needed

    def test_rgb_only_profile_advice(self):
        from faceup.engine import config
        cfg = config.load(); cfg["cameras"]["ir"] = ""; cfg["cameras"]["rgb"] = "usb:1:1:0"
        rep = health.collect(user="nobody-here", cfg=cfg)
        self.assertEqual(rep["profile"], "C")
        eng = next(s for s in rep["sections"] if s.key == "engine")
        self.assertTrue(any("still face" in it.text and it.fix and it.fix[0] == "tune" for it in eng.items))
        cfg["liveness"]["require_liveness_when_available"] = False
        rep = health.collect(user="nobody-here", cfg=cfg)
        eng = next(s for s in rep["sections"] if s.key == "engine")
        self.assertTrue(any("recognition alone" in it.text for it in eng.items))


if __name__ == "__main__":
    unittest.main()
