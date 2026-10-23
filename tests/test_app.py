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


class Privileges(unittest.TestCase):
    def test_actions_go_through_pkexec_unless_root(self):
        self.assertEqual(app.priv_cmd(["tune"], root=True)[-2:], ["faceup", "tune"])
        self.assertNotIn("pkexec", app.priv_cmd(["tune"], root=True))
        from unittest import mock
        with mock.patch.dict(os.environ, {"FACEUP_DEV_PKEXEC": "1"}):
            dev = app.priv_cmd(["models", "--json"], root=False, pkg_parent="/home/x/faceup")
        self.assertEqual(dev[0], "pkexec"); self.assertIn("PYTHONPATH=/home/x/faceup", dev); self.assertEqual(dev[-2:], ["models", "--json"])

    def test_checkout_is_never_run_as_root_without_opt_in(self):
        # priv-1: a user-writable checkout imported by root = root for anything that edits it
        from unittest import mock
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FACEUP_DEV_PKEXEC", None)
            with mock.patch("os.path.exists", return_value=True):
                self.assertEqual(app.priv_cmd(["tune"], root=False, pkg_parent="/home/x/faceup"), ["pkexec", app.INSTALLED, "tune"])
            with mock.patch("os.path.exists", return_value=False):
                cmd = app.priv_cmd(["tune"], root=False, pkg_parent="/home/x/faceup")
        self.assertNotIn("pkexec", cmd); self.assertFalse(any("PYTHONPATH" in c for c in cmd))

    def test_invoking_user_under_pkexec(self):
        from unittest import mock
        from faceup import manage
        with mock.patch.dict(os.environ, {"PKEXEC_UID": str(os.getuid()), "USER": "root"}, clear=False):
            os.environ.pop("SUDO_USER", None)
            import pwd
            self.assertEqual(manage.invoking_user(), pwd.getpwuid(os.getuid()).pw_name)
        with mock.patch.dict(os.environ, {"SUDO_USER": "alice"}):
            self.assertEqual(manage.invoking_user(), "alice")


class OneWindow(unittest.TestCase):
    """Registration and calibration draw inside the manager: snapshots cross a pipe as JSON lines."""

    def test_snapshot_roundtrip(self):
        import numpy as np
        from faceup import display
        fr = np.zeros((36, 64, 3), np.uint8); fr[:, :32] = (200, 40, 40)
        line = display.encode_snapshot({"phase": "run", "frame": fr, "box": (1, 2, 3, 4), "progress": np.float32(0.5), "target": (-20.0, 0.0)})
        self.assertNotIn("\n", line)
        s = display.decode_snapshot(line)
        self.assertEqual(s["phase"], "run"); self.assertEqual(s["box"], [1, 2, 3, 4]); self.assertAlmostEqual(s["progress"], 0.5)
        self.assertEqual(s["frame"].shape, (36, 64, 3)); self.assertGreater(int(s["frame"][10, 5, 0]), 150)   # colour order kept
        self.assertIsNone(display.decode_snapshot("not json"))

    def test_camera_commands_are_streamed(self):
        self.assertIn("enroll", app.CAMERA_CMDS); self.assertIn("calibrate", app.CAMERA_CMDS)
        from faceup import cli
        for cmd in (["enroll", "--stream-ui"], ["calibrate", "--stream-ui"]):
            with self.assertRaises(SystemExit) as cm:
                cli.main(cmd + ["--help"])
            self.assertEqual(cm.exception.code, 0)


class ModelsJson(unittest.TestCase):
    def test_models_json_for_the_app(self):
        import io, json
        from contextlib import redirect_stdout
        from faceup import cli
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli.main(["models", "--json", "-U", "nobody-json"])
        self.assertEqual(rc, 0)
        data = json.loads(buf.getvalue().strip().splitlines()[-1])
        self.assertEqual(data["user"], "nobody-json"); self.assertEqual(data["people"], [])


class Manage(unittest.TestCase):
    def _row(self, ok, reason="accept", who="", dark=False, ms=900, service=""):
        return {"accepted": ok, "reason": reason, "result": reason, "who": who, "dark": dark, "ms": ms, "service": service, "time": "2026-10-23T10:00:00"}

    def test_names(self):
        from faceup import manage
        self.assertEqual(manage.valid_name("Alice"), "")
        for bad in ("", "-rf", "a" * 40, "x/y", "--identity"):
            self.assertTrue(manage.valid_name(bad), bad)
        self.assertIn("owner", manage.valid_name("octave", owner="octave"))

    def test_people_from_store_and_log(self):
        import numpy as np
        from faceup import manage
        from faceup.engine import store
        st = store.Store(user="u")
        e = np.ones(128) / np.sqrt(128)
        st.add("C", "ir", e, 1.0); st.add("C", "rgb", e, 1.0)
        st.add("C", "ir", e, 1.0, identity="Alice"); st.add("L", "ir", e, 1.0, condition="glasses")
        rows = [self._row(True), self._row(True, who="Alice"), self._row(False, "no-match")]
        ppl = manage.people(st, "u", rows)
        self.assertEqual([p["name"] for p in ppl], ["u", "Alice"])
        self.assertTrue(ppl[0]["owner"]); self.assertEqual(ppl[0]["models"], {"ir": 2, "rgb": 1})
        self.assertEqual(ppl[0]["conditions"], ["glasses"]); self.assertEqual(ppl[1]["accepts"], 1)

    def test_reliability_verdicts_and_advice(self):
        from faceup import manage
        self.assertEqual(manage.reliability([])["verdict"], "unknown")
        good = manage.reliability([self._row(True)] * 20)
        self.assertEqual(good["verdict"], "good"); self.assertEqual(good["overall"]["rate"], 1.0)
        bad = manage.reliability([self._row(True)] * 5 + [self._row(False, "no-match")] * 5 + [self._row(False, "liveness")] * 4)
        self.assertEqual(bad["verdict"], "poor")
        argvs = [a for _, a in bad["advice"] if a]
        self.assertIn(["enroll", "--fresh", "--selfcheck"], argvs); self.assertIn(["emitter", "toggle-bench"], argvs)
        night = manage.reliability([self._row(True)] * 10 + [self._row(False, "no-match", dark=True)] * 4)
        self.assertEqual(night["night"]["attempts"], 4); self.assertIn(["emitter", "probe"], [a for _, a in night["advice"]])
        self.assertEqual(manage.reliability([self._row(False, "no-match")] * 80 + [self._row(True)] * 50)["overall"]["rate"], 1.0)   # last 50 only


if __name__ == "__main__":
    unittest.main()
