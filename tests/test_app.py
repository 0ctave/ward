"""The app's data layer (health checks) and its non-GTK helpers. The window itself needs a
display; what it renders comes from `health.collect()`, which is what is tested here."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("WARD_ETC", tempfile.mkdtemp(prefix="ward-test-"))
os.environ.setdefault("WARD_LOG", os.path.join(os.environ["WARD_ETC"], "log"))

from ward import app, health


class Helpers(unittest.TestCase):
    def test_strip_ansi_and_command(self):
        self.assertEqual(app.strip_ansi("\x1b[32m✔\x1b[0m ok \x1b[1mbold\x1b[0m"), "✔ ok bold")
        cmd = app.ward_cmd(["enroll", "--identity", "Alice"])
        self.assertEqual(cmd[-4:], ["ward", "enroll", "--identity", "Alice"]); self.assertEqual(cmd[1], "-m")
        env = app.child_env()
        self.assertIn("PYTHONPATH", env); self.assertEqual(env.get("NO_COLOR"), "1")
        self.assertIn("glasses", app.CONDITIONS)


class DoctorAuthUser(unittest.TestCase):
    """doctor pre-flights the user resolution: PAM on + the login user unenrolled → a clear warning
    (the 'sudo authenticated root → no-models' class of bug, caught before the password prompt)."""

    def test_warns_when_pam_on_but_human_unenrolled(self):
        from unittest import mock
        from ward import health
        from ward.engine import pam as pammod
        with mock.patch.object(pammod, "status", lambda: {"enabled": True, "native": True, "current": True, "howdy": False}), \
             mock.patch("ward.client._human_user", lambda: "nobody-xyz"):
            rep = health.collect(user="nobody-xyz")
        access = next(s for s in rep["sections"] if s.key == "access")
        self.assertTrue(any("no enrolled face" in i.text and i.fix == ["enroll", "--fresh"] for i in access.items))


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
        from ward.engine import config
        cfg = config.load(); cfg["cameras"]["ir"] = ""; cfg["cameras"]["rgb"] = "usb:1:1:0"
        rep = health.collect(user="nobody-here", cfg=cfg)
        self.assertEqual(rep["profile"], "C")
        eng = next(s for s in rep["sections"] if s.key == "engine")
        self.assertTrue(any(it.level == "ok" and "anti-spoof CNN" in it.text for it in eng.items))   # bundled voter
        cfg["liveness"]["antispoof_cnn"] = False
        rep = health.collect(user="nobody-here", cfg=cfg)
        eng = next(s for s in rep["sections"] if s.key == "engine")
        self.assertTrue(any(it.fix == ["antispoof", "on"] for it in eng.items))
        self.assertTrue(any("still face" in it.text and it.fix and it.fix[0] == "tune" for it in eng.items))
        cfg["liveness"]["require_liveness_when_available"] = False
        rep = health.collect(user="nobody-here", cfg=cfg)
        eng = next(s for s in rep["sections"] if s.key == "engine")
        self.assertTrue(any("recognition alone" in it.text for it in eng.items))


class Privileges(unittest.TestCase):
    def test_actions_go_through_pkexec_unless_root(self):
        self.assertEqual(app.priv_cmd(["tune"], root=True)[-2:], ["ward", "tune"])
        self.assertNotIn("pkexec", app.priv_cmd(["tune"], root=True))
        from unittest import mock
        with mock.patch.dict(os.environ, {"WARD_DEV_PKEXEC": "1"}):
            dev = app.priv_cmd(["models", "--json"], root=False, pkg_parent="/home/x/ward")
        self.assertEqual(dev[0], "pkexec"); self.assertIn("PYTHONPATH=/home/x/ward", dev); self.assertEqual(dev[-2:], ["models", "--json"])

    def test_checkout_is_never_run_as_root_without_opt_in(self):
        # priv-1: a user-writable checkout imported by root = root for anything that edits it
        from unittest import mock
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WARD_DEV_PKEXEC", None)
            with mock.patch("os.path.exists", return_value=True):
                self.assertEqual(app.priv_cmd(["tune"], root=False, pkg_parent="/home/x/ward"), ["pkexec", app.INSTALLED, "tune"])
            with mock.patch("os.path.exists", return_value=False):
                cmd = app.priv_cmd(["tune"], root=False, pkg_parent="/home/x/ward")
        self.assertNotIn("pkexec", cmd); self.assertFalse(any("PYTHONPATH" in c for c in cmd))

    def test_invoking_user_under_pkexec(self):
        from unittest import mock
        from ward import appdata as manage
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
        from ward import display
        fr = np.zeros((36, 64, 3), np.uint8); fr[:, :32] = (200, 40, 40)
        line = display.encode_snapshot({"phase": "run", "frame": fr, "box": (1, 2, 3, 4), "progress": np.float32(0.5), "target": (-20.0, 0.0)})
        self.assertNotIn("\n", line)
        s = display.decode_snapshot(line)
        self.assertEqual(s["phase"], "run"); self.assertEqual(s["box"], [1, 2, 3, 4]); self.assertAlmostEqual(s["progress"], 0.5)
        self.assertEqual(s["frame"].shape, (36, 64, 3)); self.assertGreater(int(s["frame"][10, 5, 0]), 150)   # colour order kept
        self.assertIsNone(display.decode_snapshot("not json"))

    def test_camera_commands_are_streamed(self):
        self.assertIn("enroll", app.CAMERA_CMDS); self.assertIn("calibrate", app.CAMERA_CMDS)
        from ward import cli
        for cmd in (["enroll", "--stream-ui"], ["calibrate", "--stream-ui"]):
            with self.assertRaises(SystemExit) as cm:
                cli.main(cmd + ["--help"])
            self.assertEqual(cm.exception.code, 0)


class ModelsJson(unittest.TestCase):
    def test_models_json_for_the_app(self):
        import io, json
        from contextlib import redirect_stdout
        from ward import cli
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
        from ward import appdata as manage
        self.assertEqual(manage.valid_name("Alice"), "")
        for bad in ("", "-rf", "a" * 40, "x/y", "--identity"):
            self.assertTrue(manage.valid_name(bad), bad)
        self.assertIn("owner", manage.valid_name("octave", owner="octave"))

    def test_people_from_store_and_log(self):
        import numpy as np
        from ward import appdata as manage
        from ward.engine import store
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
        from ward import appdata as manage
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


class SetupWizard(unittest.TestCase):
    """The in-app setup wizard's step list (GTK-free logic in app.setup_steps)."""

    def test_order_and_essentials(self):
        from ward.app import ENROL, setup_steps
        steps = setup_steps({"profile": "A"})
        keys = [s["key"] for s in steps]
        self.assertEqual(keys[0], "cameras"); self.assertEqual(keys[-1], "pam")
        self.assertEqual(keys, ["cameras", "emitter", "antispoof", "register", "keyring", "pam"])
        reg = next(s for s in steps if s["key"] == "register")
        self.assertEqual(reg["camera"], ENROL)                 # register draws in-app
        self.assertTrue(next(s for s in steps if s["key"] == "keyring")["keyring"])
        self.assertTrue(next(s for s in steps if s["key"] == "emitter").get("then_argv"))

    def test_emitter_only_with_ir(self):
        from ward.app import setup_steps
        self.assertNotIn("emitter", [s["key"] for s in setup_steps({"profile": "C"})])
        self.assertIn("emitter", [s["key"] for s in setup_steps({"profile": "B"})])

    def test_antispoof_only_with_colour_camera(self):
        """The anti-spoof CNN is colour-only — the step is pointless without an RGB camera (B and none)."""
        from ward.app import setup_steps
        self.assertIn("antispoof", [s["key"] for s in setup_steps({"profile": "A"})])
        self.assertIn("antispoof", [s["key"] for s in setup_steps({"profile": "C"})])
        self.assertNotIn("antispoof", [s["key"] for s in setup_steps({"profile": "B"})])
        self.assertNotIn("antispoof", [s["key"] for s in setup_steps({"profile": "-"})])

    def test_antispoof_reflects_current_state(self):
        """Already-on → a satisfied 'Done' step, not an actionable 'Turn on'; off → the real action."""
        from ward.app import setup_steps
        on = next(s for s in setup_steps({"profile": "A", "cfg": {"liveness": {"antispoof_cnn": True}}}) if s["key"] == "antispoof")
        self.assertTrue(on.get("satisfied")); self.assertNotIn("argv", on); self.assertFalse(on.get("optional"))
        off = next(s for s in setup_steps({"profile": "A", "cfg": {"liveness": {"antispoof_cnn": False}}}) if s["key"] == "antispoof")
        self.assertFalse(off.get("satisfied")); self.assertEqual(off["argv"], ["antispoof", "on"]); self.assertTrue(off.get("optional"))
        default = next(s for s in setup_steps({"profile": "A"}) if s["key"] == "antispoof")   # no cfg → defaults on
        self.assertTrue(default.get("satisfied"))

    def test_optional_and_labels(self):
        from ward.app import setup_steps
        report = {"profile": "A", "cfg": {"liveness": {"antispoof_cnn": False}}}
        for s in setup_steps(report):
            self.assertTrue(s["title"] and s["desc"] and s["run_label"])
        optional = {s["key"] for s in setup_steps(report) if s.get("optional")}
        self.assertEqual(optional, {"emitter", "antispoof", "keyring"})   # the essentials aren't skippable


if __name__ == "__main__":
    unittest.main()


class ModuleLayout(unittest.TestCase):
    """The split of cli.py into a package and of app.main's closure into app_window keeps the surface."""

    def test_cli_package_reexports(self):
        from ward import cli
        for name in ("main", "daemon_state", "cmd_doctor", "cmd_enroll", "cmd_auth", "cmd_setup", "cmd_emitter",
                     "need_root", "real_user", "POSES", "UNIT", "format_attempt", "run_calibration_session"):
            self.assertTrue(hasattr(cli, name), name)

    def test_app_imports_without_gtk(self):
        import subprocess
        code = "import sys; sys.modules['gi'] = None; import ward.app as a; assert callable(a.main); assert 'ward.app_window' not in sys.modules"
        r = subprocess.run([sys.executable, "-c", code], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
