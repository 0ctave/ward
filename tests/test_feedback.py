"""Session feedback: channel selection, messages, notification command (no PAM, no bus needed)."""
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ward import feedback
from ward.engine.auth import Outcome
from ward.engine.liveness import ABSTAIN, PASS


class _Tty(io.StringIO):
    def isatty(self):
        return True


def _accept():
    return Outcome(True, "accept", "octave", 0.21, 0.36, votes=[PASS("cross", "iou 0.8"), ABSTAIN("motion", "still (0.4px)")], elapsed_ms=930)


class Messages(unittest.TestCase):
    def test_result_text(self):
        self.assertEqual(feedback.result_text("accept"), "Face recognised")
        self.assertIn("password", feedback.result_text("no-match"))
        self.assertIn("Liveness", feedback.result_text("spoof:cross"))
        self.assertIn("No face", feedback.result_text("no-face:ir"))
        for silent in ("ssh", "no-models", "store"):
            self.assertIsNone(feedback.result_text(silent))

    def test_service_names(self):
        self.assertEqual(feedback.service_name("sudo"), "sudo")
        self.assertEqual(feedback.service_name("gdm-password"), "the lock screen")
        self.assertEqual(feedback.service_name("kde"), "the lock screen")
        self.assertEqual(feedback.service_name("polkit-1"), "an admin prompt")


class Channels(unittest.TestCase):
    def test_pam_channel_prints_lines_to_stdout(self):
        out, err = io.StringIO(), io.StringIO()
        fb = feedback.Feedback("octave", {"notify": "off"}, err=err, out=out, env={"PAM_TYPE": "auth", "PAM_SERVICE": "gdm-password"})
        fb.start()
        fb.event("face")
        fb.event("hold", still=False)
        fb.event("hold", still=True)
        fb.event("hold", still=True)             # repeated: said once
        fb.finish(Outcome(False, "no-match", "octave", 0.51, elapsed_ms=2500))
        lines = out.getvalue().splitlines()
        self.assertEqual(lines, [feedback.LOOKING, feedback.HOLD, feedback.HOLD_STILL, feedback.result_text("no-match")])
        self.assertEqual(err.getvalue(), "")

    def test_silent_reasons_say_nothing_after_looking(self):
        out = io.StringIO()
        fb = feedback.Feedback("octave", {"notify": "off"}, out=out, err=io.StringIO(), env={"PAM_TYPE": "auth"})
        fb.start(); fb.finish(Outcome(False, "no-models", "octave"))
        self.assertEqual(out.getvalue().splitlines(), [feedback.LOOKING])

    def test_terminal_channel_animates_then_prints_result(self):
        out, err = io.StringIO(), _Tty()
        fb = feedback.Feedback("octave", {"notify": "off"}, err=err, out=out, env={"PAM_TYPE": "auth", "NO_COLOR": "1"})
        self.assertTrue(fb.tty); self.assertFalse(fb.pam)
        fb.start()
        import time; time.sleep(0.25)
        fb.finish(_accept())
        text = err.getvalue()
        self.assertIn(feedback.LOOKING, text)                     # the spinner drew at least once
        self.assertTrue(text.endswith("✔ Face recognised · IR 0.21 · RGB 0.36 · cross ✓ · motion – · 0.9 s\n"), text)
        self.assertEqual(out.getvalue(), "")                       # never doubles up on stdout

    def test_sudo_spinner_goes_to_pam_tty(self):
        master, slave = os.openpty()
        path = os.ttyname(slave)
        out = io.StringIO()
        fb = feedback.Feedback("octave", {"notify": "off"}, err=io.StringIO(), out=out,
                               env={"PAM_TYPE": "auth", "PAM_SERVICE": "sudo", "PAM_TTY": path, "NO_COLOR": "1"})
        self.assertTrue(fb.tty); self.assertFalse(fb.pam)
        fb.start(); import time; time.sleep(0.25); fb.finish(_accept())
        import select
        data = ""
        deadline = time.monotonic() + 1.0          # the pty line discipline delivers asynchronously
        while "Face recognised" not in data and time.monotonic() < deadline:
            if select.select([master], [], [], 0.05)[0]:
                data += os.read(master, 65536).decode(errors="replace")
        os.close(master); os.close(slave)
        self.assertIn(feedback.LOOKING, data); self.assertIn("✔ Face recognised", data)
        self.assertEqual(out.getvalue(), "")
        # a VT or a display name never gets the spinner; the PAM channel stays
        fb = feedback.Feedback("octave", {}, err=io.StringIO(), out=io.StringIO(),
                               env={"PAM_TYPE": "auth", "PAM_SERVICE": "gdm-password", "PAM_TTY": "/dev/tty1"})
        self.assertFalse(fb.tty); self.assertTrue(fb.pam)
        fb = feedback.Feedback("octave", {}, err=io.StringIO(), out=io.StringIO(),
                               env={"PAM_TYPE": "auth", "PAM_SERVICE": "sudo", "PAM_TTY": "/dev/tty1"})
        self.assertFalse(fb.tty); self.assertTrue(fb.pam)

    def test_verbose_disables_the_spinner(self):
        fb = feedback.Feedback("octave", {}, verbose=True, err=_Tty(), out=io.StringIO(), env={})
        self.assertFalse(fb.tty); self.assertFalse(fb.pam)


class Notification(unittest.TestCase):
    def test_command_and_modes(self):
        if feedback.shutil.which("notify-send") is None:
            self.skipTest("notify-send not installed")
        cmd = feedback.notify_command(_accept(), "sudo")
        self.assertEqual(cmd[-2], "Unlocked with your face")
        self.assertIn("sudo · IR 0.21", cmd[-1])
        self.assertIn("-a", cmd); self.assertIn("ward", cmd)
        self.assertIn("boolean:transient:true", cmd)                # GNOME keeps non-transient ones in the tray forever
        self.assertIn("-p", cmd)
        sh = feedback.replacing_shell(cmd, "/run/user/1000")
        self.assertEqual(sh[:2], ["/bin/sh", "-c"])
        self.assertIn("/run/user/1000/ward-notify.id", sh[2]); self.assertIn('-r "$r"', sh[2])
        cmd = feedback.notify_command(Outcome(False, "spoof:cross", "octave"), "polkit-1")
        self.assertIn("Liveness", cmd[-1])
        spawned = []
        orig = feedback.spawn_as_user
        feedback.spawn_as_user = lambda user, c: spawned.append((user, c[-2]))
        try:
            for mode, outcome, expect in (("accept", _accept(), 1), ("accept", Outcome(False, "no-match", "o"), 0),
                                          ("all", Outcome(False, "no-match", "o"), 1), ("off", _accept(), 0),
                                          ("all", Outcome(False, "ssh", "o"), 0), ("bogus", Outcome(False, "no-match", "o"), 0)):
                spawned.clear()
                feedback.Feedback("octave", {"notify": mode}, err=io.StringIO(), out=io.StringIO(), env={}).notify(outcome)
                self.assertEqual(len(spawned), expect, (mode, outcome.reason))
        finally:
            feedback.spawn_as_user = orig

    def test_session_login_detection(self):
        me = os.environ.get("USER", "root")
        self.assertFalse(feedback.is_session_login(me, {"PAM_SERVICE": "sudo"}))            # never for sudo
        self.assertFalse(feedback.is_session_login("no-such-user-ward", {"PAM_SERVICE": "gdm-password"}))
        has_bus = os.path.exists(f"/run/user/{os.getuid()}/bus")
        # a greeter service with a running session (bus present) is an unlock, not a login
        self.assertEqual(feedback.is_session_login(me, {"PAM_SERVICE": "gdm-password"}), not has_bus)

    def test_session_env_needs_a_bus(self):
        self.assertIsNone(feedback.session_env("no-such-user-ward"))
        se = feedback.session_env(os.environ.get("USER", "root"))
        if se is not None:
            env, uid, gid = se
            self.assertIn("DBUS_SESSION_BUS_ADDRESS", env)
            self.assertEqual(uid, os.getuid())


if __name__ == "__main__":
    unittest.main()
