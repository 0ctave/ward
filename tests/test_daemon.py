"""The resident engine: the socket protocol, feedback relay and the fall-back to the in-process engine."""
import os
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("WARD_ETC", tempfile.mkdtemp(prefix="ward-test-"))

from ward import client, daemon  # noqa: E402


class FakeResident:
    def __init__(self):
        self.seen = []

    def attempt(self, req, send, **kw):
        self.seen.append(req)
        send({"event": "face"}); send({"event": "hold", "still": True})
        send({"outcome": {"accepted": True, "reason": "accept", "user": req["user"], "ir_dist": 0.21, "rgb_dist": None,
                          "elapsed_ms": 640, "frames": 2, "detail": "", "phases": {}, "logline": "x",
                          "votes": [{"voter": "cross", "result": "PASS"}]}})


class Protocol(unittest.TestCase):
    def serve(self, res):
        import socket
        d = tempfile.mkdtemp(dir=os.environ["WARD_ETC"]); path = os.path.join(d, "s")
        srv = socket.socket(socket.AF_UNIX); srv.bind(path); srv.listen(1)

        def loop():
            while True:
                conn, _ = srv.accept(); daemon.handle(res, conn)
        threading.Thread(target=loop, daemon=True).start()
        return path

    def test_attempt_relays_events_and_outcome(self):
        res = FakeResident(); path = self.serve(res)
        events = []
        fb = SimpleNamespace(event=lambda kind, **kw: events.append((kind, kw)))
        with mock.patch.dict(os.environ, {"PAM_SERVICE": "sudo"}):
            out = client.via_daemon("octave", path=path, fb=fb)
        self.assertTrue(out.accepted); self.assertEqual(out.ir_dist, 0.21)
        self.assertEqual(out.votes[0].result.value, "PASS")
        self.assertEqual(events, [("face", {}), ("hold", {"still": True})])
        self.assertEqual(res.seen[0]["env"]["PAM_SERVICE"], "sudo")

    def test_no_daemon_means_none(self):
        self.assertIsNone(client.via_daemon("octave", path="/nonexistent/sock"))

    def test_outcome_dict_matches_feedback_needs(self):
        from ward import feedback
        from ward.engine.auth import Outcome
        from ward.engine.liveness import Result, Vote
        o = Outcome(True, "accept", "u", ir_dist=0.2, votes=[Vote("cross", Result.PASS)], elapsed_ms=900)
        d = daemon.outcome_dict(o); d["votes"] = [client._vote(v) for v in d["votes"]]
        self.assertIn("IR 0.20", feedback._detail(SimpleNamespace(**d)))

    def test_client_falls_back_to_the_engine(self):
        with mock.patch.object(client, "connect", lambda path=None: None), \
             mock.patch.object(client, "_human_user", lambda: "octave"), \
             mock.patch("ward.cli.main", return_value=7) as cli_main, mock.patch.object(client.os, "geteuid", lambda: 0):
            self.assertEqual(client.main(["-v"]), 7)
        cli_main.assert_called_once_with(["auth", "-U", "octave", "-v"])   # the resolved human is threaded through
        with mock.patch("ward.cli.main", return_value=3) as cli_main:
            self.assertEqual(client.main(["--replay", "/x"]), 3)          # not plain → full CLI, no -U injected
        cli_main.assert_called_once_with(["auth", "--replay", "/x"])


def bare_resident(cfg=None):
    """A Resident without model loading; Engine is patched by the tests."""
    r = daemon.Resident.__new__(daemon.Resident)
    r.cfgmod = SimpleNamespace(load=lambda: cfg or {"timing": {"hard_timeout_ms": 1000}, "recognition": {}})
    r.models, r.key = object(), None
    r.lock = threading.Lock(); r.started = 0.0; r.budget_s = 0.0
    r.warm = lambda cfg: None
    return r


class Robustness(unittest.TestCase):
    def test_invalid_user_is_refused_before_the_cameras(self):
        # D3: null / list / missing user never reaches an attempt
        r = bare_resident(); sent = []
        with mock.patch("ward.engine.auth.Engine") as eng:
            for req in ({}, {"user": None}, {"user": ["a"]}, {"user": "../x"}):
                r.attempt(req, sent.append)
        eng.assert_not_called()
        self.assertEqual(sent, [{"error": "invalid user"}] * 4)

    def test_busy_lock_answers_at_once_and_a_wedged_attempt_exits(self):
        # D1/R1: a queued request never waits behind a long or hung attempt
        r = bare_resident(); sent = []
        r.lock.acquire(); r.started = __import__("time").monotonic(); r.budget_s = 1.0
        with mock.patch.object(daemon, "BUSY_WAIT_S", 0.05), mock.patch("ward.engine.auth.Engine") as eng:
            r.attempt({"user": "octave"}, sent.append)
            self.assertEqual(sent, [{"error": "busy"}])
            r.started -= 100                                  # far past hard timeout + margin: wedged
            with mock.patch.object(daemon.os, "_exit", side_effect=SystemExit) as ex:
                with self.assertRaises(SystemExit):
                    r.attempt({"user": "octave"}, sent.append)
            ex.assert_called_once_with(1)
        eng.assert_not_called()

    def test_client_gone_while_queued_runs_no_attempt(self):
        r = bare_resident(); sent = []
        with mock.patch("ward.engine.auth.Engine") as eng:
            r.attempt({"user": "octave"}, sent.append, alive=lambda: False)
        eng.assert_not_called(); self.assertFalse(r.lock.locked())

    def test_peer_alive(self):
        import socket
        a, b = socket.socketpair()
        self.assertTrue(daemon.peer_alive(a))
        b.close()
        self.assertFalse(daemon.peer_alive(a)); a.close()

    def test_error_before_start_falls_back_to_the_engine(self):
        # D2: a daemon that errors before the attempt must not reject: the in-process engine decides
        class Broken:
            def attempt(self, req, send, **kw):
                raise KeyError("recognition")
        path = Protocol.serve(self, Broken())
        self.assertIsNone(client.via_daemon("octave", path=path))
        path = Protocol.serve(self, Broken())
        real_connect = client.connect
        with mock.patch.object(client, "connect", lambda p=None: real_connect(path)), \
             mock.patch.object(client, "_behaviour", lambda: {}), \
             mock.patch.object(client, "_human_user", lambda: "octave"), \
             mock.patch("ward.cli.main", return_value=0) as cli_main, mock.patch.object(client.os, "geteuid", lambda: 0):
            self.assertEqual(client.main([]), 0)
        cli_main.assert_called_once_with(["auth", "-U", "octave"])


class HumanUser(unittest.TestCase):
    """The PAM helper must authenticate the person at the keyboard, not root (the sudo target)."""

    def test_non_root_pam_user_wins(self):           # login / polkit / `su bob`
        with mock.patch.dict(os.environ, {"PAM_USER": "alice"}, clear=False):
            self.assertEqual(client._human_user(), "alice")

    def test_sudo_root_pam_user_falls_to_real_uid(self):   # the bug: PAM_USER=root, caller uid is the human
        import pwd
        with mock.patch.dict(os.environ, {"PAM_USER": "root"}, clear=False), \
             mock.patch.object(client.os, "getuid", lambda: 1000), \
             mock.patch.object(pwd, "getpwuid", lambda u: type("P", (), {"pw_name": "octave"})()):
            self.assertEqual(client._human_user(), "octave")

    def test_actual_root_login_stays_root(self):
        with mock.patch.dict(os.environ, {"PAM_USER": "root"}, clear=False), \
             mock.patch.object(client.os, "getuid", lambda: 0):
            self.assertEqual(client._human_user(), "root")


if __name__ == "__main__":
    unittest.main()


class SdNotify(unittest.TestCase):
    """The stdlib sd_notify / watchdog used under Type=notify (no libsystemd)."""

    def listener(self, abstract=False):
        import socket
        s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        if abstract:
            name = f"ward-test-{os.getpid()}-{id(s)}"
            s.bind("\0" + name); addr = "@" + name
        else:
            addr = os.path.join(tempfile.mkdtemp(dir=os.environ["WARD_ETC"]), "notify"); s.bind(addr)
        s.settimeout(2)
        self.addCleanup(s.close)
        return s, addr

    def test_path_socket_receives_state(self):
        s, addr = self.listener()
        self.assertTrue(daemon.sd_notify("READY=1\nSTATUS=x", env={"NOTIFY_SOCKET": addr}))
        self.assertEqual(s.recv(256), b"READY=1\nSTATUS=x")

    def test_abstract_socket(self):
        s, addr = self.listener(abstract=True)
        self.assertTrue(daemon.sd_notify("WATCHDOG=1", env={"NOTIFY_SOCKET": addr}))
        self.assertEqual(s.recv(64), b"WATCHDOG=1")

    def test_absent_or_broken_socket_is_a_noop(self):
        self.assertFalse(daemon.sd_notify("READY=1", env={}))
        self.assertFalse(daemon.sd_notify("READY=1", env={"NOTIFY_SOCKET": "relative"}))
        self.assertFalse(daemon.sd_notify("READY=1", env={"NOTIFY_SOCKET": "/nonexistent/ward/notify"}))

    def test_watchdog_interval(self):
        me = str(os.getpid())
        self.assertEqual(daemon.watchdog_interval({"WATCHDOG_USEC": "30000000", "WATCHDOG_PID": me}), 15.0)
        self.assertEqual(daemon.watchdog_interval({"WATCHDOG_USEC": "4000000"}), 2.0)
        self.assertIsNone(daemon.watchdog_interval({}))
        self.assertIsNone(daemon.watchdog_interval({"WATCHDOG_USEC": "junk"}))
        self.assertIsNone(daemon.watchdog_interval({"WATCHDOG_USEC": "0"}))
        self.assertIsNone(daemon.watchdog_interval({"WATCHDOG_USEC": "30000000", "WATCHDOG_PID": "1"}))

    def test_wedged_attempt_silences_the_watchdog(self):
        import time
        res = daemon.Resident.__new__(daemon.Resident)
        res.started, res.budget_s = 0.0, 8.0
        self.assertFalse(res.wedged())
        res.started = time.monotonic()
        self.assertFalse(res.wedged())                                   # running, within its deadline
        self.assertTrue(res.wedged(res.started + 8.0 + daemon.STUCK_MARGIN_S + 1))
        pings, stop = [], threading.Event()

        def run(wedge_at):
            pings.clear(); stop.clear()
            res.started = time.monotonic() - (100 if wedge_at else 0)
            t = threading.Thread(target=daemon.watchdog_loop, args=(res, 0.01, stop, pings.append))
            t.start(); time.sleep(0.08); stop.set(); t.join(1)
            return list(pings)
        self.assertGreaterEqual(len(run(False)), 2)
        self.assertTrue(all(p == "WATCHDOG=1" for p in pings))
        self.assertEqual(run(True), [])                                  # wedged: no ping → systemd restarts us

    def test_unit_file_declares_notify_and_watchdog(self):
        unit = open(os.path.join(os.path.dirname(__file__), "..", "data", "ward.service")).read()
        for line in ("Type=notify", "WatchdogSec=", "CapabilityBoundingSet=\n", "DevicePolicy=closed",
                     "DeviceAllow=char-video4linux rw", "RestrictAddressFamilies=AF_UNIX", "MemoryMax="):
            self.assertIn(line, unit)
