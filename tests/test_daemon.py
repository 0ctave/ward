"""The resident engine: the socket protocol, feedback relay and the fall-back to the in-process engine."""
import os
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("FACEUP_ETC", tempfile.mkdtemp(prefix="faceup-test-"))

from faceup import client, daemon  # noqa: E402


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
        d = tempfile.mkdtemp(dir=os.environ["FACEUP_ETC"]); path = os.path.join(d, "s")
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
        from faceup import feedback
        from faceup.engine.auth import Outcome
        from faceup.engine.liveness import Result, Vote
        o = Outcome(True, "accept", "u", ir_dist=0.2, votes=[Vote("cross", Result.PASS)], elapsed_ms=900)
        d = daemon.outcome_dict(o); d["votes"] = [client._vote(v) for v in d["votes"]]
        self.assertIn("IR 0.20", feedback._detail(SimpleNamespace(**d)))

    def test_client_falls_back_to_the_engine(self):
        with mock.patch.object(client, "connect", lambda path=None: None), \
             mock.patch("faceup.cli.main", return_value=7) as cli_main, mock.patch.object(client.os, "geteuid", lambda: 0):
            self.assertEqual(client.main(["-v"]), 7)
        cli_main.assert_called_once_with(["auth", "-v"])
        with mock.patch("faceup.cli.main", return_value=3) as cli_main:
            self.assertEqual(client.main(["--replay", "/x"]), 3)          # anything but -v goes to the full CLI
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
        with mock.patch("faceup.engine.auth.Engine") as eng:
            for req in ({}, {"user": None}, {"user": ["a"]}, {"user": "../x"}):
                r.attempt(req, sent.append)
        eng.assert_not_called()
        self.assertEqual(sent, [{"error": "invalid user"}] * 4)

    def test_busy_lock_answers_at_once_and_a_wedged_attempt_exits(self):
        # D1/R1: a queued request never waits behind a long or hung attempt
        r = bare_resident(); sent = []
        r.lock.acquire(); r.started = __import__("time").monotonic(); r.budget_s = 1.0
        with mock.patch.object(daemon, "BUSY_WAIT_S", 0.05), mock.patch("faceup.engine.auth.Engine") as eng:
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
        with mock.patch("faceup.engine.auth.Engine") as eng:
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
             mock.patch("faceup.cli.main", return_value=0) as cli_main, mock.patch.object(client.os, "geteuid", lambda: 0):
            self.assertEqual(client.main([]), 0)
        cli_main.assert_called_once_with(["auth"])


if __name__ == "__main__":
    unittest.main()
