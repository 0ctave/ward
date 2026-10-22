"""Lab evaluator: policy decisions and the per-subject summary (no camera)."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("FACEUP_ETC", tempfile.mkdtemp(prefix="faceup-test-"))
os.environ.setdefault("FACEUP_LAB", tempfile.mkdtemp(prefix="faceup-lab-"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faceup.engine import lab  # noqa: E402


def sample(raw, w=None, thr=4.0, votes=(("cross", "PASS", ""), ("motion", "ABSTAIN", "")), subject="me"):
    return lab.Sample(0.0, subject, "ir", {"ir": raw is not None}, {"ir": raw}, {"ir": "C/level"},
                      {"ir": w}, {"ir": thr if w is not None else None}, list(votes), {"ir": "ok"}, 0.0, 0.0)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.ev = lab.LiveEvaluator.__new__(lab.LiveEvaluator)
        self.ev.thresholds = {"certainty": 0.40, "rgb_certainty": 0.38, "hard_cap": 0.55, "required_votes": 2}

    def test_policies_differ_where_they_should(self):
        d, _ = self.ev.decide(sample(0.30))
        self.assertEqual(d, {"strict": True, "whitened": True, "whitened+liveness": True})
        d, r = self.ev.decide(sample(0.48, w=3.0))                      # outside the gate, inside whitened
        self.assertEqual((d["strict"], d["whitened"]), (False, True)); self.assertIn("whitened", r)
        d, _ = self.ev.decide(sample(0.48, w=5.0))                      # whitened too far
        self.assertFalse(d["whitened"])
        d, _ = self.ev.decide(sample(0.58, w=1.0))                      # beyond the hard cap: never
        self.assertFalse(d["whitened"])
        d, r = self.ev.decide(sample(0.30, votes=(("cross", "FAIL", ""), ("motion", "PASS", ""))))
        self.assertEqual((d["whitened"], d["whitened+liveness"]), (True, False)); self.assertIn("FAIL", r)
        d, _ = self.ev.decide(sample(None))
        self.assertFalse(any(d.values()))

    def test_sliders_change_decisions(self):
        s = sample(0.45, w=5.0)
        self.assertFalse(self.ev.decide(s)[0]["strict"])
        self.ev.thresholds["certainty"] = 0.50
        self.assertTrue(self.ev.decide(s)[0]["strict"])

    def test_summary_per_subject(self):
        self.ev.records = []
        for raw in (0.25, 0.30, 0.45):
            s = sample(raw, w=3.0); s.decisions, s.reason = self.ev.decide(s); self.ev.records.append(s)
        for raw in (0.62, 0.70):
            s = sample(raw, w=9.0, subject="friend"); s.decisions, s.reason = self.ev.decide(s); self.ev.records.append(s)
        rows = {r["subject"]: r for r in self.ev.summary()}
        self.assertEqual(rows["me"]["n"], 3)
        self.assertAlmostEqual(rows["me"]["strict"], 2 / 3)
        self.assertEqual(rows["me"]["whitened"], 1.0)
        self.assertEqual(rows["friend"]["whitened"], 0.0)
        self.assertGreater(rows["friend"]["raw_min"], 0.55)
        text = lab.format_summary(self.ev.summary())
        self.assertIn("friend", text); self.assertIn("100%", text)


if __name__ == "__main__":
    unittest.main()


class LivenessOutsideAuthenticate(unittest.TestCase):
    def test_shadow_voter_abstains_on_a_fresh_engine(self):
        """The lab (and the calibration session) call Engine._liveness without authenticate():
        the shadow test's per-attempt state must exist and yield ABSTAIN, never AttributeError."""
        import numpy as np
        from faceup.engine.auth import Engine, Obs
        from faceup.engine.detect import Box
        from faceup.engine.capture import Frame
        from faceup.engine.liveness import Result
        eng = Engine.__new__(Engine)
        eng.verbose = False; eng.events = None; eng.cfg = {"liveness": {"shadow_toggle_ms": 33}}
        # emulate what __init__ sets without loading models
        eng._cap = None; eng._emitter = None; eng._shadow_done = None
        g = np.zeros((60, 80), np.uint8)
        fr = Frame("ir", 0.0, g, np.repeat(g[..., None], 3, axis=2))
        obs = {"ir": [Obs(fr, None, Box(10, 10, 30, 30), None, False, "ok")]}
        live = {"cross_spectrum": False, "antispoof_cnn": False, "shadow_test": "auto", "shadow_toggle_ms": 33, "motion": False}
        votes = eng._liveness(None, obs, {"ir": "/dev/videoX"}, live)
        self.assertEqual([(v.voter, v.result) for v in votes], [("shadow", Result.ABSTAIN)])
        self.assertIn("unavailable", votes[0].reason)

    def test_engine_init_creates_the_per_attempt_state(self):
        import inspect
        from faceup.engine.auth import Engine
        src = inspect.getsource(Engine.__init__)
        for name in ("_cap", "_emitter", "_shadow_done"):
            self.assertIn(f"self.{name} = None", src)
