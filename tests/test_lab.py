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
