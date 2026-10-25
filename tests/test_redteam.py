"""Red-team harness: the digital transforms and the bypass accounting (no camera; a fake models/anti)."""
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("WARD_ETC", tempfile.mkdtemp(prefix="ward-test-"))

from ward import redteam


class Transforms(unittest.TestCase):
    def test_every_attack_keeps_a_valid_frame(self):
        img = (np.random.default_rng(0).integers(0, 255, (120, 120, 3))).astype(np.uint8)
        for name, (fn, _spoof) in redteam.ATTACKS.items():
            out = fn(img)
            self.assertEqual(out.shape, img.shape, name)
            self.assertEqual(out.dtype, np.uint8, name)

    def test_spoof_sims_are_flagged(self):
        self.assertTrue(redteam.ATTACKS["SCREEN REPLAY"][1])
        self.assertTrue(redteam.ATTACKS["PRINT"][1])
        self.assertFalse(redteam.ATTACKS["clean"][1])


class Accounting(unittest.TestCase):
    """`run` with fakes: the recogniser returns a fixed distance, the CNN a fixed p_real, so the
    recognised / antispoof-pass / bypass counts are exactly predictable."""

    def _fakes(self, dist, p_real, detect=True):
        box = object()
        models = SimpleNamespace(
            detect=lambda gray: box if detect else None,
            embed=lambda rgb, b: np.zeros(4),
            distance_scale=1.0,
        )
        anti = SimpleNamespace(_load=lambda: object(), PASS_P=0.90,
                               score=lambda rgb, b, net: p_real)
        # template at distance `dist` from the zero embedding the fake returns
        template = np.array([dist, 0, 0, 0], dtype=float)
        return models, anti, template

    def test_recognised_and_bypass_when_close_and_live_looking(self):
        models, anti, template = self._fakes(dist=0.2, p_real=0.99)
        held = [("a", np.zeros((10, 10, 3), np.uint8))] * 4
        rows = redteam.run(models, anti, template, held, gate=0.33, pass_p=0.90)
        self.assertEqual(rows["clean"]["recognised"], 1.0)
        self.assertEqual(rows["SCREEN REPLAY"]["bypass_rate"], 1.0)   # close + high p_real → bypass
        self.assertIsNone(rows["clean"]["bypass_rate"])               # not a spoof sim

    def test_no_bypass_when_antispoof_catches_it(self):
        models, anti, template = self._fakes(dist=0.2, p_real=0.01)   # recognised but CNN says spoof
        held = [("a", np.zeros((10, 10, 3), np.uint8))] * 4
        rows = redteam.run(models, anti, template, held, gate=0.33, pass_p=0.90)
        self.assertEqual(rows["SCREEN REPLAY"]["recognised"], 1.0)
        self.assertEqual(rows["SCREEN REPLAY"]["bypass_rate"], 0.0)
        self.assertEqual(rows["SCREEN REPLAY"]["antispoof_pass"], 0.0)

    def test_no_detect_counts_against_recognition(self):
        models, anti, template = self._fakes(dist=0.2, p_real=0.99, detect=False)
        held = [("a", np.zeros((10, 10, 3), np.uint8))] * 4
        rows = redteam.run(models, anti, template, held, gate=0.33, pass_p=0.90)
        self.assertEqual(rows["clean"]["no_detect"], 4)
        self.assertEqual(rows["clean"]["recognised"], 0.0)

    def test_verdict_tags_weak_bypass(self):
        rows = {"clean": {"is_spoof": False, "n": 4, "recognised": 1.0, "no_detect": 0,
                          "dist_mean": 0.2, "p_real_mean": 1.0, "antispoof_pass": 1.0, "bypass_rate": None},
                "SCREEN REPLAY": {"is_spoof": True, "n": 4, "recognised": 1.0, "no_detect": 0,
                                  "dist_mean": 0.2, "p_real_mean": 1.0, "antispoof_pass": 1.0, "bypass_rate": 1.0}}
        v = " ".join(redteam.verdict(rows))
        self.assertIn("WEAK", v); self.assertIn("ir_pad", v)


if __name__ == "__main__":
    unittest.main()
