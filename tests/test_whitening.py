"""Within-class whitening: the user's own variation directions are down-weighted, strangers are not."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("WARD_ETC", tempfile.mkdtemp(prefix="ward-test-"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ward.engine import store  # noqa: E402


def unit(v):
    return v / np.linalg.norm(v)


class WhiteningTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.identity = unit(rng.normal(size=128))
        # the user's face varies along 6 "condition" directions (pose/expression/lighting), strongly
        self.var_dirs = np.linalg.qr(rng.normal(size=(128, 6)))[0].T
        self.rng = rng

    def sample(self, pose_shift=0.0, cond=0.25):
        """An embedding of the user under some condition: identity + condition + small noise."""
        c = self.var_dirs.T @ self.rng.normal(scale=cond, size=6)
        return unit(self.identity + pose_shift + c + self.rng.normal(scale=0.03, size=128))

    def test_whitened_distance_separates_conditions_from_strangers(self):
        st = store.load("wtest")
        extra = []
        for cell in range(8):                       # 8 pose bins, 3 samples each, fused mean stored
            samples = [self.sample() for _ in range(3)]
            st.add(f"cell{cell}", "ir", np.mean(samples, axis=0), 0.9, pose=(cell * 5.0, 0.0), samples=samples)
            # synthetic lighting variants, as enrollment produces them (scatter only)
            extra += [(f"cell{cell}", unit(s + self.var_dirs.T @ self.rng.normal(scale=0.3, size=6))) for s in samples]
        ent = st.fit_whitening("ir", extra)
        self.assertIsNotNone(ent)
        # a night-time embedding as auth sees it: same identity, a coherent step along the variation
        # directions, fused over 4 frames
        night = unit(np.mean([unit(self.identity + self.var_dirs.T @ np.full(6, 0.3) + self.rng.normal(scale=0.03, size=128))
                              for _ in range(4)], axis=0))
        raw, _ = st.min_distance(night, "ir")
        wd, thr = st.wdist(night, "ir")
        # a stranger: different identity, no special relation to the user's variation directions
        stranger = unit(self.rng.normal(size=128))
        raw_s, _ = st.min_distance(stranger, "ir")
        wd_s, _ = st.wdist(stranger, "ir")
        self.assertGreater(raw, 0.4)                      # raw distance: the night face is outside the plain gate
        self.assertLessEqual(wd, thr)                     # whitened: recognised as the user's own variation
        self.assertGreater(wd_s, thr)                     # the stranger stays rejected
        self.assertGreater(wd_s / wd, raw_s / raw)        # separation improved, not just rescaled
        # persistence
        st.save()
        st2 = store.load("wtest")
        self.assertIn("ir", st2.whitening)
        wd2, thr2 = st2.wdist(night, "ir")
        self.assertAlmostEqual(wd2, wd, places=3)
        self.assertEqual(len(st2.models[0].samples), 3)

    def test_needs_enough_samples(self):
        st = store.load("wsmall")
        st.add("c", "ir", self.sample(), 0.9, samples=[self.sample()])
        self.assertIsNone(st.fit_whitening("ir"))
        self.assertEqual(st.wdist(self.sample(), "ir"), (None, None))


if __name__ == "__main__":
    unittest.main()
