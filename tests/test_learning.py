"""Online learning stays bounded (review D1/D2): learned models never anchor further learning,
one frame never learns, a total cap per camera, and a refit moves the whitening threshold ±20 %."""
from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("FACEUP_ETC", tempfile.mkdtemp(prefix="faceup-learn-"))

from faceup.engine import store  # noqa: E402
from faceup.engine.auth import Engine  # noqa: E402
from faceup.engine.liveness import Result, Vote  # noqa: E402


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


class Learning(unittest.TestCase):
    BEH = {"learn": True, "learn_max_dist": 0.30, "learn_min_novelty": 0.10, "learn_per_bin": 50, "learn_max_total": 3}

    def setUp(self):
        self.rng = np.random.default_rng(3)
        self.base = unit(self.rng.normal(size=128))
        self.st = store.Store(user="learn", path=store.path_for("learn"))
        self.st.add("C/level", "ir", self.base, 0.9, None, samples=[self.base] * 3)
        self.st.save = lambda: None
        self.votes = [Vote("shadow", Result.PASS, "")]

    def learn(self, emb, frames=3):
        eng = types.SimpleNamespace(_last_fused={"ir": emb}, _last_samples={"ir": [emb] * frames}, models=None)
        return Engine._learn(eng, self.st, {}, {"ir": 0.0}, self.votes, self.BEH, {})

    def step_away(self, frm, d):
        """A unit vector at distance ~d from `frm`, moving further from the original each time."""
        away = unit(frm - self.base + 1e-9 * self.rng.normal(size=128)) if np.linalg.norm(frm - self.base) > 0 else unit(self.rng.normal(size=128))
        return unit(frm + d * away)

    def test_learned_models_do_not_chain(self):
        e1 = self.step_away(self.base, 0.22)
        self.assertTrue(self.learn(e1))                                     # 0.22 from the original: learned
        e2 = self.step_away(e1, 0.22)                                        # 0.22 from e1 but ~0.44 from the original
        self.assertGreater(self.st.distance(self.base, e2), 0.30)
        self.assertEqual(self.learn(e2), "")                                 # a learned model is no anchor

    def test_motion_only_accept_never_learns(self):
        # A3: a motion PASS alone is weak liveness — it must not widen the accept region
        self.votes = [Vote("motion", Result.PASS, ""), Vote("cross", Result.ABSTAIN, "")]
        self.assertEqual(self.learn(self.step_away(self.base, 0.2)), "")

    def test_learn_distance_never_exceeds_gate_minus_margin(self):
        e = self.step_away(self.base, 0.28)
        eng = types.SimpleNamespace(_last_fused={"ir": e}, _last_samples={"ir": [e] * 3}, models=None)
        beh = dict(self.BEH, learn_max_dist=0.60)                            # hand-edited past the gate
        self.assertEqual(Engine._learn(eng, self.st, {}, {"ir": 0.0}, self.votes, beh, {"certainty": 0.35}), "")

    def test_single_frame_never_learns(self):
        self.assertEqual(self.learn(self.step_away(self.base, 0.2), frames=1), "")

    def test_total_cap_per_camera(self):
        for k in range(6):
            e = unit(self.base + 0.12 * unit(self.rng.normal(size=128)))
            self.learn(e)
        self.assertLessEqual(len([m for m in self.st.models if m.learned]), self.BEH["learn_max_total"])

    def test_refit_threshold_moves_at_most_20_percent(self):
        for k in range(8):
            self.st.add(f"P{k}", "ir", unit(self.base + 0.1 * self.rng.normal(size=128)), 0.9, None,
                        samples=[unit(self.base + 0.1 * self.rng.normal(size=128)) for _ in range(4)])
        self.st.fit_whitening("ir")
        self.st.whitening["ir"]["threshold"] = 1.0                           # a tight session threshold
        self.st.whitening["ir"]["floor"] = 1.0
        self.st.fit_whitening("ir", clamp=0.2)
        self.assertLessEqual(self.st.whitening["ir"]["threshold"], 1.2 + 1e-9)
        self.assertEqual(self.st.whitening["ir"]["floor"], 1.0)              # the session floor survives


if __name__ == "__main__":
    unittest.main()
