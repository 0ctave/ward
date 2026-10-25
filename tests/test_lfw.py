"""LFW benchmark harness: pair parsing, metrics math and the embedding/skip path (no dataset,
no camera — a fake embedder stands in for the detector + network)."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("WARD_ETC", tempfile.mkdtemp(prefix="ward-test-"))

from ward import lfw


class Pairs(unittest.TestCase):
    def test_parse_same_and_different(self):
        f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        f.write("10\t300\n")                       # header: ignored
        f.write("Abel_Pacheco\t1\t4\n")            # same person, images 1 and 4
        f.write("Amber_Tamblyn\t1\tAnna_Kournikova\t1\n")   # different people
        f.close()
        pairs = lfw.read_pairs(f.name, "/root", ext=".jpg")
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0], ("/root/Abel_Pacheco/Abel_Pacheco_0001.jpg",
                                    "/root/Abel_Pacheco/Abel_Pacheco_0004.jpg", True))
        self.assertEqual(pairs[1], ("/root/Amber_Tamblyn/Amber_Tamblyn_0001.jpg",
                                    "/root/Anna_Kournikova/Anna_Kournikova_0001.jpg", False))

    def test_balanced_subset_is_even_and_deterministic(self):
        pairs = [("a", "b", True)] * 100 + [("c", "d", False)] * 100
        s = lfw.balanced_subset(pairs, 40, seed=1)
        self.assertEqual(len(s), 40)
        self.assertEqual(sum(p[2] for p in s), 20)          # 20 same, 20 different
        self.assertEqual(s, lfw.balanced_subset(pairs, 40, seed=1))   # deterministic


class Metrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.genuine = np.clip(rng.normal(0.25, 0.05, 2000), 0.02, None)
        self.impostor = np.clip(rng.normal(0.70, 0.08, 2000), 0.02, None)

    def test_rates_and_separation(self):
        fmr, fnmr = lfw.rates_at(self.genuine, self.impostor, 0.40)
        self.assertLess(fmr, 0.01)                 # strangers almost never inside a 0.40 gate
        self.assertLess(fnmr, 0.01)                # and you are almost never rejected
        self.assertGreater(lfw.auc(self.genuine, self.impostor), 0.99)
        t, rate = lfw.eer(self.genuine, self.impostor)
        self.assertTrue(0.3 < t < 0.6); self.assertLess(rate, 0.02)

    def test_monotonic_fmr(self):
        f_low, _ = lfw.rates_at(self.genuine, self.impostor, 0.3)
        f_high, _ = lfw.rates_at(self.genuine, self.impostor, 0.9)
        self.assertLessEqual(f_low, f_high)        # a looser gate accepts at least as many impostors

    def test_metrics_shape(self):
        m = lfw.metrics(self.genuine, self.impostor, {"gate": 0.4, "classic": 0.6})
        self.assertEqual(m["n_genuine"], 2000)
        self.assertIn("gate", m["at"]); self.assertIn("fmr", m["at"]["gate"])
        self.assertIn("LFW", lfw.format_report(m, skipped=3, seconds=1.0))


class EvaluateAndEmbed(unittest.TestCase):
    def test_evaluate_skips_faceless_pairs(self):
        emb = {"a": np.array([0.0, 0.0]), "b": np.array([0.1, 0.0]),
               "c": np.array([1.0, 0.0]), "x": None}      # x: no face
        pairs = [("a", "b", True), ("a", "c", False), ("a", "x", True)]
        g, i, skipped = lfw.evaluate(lambda p: emb[p], pairs)
        self.assertEqual(skipped, 1)
        self.assertEqual(len(g), 1); self.assertEqual(len(i), 1)
        self.assertAlmostEqual(g[0], 0.1); self.assertAlmostEqual(i[0], 1.0)

    def test_embedder_caches_and_counts_misses(self):
        class FakeModels:
            def __init__(self): self.calls = 0
            def detect(self, gray):
                self.calls += 1
                return None if self.calls == 1 else object()   # first image: no face
            def embed(self, rgb, box): return np.ones(4) / 2.0
        import unittest.mock as mock

        import cv2
        fm = FakeModels()
        e = lfw.Embedder(fm)
        with mock.patch.object(cv2, "imread", lambda p, flag=1: np.zeros((20, 20, 3), np.uint8)):
            self.assertIsNone(e("img1.jpg"))                   # no face → miss
            self.assertIsNotNone(e("img2.jpg"))                # face → embedding
            e("img1.jpg")                                      # cached: detect not called again
        self.assertEqual(e.misses, 1)
        self.assertEqual(fm.calls, 2)                          # img1 then img2, img1 cached


if __name__ == "__main__":
    unittest.main()
