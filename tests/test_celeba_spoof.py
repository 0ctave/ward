"""CelebA-Spoof benchmark: image collection, label handling, APCER/BPCER math and the skip path
(no dataset, no camera — a fake scorer and a tiny synthetic tree stand in)."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("WARD_ETC", tempfile.mkdtemp(prefix="ward-test-"))

from ward import celeba_spoof as cs


def tree():
    """A minimal CelebA-Spoof-like tree: 1 subject, 2 live + 2 spoof PNG stubs."""
    root = tempfile.mkdtemp()
    for kind, n in (("live", 2), ("spoof", 2)):
        d = os.path.join(root, "9999", kind); os.makedirs(d)
        for i in range(n):
            open(os.path.join(d, f"{i}.png"), "wb").close()
            open(os.path.join(d, f"{i}_BB.txt"), "w").write("30 30 160 160 0.99\n")
    return root


class Collect(unittest.TestCase):
    def test_labels_from_folder(self):
        items = cs.collect_images(tree())
        self.assertEqual(len(items), 4)
        self.assertEqual(sum(s for _, s, _ in items), 2)            # 2 spoof
        self.assertFalse(any(p.endswith("_BB.txt") for p, _, _ in items))

    def test_per_type_from_labels_json(self):
        root = tree()
        # label a spoof image as a phone screen (index 40 == 9)
        spoof_img = next(p for p in (os.path.join(dp, f) for dp, _, fs in os.walk(root) for f in fs)
                         if "spoof" in p and p.endswith("0.png"))
        lbl = [0] * 43; lbl[cs.SPOOF_TYPE_INDEX] = 9
        labels = {"/".join(spoof_img.split("/")[-4:]): lbl}
        items = cs.collect_images(root, labels)
        kinds = {k for _, s, k in items if s}
        self.assertIn("phone-screen", kinds)

    def test_balanced_subset(self):
        items = [("l%d" % i, False, "live") for i in range(50)] + [("s%d" % i, True, "spoof") for i in range(50)]
        s = cs.balanced_subset(items, 20, seed=3)
        self.assertEqual(len(s), 20); self.assertEqual(sum(x[1] for x in s), 10)
        self.assertEqual(s, cs.balanced_subset(items, 20, seed=3))

    def test_read_bb_scales_from_224(self):
        root = tree()
        img = next(os.path.join(dp, f) for dp, _, fs in os.walk(root) for f in fs if f.endswith("0.png"))
        bb = cs.read_bb(img, 448, 224)                              # width doubled → x,w doubled
        self.assertEqual(bb, (60, 30, 320, 160))


class Metrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.live = np.clip(rng.normal(0.98, 0.03, 1000), 0, 1)     # real faces score high
        self.spoof = np.clip(rng.normal(0.05, 0.05, 1000), 0, 1)    # spoofs score low

    def test_apcer_bpcer_direction(self):
        # at the live-vote threshold 0.90: few spoofs pass as live, few real faces rejected
        self.assertLess(cs.apcer(self.spoof, 0.90), 0.02)
        self.assertLess(cs.bpcer(self.live, 0.90), 0.05)
        # a lower threshold lets MORE attacks through (APCER up) and rejects FEWER real (BPCER down)
        self.assertGreaterEqual(cs.apcer(self.spoof, 0.10), cs.apcer(self.spoof, 0.90))
        self.assertLessEqual(cs.bpcer(self.live, 0.10), cs.bpcer(self.live, 0.90))

    def test_separation_and_eer(self):
        self.assertGreater(cs.auc(self.live, self.spoof), 0.99)
        t, rate = cs.eer(self.live, self.spoof)
        self.assertTrue(0.0 < t < 1.0); self.assertLess(rate, 0.03)

    def test_metrics_and_report(self):
        by_type = {"live": self.live, "phone-screen": self.spoof[:500], "print": self.spoof[500:]}
        m = cs.metrics(self.live, self.spoof, by_type,
                       {"PASS_P (live vote)": 0.90, "FAIL_P (spoof vote)": 0.02})
        self.assertEqual(m["n_spoof"], 1000)
        self.assertIn("phone-screen", m["per_type_apcer"])
        self.assertIn("ACER", cs.format_report(m, skipped=5, seconds=1.0))


class Evaluate(unittest.TestCase):
    def test_skips_unscorable_images(self):
        scores = {"a": 0.99, "b": 0.01, "c": None}                  # c: no face
        items = [("a", False, "live"), ("b", True, "spoof"), ("c", True, "spoof")]
        live, spoof, by_type, skipped = cs.evaluate(lambda p: scores[p], items)
        self.assertEqual(skipped, 1)
        self.assertEqual(list(live), [0.99]); self.assertEqual(list(spoof), [0.01])
        self.assertIn("spoof", by_type)


if __name__ == "__main__":
    unittest.main()
