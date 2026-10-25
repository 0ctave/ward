"""IR cross-spectrum PAD harness: list parsing, name pairing, confusion metrics (no dataset, no camera)."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("WARD_ETC", tempfile.mkdtemp(prefix="ward-test-"))

from ward import ir_pad


class Lists(unittest.TestCase):
    def test_read_list_3_and_4_columns(self):
        f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        f.write("c/0.jpg d/0.jpg i/0.jpg 1\n")        # 4-col CASIA-SURF: color depth ir label(1=live)
        f.write("c/1.jpg i/1.jpg 0\n")                # 3-col: color ir label(0=spoof)
        f.close()
        items = ir_pad.read_list(f.name, "/root")
        self.assertEqual(items[0], ("/root/c/0.jpg", "/root/i/0.jpg", False))   # live
        self.assertEqual(items[1], ("/root/c/1.jpg", "/root/i/1.jpg", True))    # spoof

    def test_pair_by_name_matches_and_labels(self):
        root = tempfile.mkdtemp()
        for kind in ("live", "spoof"):
            for sub in ("color", "ir"):
                os.makedirs(os.path.join(root, sub, kind))
                open(os.path.join(root, sub, kind, "0.png"), "wb").close()
        items = ir_pad.pair_by_name(os.path.join(root, "color"), os.path.join(root, "ir"))
        self.assertEqual(len(items), 2)
        self.assertEqual({s for _, _, s in items}, {True, False})


class Metrics(unittest.TestCase):
    def test_confusion_to_rates(self):
        res = {"live": {"PASS": 90, "FAIL": 8, "ABSTAIN": 2},
               "spoof": {"PASS": 15, "FAIL": 80, "ABSTAIN": 5}, "skipped": 3}
        m = ir_pad.metrics(res)
        self.assertAlmostEqual(m["live_pass_rate"], 0.90)
        self.assertAlmostEqual(m["bpcer"], 0.08)              # live wrongly failed
        self.assertAlmostEqual(m["spoof_catch_rate"], 0.80)
        self.assertAlmostEqual(m["apcer"], 0.15)             # spoof wrongly passed
        self.assertIn("APCER", ir_pad.format_report(m, skipped=3, seconds=1.0))

    def test_evaluate_counts_votes_and_skips(self):
        votes = {("ca", "ia"): "PASS", ("cb", "ib"): "FAIL", ("cc", "ic"): None}
        items = [("ca", "ia", False), ("cb", "ib", True), ("cc", "ic", True)]
        res = ir_pad.evaluate(lambda c, i: votes[(c, i)], items)
        self.assertEqual(res["live"]["PASS"], 1)
        self.assertEqual(res["spoof"]["FAIL"], 1)
        self.assertEqual(res["skipped"], 1)


if __name__ == "__main__":
    unittest.main()
