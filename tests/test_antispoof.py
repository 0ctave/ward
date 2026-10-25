"""Anti-spoof CNN voter: preprocessing, asymmetric decision, fail-open (fake net, no camera)."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("WARD_ETC", tempfile.mkdtemp(prefix="ward-test-"))
os.environ.setdefault("WARD_LOG", os.path.join(os.environ["WARD_ETC"], "log"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ward.engine.detect import Box  # noqa: E402
from ward.engine.liveness import Result  # noqa: E402
from ward.engine.liveness import antispoof_cnn as asc  # noqa: E402


class FakeNet:
    """Records the blob it was given; answers fixed logits (class 1 = real)."""
    def __init__(self, logits):
        self.logits = list(logits); self.blobs = []

    def setInput(self, blob):
        self.blobs.append(blob)

    def forward(self):
        return np.array([self.logits.pop(0) if len(self.logits) > 1 else self.logits[0]], dtype=np.float32)


def colour_frame(h=360, w=640):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    img[..., 0] = 200; img[..., 2] = 40          # clearly colour: R ≠ B
    return img


REAL = [-4.0, 6.0, -4.0]
SPOOF = [8.0, -6.0, 0.0]
UNSURE = [0.0, 0.0, 0.0]


class Voter(unittest.TestCase):
    def setUp(self):
        self.box = Box(270, 120, 100, 100)

    def run_with(self, net, frames):
        with mock.patch.object(asc, "_net", net), mock.patch.object(asc, "_error", ""):
            return asc.vote_frames(frames)

    def test_bundled_model_present_and_pinned(self):
        ok, got = asc.verify()
        self.assertTrue(ok, f"bundled model checksum {got}")

    def test_preprocessing_matches_reference(self):
        net = FakeNet([REAL])
        rgb = colour_frame()
        v = self.run_with(net, [(rgb, self.box, False)])
        self.assertEqual(v.result, Result.PASS)
        blob = net.blobs[0]
        self.assertEqual(blob.shape, (1, 3, asc.INPUT, asc.INPUT))
        self.assertEqual(blob.dtype, np.float32)
        self.assertGreater(blob.max(), 1.5)                        # raw 0–255, no /255 or mean
        # BGR order: the frame's R channel (200) is the blob's LAST channel
        self.assertAlmostEqual(float(blob[0, 2].mean()), 200.0, delta=1)
        self.assertAlmostEqual(float(blob[0, 0].mean()), 40.0, delta=1)

    def test_crop_scale_and_clamp(self):
        img = np.zeros((360, 640, 3), np.uint8)
        img[:, :320] = 255                                         # left half white
        c = asc._crop(img, Box(270, 130, 100, 100))                # 270 px crop centred at x=320
        self.assertEqual(c.shape, (80, 80, 3))
        self.assertGreater(c[:, :35].mean(), 250); self.assertLess(c[:, 45:].mean(), 5)
        big = asc._crop(img, Box(200, 50, 260, 260))               # scale clamped to fit the frame
        self.assertEqual(big.shape, (80, 80, 3))
        self.assertIsNone(asc._crop(img, Box(10, 10, 20, 20)))     # too small to judge

    def test_crop_at_edge_shifts_not_clips(self):
        # reference _get_new_box: a box near the left edge keeps its 270 px size, shifted inside
        img = np.zeros((360, 640, 3), np.uint8)
        img[:, 255:271] = 255                                      # marker band ending at x=270
        c = asc._crop(img, Box(0, 130, 100, 100))                  # 270 px crop → x 0..270
        self.assertEqual(c.shape, (80, 80, 3))
        self.assertGreater(c[:, 78:].mean(), 200); self.assertLess(c[:, 70:72].mean(), 5)  # band at the right edge, not squashed

    def test_single_spoof_frame_only_abstains(self):
        v = self.run_with(FakeNet([SPOOF]), [(colour_frame(), self.box, False)])
        self.assertEqual(v.result, Result.ABSTAIN)

    def test_repeated_confident_spoof_fails(self):
        f = colour_frame()
        v = self.run_with(FakeNet([SPOOF]), [(f, self.box, False)] * 3)
        self.assertEqual(v.result, Result.FAIL)

    def test_one_real_frame_among_spoofs_abstains(self):
        f = colour_frame()
        v = self.run_with(FakeNet([SPOOF, REAL, SPOOF]), [(f, self.box, False)] * 3)
        self.assertNotEqual(v.result, Result.FAIL)

    def test_uncertain_abstains(self):
        f = colour_frame()
        v = self.run_with(FakeNet([UNSURE]), [(f, self.box, False)] * 3)
        self.assertEqual(v.result, Result.ABSTAIN)

    def test_grey_or_dark_frames_abstain(self):
        grey = np.full((360, 640, 3), 128, np.uint8)
        net = FakeNet([SPOOF])
        self.assertEqual(self.run_with(net, [(grey, self.box, False)] * 3).result, Result.ABSTAIN)
        self.assertEqual(self.run_with(net, [(colour_frame(), self.box, True)] * 3).result, Result.ABSTAIN)
        self.assertEqual(self.run_with(net, [(None, None, True)]).result, Result.ABSTAIN)
        self.assertEqual(net.blobs, [])                             # never scored

    def test_broken_net_fails_open(self):
        class Boom(FakeNet):
            def forward(self):
                raise RuntimeError("dnn")
        v = self.run_with(Boom([REAL]), [(colour_frame(), self.box, False)] * 3)
        self.assertEqual(v.result, Result.ABSTAIN)
        with mock.patch.object(asc, "_net", None), mock.patch.object(asc, "_error", "checksum mismatch"):
            self.assertEqual(asc.vote_frames([(colour_frame(), self.box, False)]).result, Result.ABSTAIN)

    def test_real_bundled_model_runs(self):
        net = asc._load()
        self.assertIsNotNone(net)
        s = asc.score(colour_frame(), self.box, net)
        self.assertTrue(0.0 <= s <= 1.0)


if __name__ == "__main__":
    unittest.main()
