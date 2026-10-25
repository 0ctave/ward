"""Replay harness (roadmap L7): record → replay round trip, and the whole engine loop on recorded frames."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("WARD_ETC", tempfile.mkdtemp(prefix="ward-test-"))
os.environ.setdefault("WARD_LOG", os.path.join(os.environ["WARD_ETC"], "log"))

from ward.engine import config, replay, store
from ward.engine.capture import CaptureError, Frame


def _frames(level: int, n: int = 6, size=(120, 160)):
    """n frame sets of flat grey `level` (no face) for IR and RGB."""
    d = Path(tempfile.mkdtemp())
    rec = replay.Recorder(d)
    for _ in range(n):
        g = np.full(size, level, np.uint8)
        rgb = np.repeat(g[..., None], 3, axis=2)
        rec.write({"ir": Frame("ir", 0.0, g, rgb), "rgb": Frame("rgb", 0.0, g, rgb)})
    rec.close()
    return d


class RoundTrip(unittest.TestCase):
    def test_record_then_replay(self):
        d = _frames(90, n=4)
        self.assertEqual(len(json.loads((d / "index.json").read_text())), 4)
        rc = replay.ReplayCapture(d)
        self.assertEqual(rc.spectra, ["ir", "rgb"])
        fs = rc.read_set()
        self.assertEqual(set(fs), {"ir", "rgb"}); self.assertEqual(int(fs["ir"].gray.mean()), 90)
        self.assertEqual(fs["rgb"].rgb.shape, (120, 160, 3))
        for _ in range(3):
            self.assertTrue(rc.read_set())
        self.assertEqual(rc.read_set(), {}); self.assertTrue(rc.exhausted)
        with self.assertRaises(CaptureError):
            replay.ReplayCapture(tempfile.mkdtemp())


class EngineOnReplay(unittest.TestCase):
    """Engine.authenticate end to end without a camera: the store has a model, frames have no face."""

    def _store(self, user):
        st = store.Store(user=user, path=store.path_for(user))
        e = np.random.default_rng(0).normal(size=128); e /= np.linalg.norm(e)
        st.add("C/level", "ir", e, 0.9, None, samples=[e]); st.add("C/level", "rgb", e, 0.9, None, samples=[e])
        st.save()

    def _run(self, d, user="replayuser"):
        from ward.engine.auth import Engine
        self._store(user)
        os.environ["WARD_REPLAY"] = str(d)
        try:
            cfg = config.load(); cfg["timing"]["window_ms"] = 400; cfg["timing"]["hard_timeout_ms"] = 800
            return Engine(cfg).authenticate(user)
        finally:
            del os.environ["WARD_REPLAY"]

    def test_dark_frames_end_in_dark(self):
        out = self._run(_frames(5))
        self.assertFalse(out.accepted); self.assertEqual(out.reason, "dark"); self.assertGreater(out.frames, 0)

    def test_blank_lit_frames_end_in_no_face(self):
        out = self._run(_frames(110))
        self.assertFalse(out.accepted); self.assertEqual(out.reason, "no-face")
        self.assertIn(out.reason, open(os.environ["WARD_LOG"]).read())          # one log line per attempt

    def test_missing_recording_is_a_camera_failure(self):
        out = self._run(Path(tempfile.mkdtemp()))
        self.assertEqual(out.reason, "camera")


if __name__ == "__main__":
    unittest.main()
