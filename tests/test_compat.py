"""Compatibility matrix (publication): the whole pipeline on recorded frames with a synthetic
face, across hardware profiles — IR+RGB, IR only, RGB only, other frame sizes, shadow test with
no emitter — and the lab evaluator driven the same way. No camera, no dlib inference: the
detector, landmarks and embedding are patched to deterministic stand-ins so what is exercised
is ward's own logic (capture, fusion, gates, whitening, liveness votes, decision, learning,
rolling calibration, lab policies)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("WARD_ETC", tempfile.mkdtemp(prefix="ward-compat-"))
os.environ.setdefault("WARD_LOG", os.path.join(os.environ["WARD_ETC"], "log"))

from ward.engine import config, replay, store
from ward.engine.capture import Frame
from ward.engine.detect import Box, Models

RNG = np.random.default_rng(11)
USER_VEC = RNG.normal(size=128); USER_VEC /= np.linalg.norm(USER_VEC)
OTHER_VEC = RNG.normal(size=128); OTHER_VEC /= np.linalg.norm(OTHER_VEC)


def frames_dir(size=(360, 640), n=8, level=110, spectra=("ir", "rgb"), rgb_size=None):
    d = Path(tempfile.mkdtemp())
    rec = replay.Recorder(d)
    for _ in range(n):
        fs = {}
        for s in spectra:
            h, w = (rgb_size or size) if s == "rgb" else size
            g = np.full((h, w), level, np.uint8)
            g[h // 4:3 * h // 4, w // 3:2 * w // 3] = level + 40          # a "face" blob, some texture
            g += (np.random.default_rng(1).random((h, w)) * 12).astype(np.uint8)   # SAME texture every frame: a still face
            fs[s] = Frame(s, 0.0, g, np.repeat(g[..., None], 3, axis=2))
        rec.write(fs)
    rec.close()
    return d


def synthetic_face(who=USER_VEC, jitter=0.02, face_frac=0.33):
    """Patches that make any frame contain `who`: detect → a centred box, embed → who + noise,
    landmarks68 → a plausible frontal 68-point layout inside the box."""
    def detect(self, gray, near=None):
        h, w = gray.shape[:2]
        s = int(min(h, w) * face_frac)
        return Box((w - s) // 2, (h - s) // 2, s, s, None)

    def embed(self, rgb, box):
        v = who + RNG.normal(scale=jitter, size=who.shape)
        return v / np.linalg.norm(v)

    def landmarks68(self, gray, box):
        # eyes, nose, mouth roughly where a frontal face has them; jaw along the box bottom
        x, y, w, h = box.x, box.y, box.w, box.h
        pts = np.zeros((68, 2))
        for i in range(17):
            pts[i] = (x + w * i / 16, y + h * (0.55 + 0.45 * np.sin(np.pi * i / 16)))
        for i in range(17, 27):
            pts[i] = (x + w * (0.2 + 0.6 * (i - 17) / 9), y + h * 0.28)
        for i in range(27, 36):
            pts[i] = (x + w * 0.5, y + h * (0.35 + 0.3 * (i - 27) / 8))
        for i in range(36, 42):
            pts[i] = (x + w * (0.3 + 0.1 * np.cos(2 * np.pi * (i - 36) / 6)), y + h * (0.42 + 0.03 * np.sin(2 * np.pi * (i - 36) / 6)))
        for i in range(42, 48):
            pts[i] = (x + w * (0.7 + 0.1 * np.cos(2 * np.pi * (i - 42) / 6)), y + h * (0.42 + 0.03 * np.sin(2 * np.pi * (i - 42) / 6)))
        for i in range(48, 68):
            pts[i] = (x + w * (0.5 + 0.18 * np.cos(2 * np.pi * (i - 48) / 20)), y + h * (0.75 + 0.06 * np.sin(2 * np.pi * (i - 48) / 20)))
        return pts

    return mock.patch.multiple(Models, detect=detect, embed=embed, landmarks68=landmarks68,
                               preload=lambda self: None)


def enrolled_store(user, spectra=("ir", "rgb"), dim=128):
    st = store.Store(user=user, path=store.path_for(user))
    for sp in spectra:
        samples = []
        for _ in range(5):
            v = USER_VEC + RNG.normal(scale=0.02, size=dim); samples.append(v / np.linalg.norm(v))
        st.add("C/level", sp, USER_VEC, 0.9, (0.0, 0.0), samples=samples)
        st.add("L1/level", sp, USER_VEC, 0.9, (-18.0, 0.0), samples=samples)
    for sp in spectra:
        st.fit_whitening(sp)
    st.calib["neutral_ratios"] = None
    st.save()
    return st


def run_auth(user, d, cfg_over=None, who=USER_VEC):
    from ward.engine.auth import Engine
    cfg = config.load()
    cfg["timing"]["window_ms"] = 600; cfg["timing"]["hard_timeout_ms"] = 1500; cfg["timing"]["settle_ms"] = 0
    cfg["cameras"]["ir"] = "usb:1:1:2"; cfg["cameras"]["rgb"] = "usb:1:1:0"
    for sec, kv in (cfg_over or {}).items():
        cfg[sec].update(kv)
    os.environ["WARD_REPLAY"] = str(d)
    try:
        with synthetic_face(who):
            return Engine(cfg).authenticate(user)
    finally:
        os.environ.pop("WARD_REPLAY", None)


class Profiles(unittest.TestCase):
    """Engine.authenticate end to end per hardware profile."""

    def test_profile_a_ir_and_rgb_accepts_the_user(self):
        enrolled_store("pa")
        out = run_auth("pa", frames_dir())
        self.assertTrue(out.accepted, out.logline())
        self.assertIsNotNone(out.ir_dist); self.assertIsNotNone(out.rgb_dist)
        self.assertLess(out.ir_dist, 0.25)

    def test_spoof_fail_is_sticky_for_the_attempt(self):
        """A1: a cross FAIL on one frame then PASSes on later frames (an attacker probing angles)
        must not accept; the attempt ends as spoof:cross so the lockout can count it."""
        from ward.engine import auth
        from ward.engine.liveness import FAIL, PASS
        enrolled_store("sticky")
        seq = iter([FAIL("cross", "face in RGB only — screen replay?")])
        with mock.patch.object(auth.cross_spectrum, "vote", lambda *a, **k: next(seq, PASS("cross", "agree"))):
            out = run_auth("sticky", frames_dir())
        self.assertFalse(out.accepted, out.logline())
        self.assertEqual(out.reason, "spoof:cross")

    def test_profile_b_ir_only(self):
        """IR only, no probed emitter, a STILL face: cross-spectrum has no RGB, shadow has no
        toggle, motion abstains — the documented rule asks for the password (the on-screen hint
        says "move your head a little", which makes motion PASS). Recognition itself is fine, and
        the degraded rule accepts."""
        enrolled_store("pb", spectra=("ir",))
        out = run_auth("pb", frames_dir(spectra=("ir",)))
        self.assertFalse(out.accepted); self.assertEqual(out.reason, "liveness"); self.assertIsNone(out.rgb_dist)
        self.assertLess(out.ir_dist, 0.25)                     # it DID recognise the user
        out = run_auth("pb", frames_dir(spectra=("ir",)), {"liveness": {"require_liveness_when_available": False}})
        self.assertTrue(out.accepted, out.logline())

    def test_profile_c_rgb_only_degrades_honestly(self):
        """RGB only: cross-spectrum and shadow cannot vote; with a still face only motion could,
        so the default rule holds (liveness required) → 'liveness', never a crash; with the
        degraded rule the user gets in."""
        enrolled_store("pc", spectra=("rgb",))
        out = run_auth("pc", frames_dir(spectra=("rgb",)))
        self.assertFalse(out.accepted); self.assertIn(out.reason, ("liveness", "no-match"))
        out = run_auth("pc", frames_dir(spectra=("rgb",)), {"liveness": {"require_liveness_when_available": False}})
        self.assertTrue(out.accepted, out.logline())

    def test_other_frame_sizes(self):
        enrolled_store("hd")
        out = run_auth("hd", frames_dir(size=(720, 1280), rgb_size=(480, 640)))
        self.assertTrue(out.accepted, out.logline())
        out = run_auth("hd", frames_dir(size=(240, 320)))
        self.assertTrue(out.accepted, out.logline())

    def test_stranger_is_rejected_and_ends_early(self):
        enrolled_store("str")
        cfg_over = {"recognition": {"early_reject": 0.65}}
        out = run_auth("str", frames_dir(n=20), cfg_over, who=OTHER_VEC)
        self.assertFalse(out.accepted); self.assertEqual(out.reason, "no-match")
        self.assertLess(out.frames, 20)                       # early reject fired before the recording ran out

    def test_shadow_enabled_without_emitter_abstains(self):
        enrolled_store("sh")
        out = run_auth("sh", frames_dir(), {"liveness": {"shadow_test": True}})
        self.assertTrue(out.accepted, out.logline())
        self.assertTrue(any(v.voter == "shadow" and v.result.value == "ABSTAIN" for v in out.votes), [str(v) for v in out.votes])

    def test_learning_and_rolling_calibration_ran(self):
        enrolled_store("learn")
        for _ in range(3):
            out = run_auth("learn", frames_dir())
            self.assertTrue(out.accepted)
        st = store.load("learn")
        self.assertIn("recent", st.calib); self.assertGreaterEqual(len(st.calib["recent"]["ir"]), 3)


class Lab(unittest.TestCase):
    """The lab evaluator on the same synthetic face: step, decide, record, summary."""

    def test_lab_step_with_a_face(self):
        from ward.engine import lab as labmod
        enrolled_store("lab")
        d = frames_dir(n=6)
        cfg = config.load(); cfg["cameras"]["ir"] = "usb:1:1:2"; cfg["cameras"]["rgb"] = "usb:1:1:0"
        with synthetic_face():
            ev = labmod.LiveEvaluator("lab", cfg)
            ev.cap = replay.ReplayCapture(d); ev.devices = {"ir": "replay:ir", "rgb": "replay:rgb"}
            ev.primary = "ir"
            from collections import deque
            ev.obs = {s: deque(maxlen=ev.window) for s in ev.devices}
            ev.start_recording("me")
            samples = [ev.step() for _ in range(6)]
            ev.stop_recording()
        samples = [s for s in samples if s is not None]
        self.assertTrue(samples)
        last = samples[-1]
        self.assertTrue(last.faces["ir"]); self.assertIsNotNone(last.raw["ir"])
        self.assertIn("strict", last.decisions); self.assertTrue(last.decisions["strict"], last.reason)
        self.assertIsNotNone(last.yaw)                       # pose ran on the synthetic landmarks
        rows = ev.summary()
        self.assertEqual(rows[0]["subject"], "me"); self.assertGreater(rows[0]["n"], 0)
        self.assertIn("me", labmod.format_summary(rows))

    def test_lab_single_camera(self):
        from ward.engine import lab as labmod
        enrolled_store("lab1", spectra=("ir",))
        d = frames_dir(n=4, spectra=("ir",))
        cfg = config.load(); cfg["cameras"]["ir"] = "usb:1:1:2"; cfg["cameras"]["rgb"] = ""
        with synthetic_face():
            ev = labmod.LiveEvaluator("lab1", cfg)
            ev.cap = replay.ReplayCapture(d); ev.devices = {"ir": "replay:ir"}; ev.primary = "ir"
            from collections import deque
            ev.obs = {"ir": deque(maxlen=ev.window)}
            out = [ev.step() for _ in range(4)]
        self.assertTrue(any(s and s.raw.get("ir") is not None for s in out))


if __name__ == "__main__":
    unittest.main()
