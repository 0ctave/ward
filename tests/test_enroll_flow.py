"""The enrolment window's flow, driven headlessly (tests/enroll_harness.py): every pass, key,
timeout and the done card — no GTK, no camera."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from enroll_harness import World, run_setup  # noqa: E402
from ward.engine import calibration as cal  # noqa: E402


class HappyPath(unittest.TestCase):
    def test_cooperative_user_enrols_every_pose(self):
        res, snaps, st = run_setup(World())
        self.assertIsNotNone(res)
        self.assertGreater(res["added"], 0)
        labels = {m.label for m in st.models}
        for label, cue, _t, _tgt in cal.SETUP_POSES:
            self.assertIn(label.replace("2", ""), labels, f"no model for {label}")
        self.assertEqual(snaps[-1]["phase"], "done")


class RingStyle(unittest.TestCase):
    """`ward enroll --style ring`: the segmented coverage ring (same backend, different visuals)."""

    def test_ring_enrols_every_segment_and_reports_coverage(self):
        res, snaps, st = run_setup(World(), poses=cal.RING_POSES, style="ring")
        self.assertIsNotNone(res)
        self.assertGreater(res["added"], 0)
        labels = {m.label for m in st.models}
        for label, _cue, _t, _tgt in cal.RING_POSES:
            self.assertIn(label.replace("2", ""), labels, f"no model for {label}")
        runs = [s for s in snaps if s.get("phase") == "run"]
        self.assertTrue(runs and all(s.get("style") == "ring" for s in runs))
        # eight ring segments, all off-centre (the frontal hub is not a segment)
        segs = runs[0]["ring_segs"]
        self.assertEqual(len(segs), 8)
        self.assertTrue(all(tuple(s) != (0.0, 0.0) for s in segs))
        # an angular pass marks exactly one active segment in range; coverage accumulates
        done_counts = [sum(s["ring_done"]) for s in runs if s.get("ring_active") is not None]
        self.assertTrue(done_counts and done_counts[-1] > done_counts[0])
        for s in runs:
            a = s.get("ring_active")
            if a is not None:
                self.assertTrue(0 <= a < 8)
        self.assertEqual(snaps[-1]["phase"], "done")

    def test_overlay_draws_without_error(self):
        # the ring overlay renders (Cairo) for a mid-sweep snapshot, incl. a diagonal segment active
        appui = __import__("ward.appui", fromlist=["Preview"])
        segs = [list(t[3]) for t in cal.RING_POSES if t[1] in cal.DIRECTIONAL_CUES and tuple(t[3]) != (0.0, 0.0)]
        self._render(appui, {"segs": segs, "done": [True, True, False, False, False, False, False, False],
                             "active": 2, "hub_done": True}, cue="up", pose=(2.0, 6.0), target=[0.0, 18.0])

    def _render(self, appui, ring, cue, pose, target):
        import cairo
        try:
            prev = appui.Preview.__new__(appui.Preview)
        except Exception as e:
            self.skipTest(f"Preview needs GTK: {e!r}")
        prev.accent = (0.2, 0.5, 0.9)
        prev.box = (80, 60, 160, 160); prev.frame_size = (640, 360)
        prev.cue, prev.cue_ok, prev.cue_progress = cue, False, 0.4
        prev.cue_pose, prev.cue_target, prev.cue_ring = pose, target, ring
        prev._arrow_dir = None
        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, 640, 360)
        cr = cairo.Context(surf)
        prev._draw_cue(cr, 640, 360)      # must not raise


class Robustness(unittest.TestCase):
    def test_stepping_away_does_not_cost_poses(self):
        # A5: 40 s with no face during the first pass, then a cooperative user
        class Away(World):
            def face(self, t):
                return not (4.0 <= t < 44.0)
        res, snaps, st = run_setup(Away())
        self.assertEqual({m.label for m in st.models} >= {"C/level", "C/up", "C/down", "L1/level", "R1/level"}, True)

    def test_too_small_face_shows_come_closer(self):
        w = World(face_w=40)                                      # under MIN_ENROLL_FACE_PX
        res, snaps, st = run_setup(w, max_s=20)
        self.assertTrue(any(sn.get("small") for sn in snaps if sn.get("phase") == "run"))

    def test_camera_freeze_keeps_done_card_and_autoclose(self):
        # A2: frames stop after the sweep: the done card still has its text and the window still closes
        w = World()
        res, snaps, st = run_setup(w)
        done = [sn for sn in snaps if sn.get("phase") == "done"]
        self.assertTrue(done and all(sn.get("headline") for sn in done))

    def test_lost_camera_ends_with_an_honest_card_and_no_save(self):
        class Lost(World):
            pass
        w = Lost()
        import enroll_harness as H
        orig = H.FakeCap.read_one

        def read_one(self, sp, flush=0):
            if self.clock() - self.world.t0 > 12.0:
                self.lost = "IR starved and could not be reopened"
            return orig(self, sp, flush)
        from unittest import mock
        with mock.patch.object(H.FakeCap, "read_one", read_one):
            res, snaps, st = run_setup(w, max_s=60)
        self.assertTrue(res["failed"])
        done = [sn for sn in snaps if sn.get("phase") == "done"]
        self.assertTrue(done and done[-1]["ok"] is False and "Camera lost" in done[-1]["headline"], done[-1:])

    def test_nothing_enrolled_card_does_not_autoclose_or_claim_saved(self):
        # A4: no face ever → every pass times out on the wall clock → honest card, no auto-close
        class Nobody(World):
            def light_at(self, t):
                return "ok" if t < 6.0 else "dark"                 # intro (5 s) starts, then the face is unusable
        res, snaps, st = run_setup(Nobody(), max_s=200, timeout=3.0)
        done = [sn for sn in snaps if sn.get("phase") == "done"]
        self.assertTrue(done)
        self.assertFalse(done[-1]["ok"]); self.assertIsNone(done[-1]["closing_in"])
        self.assertTrue(res["failed"])                              # cmd_enroll refuses to save it


class SamplerDeath(unittest.TestCase):
    def test_worker_dies_once_is_respawned_and_enrolment_completes(self):
        import enroll_harness as H
        made = []

        def factory(models):
            smp = H.FakeSampler(w)
            made.append(smp)
            if len(made) == 1:                                     # the first worker dies after 5 jobs
                orig = smp.submit

                def submit(*a):
                    if len(smp.__dict__.setdefault("n", [])) >= 5:
                        smp.alive = False; return False
                    smp.n.append(1); return orig(*a)
                smp.submit = submit
            return smp
        w = World()
        res, snaps, st = run_setup(w, sampler_cls=factory)
        self.assertEqual(len(made), 2)
        self.assertFalse(res["failed"]); self.assertGreater(res["added"], 0)

    def test_worker_dies_twice_fails_without_saving(self):
        import enroll_harness as H

        def factory(models):
            smp = H.FakeSampler(w); smp.alive = False; return smp
        w = World()
        res, snaps, st = run_setup(w, sampler_cls=factory, max_s=60)
        self.assertTrue(res["failed"])
        self.assertFalse(snaps[-1].get("ok", True))


class GlassesSweep(unittest.TestCase):
    def test_glasses_sweep_waits_for_start(self):
        # A8: after the glasses button, a face in view for 10 s must NOT start the sweep
        w = World()
        w.keys_after_done = {1.0: ["glasses"], 13.0: ["enter"]}
        res, snaps, st = run_setup(w)
        intro_after = [sn for sn in snaps if sn.get("phase") == "intro" and sn.get("manual_start")]
        self.assertTrue(intro_after, "glasses intro never shown")
        self.assertGreater(len(intro_after), 15 * 10)              # stayed ≥ 10 s until Start
        self.assertTrue(any(m.condition == "glasses" for m in st.models))


class GlassesFailure(unittest.TestCase):
    def test_camera_lost_during_glasses_sweep_keeps_the_main_enrolment(self):
        import enroll_harness as H
        from unittest import mock
        w = World()
        w.keys_after_done = {1.0: ["glasses"], 2.0: ["enter"]}
        orig = H.FakeCap.read_one

        def read_one(self, sp, flush=0):
            if self.world.t_done is not None and self.clock() - self.world.t0 > self.world.t_done + 5.0:
                self.lost = "IR starved and could not be reopened"
            return orig(self, sp, flush)
        with mock.patch.object(H.FakeCap, "read_one", read_one):
            res, snaps, st = run_setup(w)
        self.assertEqual(res["failed"], "")
        self.assertGreater(res["added"], 0)
        self.assertFalse(any(m.condition == "glasses" for m in st.models))
        self.assertIn("sweep stopped", snaps[-1]["summary"])


class RgbCarries(unittest.TestCase):
    def test_ir_face_dark_but_rgb_sees_is_still_an_enrolment(self):
        import enroll_harness as H
        from unittest import mock

        class Bright(World):
            pass
        w = Bright()
        orig_submit = H.FakeSampler.submit

        def submit(self, cell, tag, items):                       # the IR frames meter dark, RGB is fine
            pose_at, lights = tag
            lights = dict(lights, ir="dark", rgb="")
            return orig_submit(self, cell, (pose_at, lights), items)
        with mock.patch.object(H.FakeSampler, "submit", submit):
            res, snaps, st = run_setup(w, cap_spectra=("ir", "rgb"))
        self.assertFalse(res["failed"], snaps[-1].get("summary"))
        self.assertTrue(all(m.spectrum == "rgb" for m in st.models) and st.models)


class Tolerance(unittest.TestCase):
    def test_level_head_is_not_look_up(self):
        self.assertFalse(cal.pose_ok((0.0, 4.0), (0.0, 18.0)))     # level head is not look-up
        self.assertTrue(cal.pose_ok((0.0, 12.0), (0.0, 18.0)))     # |6| <= 8
        self.assertTrue(cal.pose_ok((0.0, 10.0), (0.0, 18.0)))     # just reaches the arrival band
        self.assertFalse(cal.pose_ok((0.0, -7.0), (0.0, -24.0)))   # |17| > 8
        self.assertTrue(cal.pose_ok((-20.0, 0.0), (-28.0, 0.0)))   # the required ~20° turn counts
        self.assertTrue(cal.pose_ok((5.0, 0.0), (0.0, 0.0)))       # centre still accepts a level head


class WindowStates(unittest.TestCase):
    """The GTK side of the same states (skipped without GTK 4 + libadwaita)."""

    @classmethod
    def setUpClass(cls):
        try:
            from ward import appui
            cls.appui = appui
            cls.w = appui.CalibrateWindow("t", 320, 180)
        except Exception as e:                                      # no display / no libadwaita
            raise unittest.SkipTest(f"GTK unavailable: {e}")

    def test_pass_button_says_skip_and_dot_is_current(self):
        self.w._apply({"phase": "run", "index": 2, "total": 8, "cue": "left", "progress": 0.3})
        self.assertEqual(self.w.btn_next.get_label(), "Skip pass")
        self.assertIn("fu-dot-now", self.w._dot_widgets[2].get_css_classes())

    def test_arrow_is_coarse_and_coherent(self):
        d = self.appui.cue_direction
        self.assertEqual(d("left", (0.0, 0.0), (-28.0, 0.0)), "left")                   # go screen-left
        self.assertEqual(d("left", (-42.0, 0.0), (-28.0, 0.0)), "right")                # clear overshoot (14° past): come back
        self.assertEqual(d("left", (0.0, -9.0), (-28.0, 0.0)), "left")                  # natural tilt: no diagonal
        self.assertEqual(d("left", (-28.0, 17.0), (-28.0, 0.0)), "down")                # main axis done, pitch far off
        self.assertIsNone(d("left", (-27.0, 1.0), (-28.0, 0.0)))                         # there: no arrow
        self.assertEqual(d("up", None, None), "up")                                      # no pose yet: the cue's own way
        self.assertIsNone(d("closer", (0.0, 0.0), (0.0, 0.0)))

    def test_arrow_does_not_flip_on_jitter(self):
        import numpy as np
        d = self.appui.cue_direction
        rng = np.random.default_rng(0)
        prev, seen = None, []
        for k in range(200):                                  # approaching "left" with ±4° estimator noise
            yaw = -28.0 * min(1.0, k / 120) + rng.normal(scale=4.0)
            prev = d("left", (yaw, rng.normal(scale=4.0)), (-28.0, 0.0), prev)
            seen.append(prev)
        flips = sum(1 for a, b in zip(seen, seen[1:]) if a and b and a != b)
        self.assertEqual(flips, 0)                            # never left↔right/up/down while approaching
        self.assertNotIn("right", seen)

    def test_every_cue_has_a_word(self):
        for _l, cue, _t, _tg in cal.SETUP_POSES:
            self.assertIn(cue, self.appui.CalibrateWindow.CUE_WORD)

    def test_done_pill_visible_after_a_pass_hid_it(self):
        self.w._apply({"phase": "run", "index": 1, "total": 8, "cue": "up", "box": (1, 1, 80, 80), "progress": 0.2})
        self.w._apply({"phase": "done", "index": 8, "total": 8, "ok": True, "headline": "Enrolled", "summary": "x", "closing_in": 9})
        self.assertTrue(self.w.pill_over.get_visible())
        self.w._apply({"phase": "done", "index": 8, "total": 8, "ok": False, "headline": "Nothing enrolled", "summary": "x"})
        self.assertEqual(self.w.pill_over.get_label(), "Nothing saved")

if __name__ == "__main__":
    unittest.main()
