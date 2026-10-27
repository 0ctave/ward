"""Threshold calibration (DESIGN.md §6.7): derivation, guardrails, store/config application, rolling refresh."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("WARD_ETC", tempfile.mkdtemp(prefix="ward-test-"))
os.environ.setdefault("WARD_LOG", os.path.join(os.environ["WARD_ETC"], "log"))

from ward.engine import calibration as cal, config, store


def _store(dim=128, per=6):
    rng = np.random.default_rng(3)
    st = store.Store(user="cal", path=store.path_for("cal"))
    for label, sp in (("C/level", "ir"), ("C/up", "ir"), ("C/level", "rgb"), ("C/up", "rgb")):
        base = rng.normal(size=dim); base /= np.linalg.norm(base)
        samples = []
        for _ in range(per):
            v = base + rng.normal(scale=0.03, size=dim); samples.append(v / np.linalg.norm(v))
        st.add(label, sp, base, 0.9, None, samples=samples)
    for sp in ("ir", "rgb"):
        st.fit_whitening(sp)
    return st


def _measures(n=30, raw_center=0.25, spread=0.05, w=8.0, light=("ok",), spectra=("ir", "rgb"), fused=None):
    rng = np.random.default_rng(5)
    out = []
    for i in range(n):
        for sp in spectra:
            out.append(cal.Measure(cal.PROMPTS[i % len(cal.PROMPTS)][0], sp, float(abs(rng.normal(raw_center, spread))),
                                   float(abs(rng.normal(w, 1.5))), light[i % len(light)], fused, "C/level"))
    return out


class Derivation(unittest.TestCase):
    def test_gate_is_p95_plus_margin_within_bounds(self):
        g, note = cal.derive_gate([0.20] * 10 + [0.30] * 10)          # p95 = 0.30
        self.assertAlmostEqual(g, 0.30 + cal.MARGIN, places=2); self.assertEqual(note, "")
        g, note = cal.derive_gate([0.10] * 20)
        self.assertEqual(g, cal.GATE_MIN)
        g, note = cal.derive_gate([0.55] * 20)
        self.assertEqual(g, cal.GATE_MAX); self.assertIn("capped", note)      # never loosened beyond the cap

    def test_whitening_threshold_needs_samples(self):
        self.assertIsNone(cal.derive_whitening_threshold([5.0] * 3))
        self.assertAlmostEqual(cal.derive_whitening_threshold([5.0] * 10), 5.0 * cal.WHITEN_MARGIN, places=3)

    def test_derived_config_keys(self):
        d = cal.derived_config(0.40, 0.38)
        self.assertEqual(d["recognition"]["certainty"], 0.40); self.assertEqual(d["recognition"]["rgb_certainty"], 0.38)
        self.assertEqual(d["recognition"]["hard_cap"], 0.55); self.assertEqual(d["recognition"]["early_reject"], 0.65)
        self.assertEqual(d["behaviour"]["learn_max_dist"], 0.30)
        self.assertEqual(cal.derived_config(None, None), {})

    def test_summary_warnings(self):
        res = cal.summarise(_measures(n=30, light=("ok",)))
        self.assertTrue(res.ok()); self.assertIn("ir", res.gate)
        self.assertTrue(any("only one light condition" in w for w in res.warnings))
        res = cal.summarise(_measures(n=30, light=("ok", "dim")))
        self.assertFalse(any("only one light condition" in w for w in res.warnings))
        res = cal.summarise(_measures(n=3))
        self.assertFalse(res.ok()); self.assertTrue(any("not enough" in w for w in res.warnings))


class Collect(unittest.TestCase):
    def test_collect_aborts_when_nobody_is_there_and_measures_when_someone_is(self):
        class FakeSample:
            def __init__(self, raw):
                self.raw = raw; self.wdist = {k: 6.0 for k in raw}; self.light = {k: "ok" for k in raw}; self.label = {k: "C/level" for k in raw}
        class Ev:
            def __init__(self, face): self.face = face; self.calls = 0; self.last_fused = {"ir": np.ones(3)}
            def step(self):
                self.calls += 1
                return FakeSample({"ir": 0.2} if self.face else {})
        seen = []
        out = cal.collect(Ev(False), seconds=0.05, on_prompt=lambda k, t: seen.append(k))
        self.assertEqual(out, []); self.assertEqual(len(seen), 2)                     # aborted after two faceless prompts
        seen.clear()
        out = cal.collect(Ev(True), seconds=0.05, on_prompt=lambda k, t: seen.append(k))
        self.assertEqual(len(seen), len(cal.PROMPTS)); self.assertTrue(out)
        self.assertEqual({m.spectrum for m in out}, {"ir"}); self.assertTrue(all(m.fused is not None for m in out))


class Passes(unittest.TestCase):
    """Pose-conditioned passes: display-space conditions, width baseline, skip, timeout."""

    def _smp(self, yaw=None, pitch=None, roll=None, face_w=None, raw=0.25):
        class S:
            pass
        s = S(); s.raw = {"ir": raw}; s.wdist = {"ir": 6.0}; s.light = {"ir": "ok"}; s.label = {"ir": "C/level"}
        s.yaw, s.pitch, s.roll, s.face_w = yaw, pitch, roll, face_w
        return s

    def test_satisfied(self):
        self.assertTrue(cal.satisfied({"yaw": (-10, 10)}, 3, 0, 0, None, None))
        self.assertFalse(cal.satisfied({"yaw": (9, 90)}, 3, 0, 0, None, None))
        self.assertIsNone(cal.satisfied({"yaw": (9, 90)}, None, 0, 0, None, None))         # pose unknown
        self.assertTrue(cal.satisfied({"roll_abs": (7, 90)}, 0, 0, -12, None, None))
        self.assertTrue(cal.satisfied({"width": (1.12, 9)}, 0, 0, 0, 130, 100)); self.assertFalse(cal.satisfied({"width": (1.12, 9)}, 0, 0, 0, 105, 100))
        self.assertIsNone(cal.satisfied({"width": (1.12, 9)}, 0, 0, 0, 105, None))
        self.assertTrue(cal.satisfied(None, None, None, None, None, None))

    def test_collector_walks_the_passes_on_movement(self):
        col = cal.Collector(need=3, timeout=100, sgn=-1.0)
        # pass 1 "screen": centred, sets the width baseline
        for _ in range(3):
            st = col.feed(self._smp(yaw=2, pitch=1, roll=0, face_w=100))
        self.assertTrue(st["pass_done"]); self.assertEqual(col.current[0], "keyboard"); self.assertEqual(col.baseline_w, 100)
        # "keyboard": looking straight does not count, looking down does
        st = col.feed(self._smp(yaw=0, pitch=0, roll=0, face_w=100)); self.assertFalse(st["ok"]); self.assertEqual(st["hits"], 0)
        for _ in range(3):
            st = col.feed(self._smp(yaw=0, pitch=-15, roll=0, face_w=100))
        self.assertEqual(col.current[0], "back")
        for _ in range(3): col.feed(self._smp(0, 0, 0, face_w=70))          # 0.7 × baseline
        self.assertEqual(col.current[0], "close")
        for _ in range(3): col.feed(self._smp(0, 0, 0, face_w=130))
        self.assertEqual(col.current[0], "tilt")
        for _ in range(3): col.feed(self._smp(0, 0, 12, face_w=100))
        self.assertEqual(col.current[0], "left")
        # camera yaw +12 → display yaw −12 (mirrored): that is the user's left
        st = col.feed(self._smp(yaw=-12, pitch=0, roll=0, face_w=100)); self.assertFalse(st["ok"])
        for _ in range(3): col.feed(self._smp(yaw=12, pitch=0, roll=0, face_w=100))
        self.assertEqual(col.current[0], "right")
        col.skip(); self.assertEqual(col.current[0], "light")
        for _ in range(3): st = col.feed(self._smp(yaw=None, face_w=100))  # light: any pose, unknown pose counts
        self.assertTrue(st["finished"]); self.assertTrue(col.finished)
        self.assertGreater(len(col.measures), 15)
        self.assertEqual(col.feed(self._smp()), {"finished": True})

    def test_timeout_moves_on_and_flags_it(self):
        col = cal.Collector(need=50, timeout=0.0)
        st = col.feed(self._smp(yaw=0, pitch=0, roll=0, face_w=100))
        self.assertTrue(st["pass_done"]); self.assertTrue(st["timed_out"]); self.assertEqual(col.i, 1)

    def test_no_face_does_not_count(self):
        col = cal.Collector(need=2, timeout=100)
        class Empty:
            raw = {}; wdist = {}; light = {}; label = {}; yaw = None
        st = col.feed(Empty())
        self.assertEqual(st["hits"], 0); self.assertIsNone(st["ok"]); self.assertEqual(col.measures, [])


class Apply(unittest.TestCase):
    def test_apply_writes_config_store_and_evidence(self):
        st = _store()
        cfg = config.load(); cfg["recognition"]["certainty"] = 0.517
        fused = st.models[0].embedding
        ms = _measures(n=30, raw_center=0.25, spread=0.04, light=("ok", "dim"), fused=fused)
        res = cal.summarise(ms)
        changed = cal.apply(st, cfg, res, ms, save=True)
        self.assertIn("recognition.certainty", changed)
        self.assertLessEqual(cfg["recognition"]["certainty"], cal.GATE_MAX); self.assertGreaterEqual(cfg["recognition"]["certainty"], cal.GATE_MIN)
        self.assertEqual(cfg["recognition"]["hard_cap"], round(min(cfg["recognition"]["certainty"] + 0.15, 0.55), 3))
        for sp in ("ir", "rgb"):
            self.assertAlmostEqual(st.whitening[sp]["threshold"], res.w_threshold[sp], places=3)
            self.assertEqual(st.whitening[sp]["floor"], res.w_threshold[sp])
        ev = st.calib["calibration"]
        self.assertEqual(ev["gate"], res.gate); self.assertIn("date", ev); self.assertEqual(ev["conditions"]["ir"], {"ok": 15, "dim": 15})
        back = store.load("cal")                                      # evidence survives a round trip
        self.assertEqual(back.calib["calibration"]["gate"], res.gate)


class Rolling(unittest.TestCase):
    def test_ring_and_guarded_refresh(self):
        st = _store()
        for sp in ("ir", "rgb"):
            st.whitening[sp]["threshold"] = 5.0; st.whitening[sp]["floor"] = 5.0
        refreshed = 0
        for i in range(cal.REFRESH_EVERY * 2):
            # real-world whitened distances are far above 5.0: the threshold may follow, 20 % per refresh
            refreshed += cal.record_accept(st, {"ir": 0.22, "rgb": 0.30}, {"ir": (12.0, 5.0), "rgb": (11.0, 5.0)}, {"ir": "ok", "rgb": "ok"})
        self.assertEqual(refreshed, 2)
        self.assertAlmostEqual(st.whitening["ir"]["threshold"], 5.0 * 1.2 * 1.2, places=3)     # bounded steps
        self.assertLessEqual(len(st.calib["recent"]["ir"]), cal.RECENT_MAX)
        # a window of small distances can lower it, but never below the session floor
        st.calib["recent"]["ir"] = [[0, 0.2, 1.0, "ok"]] * 30
        cal.refresh(st)
        self.assertGreaterEqual(st.whitening["ir"]["threshold"], 5.0)

    def test_gate_advice(self):
        st = _store(); cfg = config.load(); cfg["recognition"]["certainty"] = 0.517
        st.calib["recent"] = {"ir": [[0, 0.22, 10.0, "ok"]] * 20}
        self.assertIn("looser than needed", cal.gate_advice(st, cfg))
        cfg["recognition"]["certainty"] = 0.22                                # p95 + margin = 0.27 > gate + 0.03
        self.assertIn("close to the gate", cal.gate_advice(st, cfg))
        cfg["recognition"]["certainty"] = 0.30; st.calib["recent"] = {"ir": [[0, 0.22, 10.0, "ok"]] * 3}
        self.assertIsNone(cal.gate_advice(st, cfg))


if __name__ == "__main__":
    unittest.main()


class BuildFromCaptures(unittest.TestCase):
    """Fused enroll+calibrate: models from the best frames, gate from the held-out rest."""

    def test_builds_models_and_holds_out_for_the_gate(self):
        from ward.engine.detect import fuse
        rng = np.random.default_rng(7)
        # two poses, each with 15 captured frames per spectrum (quality, emb, aug, light)
        poses = [("C/level", "centre", "", (0.0, 0.0)), ("L1/level", "left", "", (-20.0, 0.0))]
        caps = {0: {}, 1: {}}
        centres = {}
        for pi, (label, *_ ) in enumerate(poses):
            for sp in ("ir", "rgb"):
                base = rng.normal(size=128); base /= np.linalg.norm(base)
                centres[(pi, sp)] = base
                rows = []
                for j in range(15):
                    v = base + rng.normal(scale=0.03, size=128); v /= np.linalg.norm(v)
                    rows.append((0.5 + 0.5 * rng.random(), v, [base], "ok"))
                caps[pi][sp] = rows
        st = store.Store(user="fused", path=store.path_for("fused"))
        added, dist_by, extra = cal.build_from_captures(st, caps, poses, 8, fuse)
        self.assertEqual(added, 4)                                     # 2 poses × 2 spectra
        self.assertEqual({m.label for m in st.models}, {"C/level", "L1/level"})
        # the first 2/3 in time (10) are candidates; the worst-quality third is dropped → 7 build the model
        self.assertTrue(all(len(m.samples) == 7 for m in st.models))
        # the LAST third in time (5) is held out per (pose, spectrum) → 10 per spectrum
        self.assertEqual(len(dist_by["ir"]), 10); self.assertEqual(len(dist_by["rgb"]), 10)
        # held-out frames are close to their own pose's model (fresh-frame, not self-match-zero, not huge)
        for sp in ("ir", "rgb"):
            ds = [d for d, _w, _l in dist_by[sp]]
            # a held-out frame is close to its OWN pose model (same-person spread), far below the
            # cross-pose distance (~1.4 to the unrelated other pose) — so min_distance found the right model
            self.assertLess(np.median(ds), 0.5); self.assertGreater(np.median(ds), 0.05)
        res = cal.derive_from_holdout(dist_by)
        self.assertIn("ir", res.gate)

    def test_folds_numbered_pass_into_base_label(self):
        from ward.engine.detect import fuse
        rng = np.random.default_rng(1)
        poses = [("C/level", "centre", "", (0.0, 0.0)), ("C/level2", "light", "", (0.0, 0.0))]
        caps = {}
        for pi in (0, 1):
            base = rng.normal(size=128); base /= np.linalg.norm(base)
            caps[pi] = {"ir": [(0.9, (base + rng.normal(scale=0.02, size=128)) / np.linalg.norm(base), [], "ok") for _ in range(6)]}
        st = store.Store(user="fold", path=store.path_for("fold"))
        added, _dist, _extra = cal.build_from_captures(st, caps, poses, 8, fuse)
        self.assertEqual({m.label for m in st.models}, {"C/level"})    # the "…2" pass folded in, no C/level2


class SelectRows(unittest.TestCase):
    """Review A1/A2: pose-true frames only, temporal holdout, nothing from a blind camera."""

    def row(self, q, pose=None, light="ok", tag=0):
        return (q, np.full(4, float(tag)), [], light, pose)

    def test_transition_frames_are_not_enrolled(self):
        # turning toward the left target (-20): the first frames are at -2..-6 (still turning)
        rows = [self.row(0.99, (-3.0, 0.0), tag=1)] * 4 + [self.row(0.5, (-19.0, 1.0), tag=2)] * 9
        model, held = cal.select_rows(rows, (-20.0, 0.0), "left", 8)
        self.assertTrue(all(r[1][0] == 2 for r in model + held))      # the sharp transition frames are gone

    def test_model_frames_are_spread_over_the_pass(self):
        # quality rises over time (sharpest at the end): the picks must still span the window
        rows = [self.row(0.5 + 0.02 * i, (0.0, 0.0), tag=i) for i in range(18)]
        model, held = cal.select_rows(rows, (0.0, 0.0), "centre", 6)
        tags = [int(r[1][0]) for r in model]
        self.assertEqual(len(tags), 6)
        self.assertGreaterEqual(max(tags) - min(tags), 6)                    # ≥ half of the 12 candidates

    def test_pose_never_reached_gives_no_model(self):
        rows = [self.row(0.9, (-5.0, 0.0))] * 20
        self.assertIsNone(cal.select_rows(rows, (-20.0, 0.0), "left", 8))

    def test_holdout_is_the_last_third_in_time(self):
        rows = [self.row(0.1 * (i % 3), (0.0, 0.0), tag=i) for i in range(12)]
        model, held = cal.select_rows(rows, (0.0, 0.0), "centre", 8)
        self.assertEqual([int(r[1][0]) for r in held], [8, 9, 10, 11])
        self.assertTrue(all(r[1][0] < 8 for r in model))

    def test_distance_cues_ignore_pose_and_unknown_pose_is_kept(self):
        rows = [self.row(0.9, (10.0, 9.0))] * 6                       # "lean in" at any head angle
        self.assertIsNotNone(cal.select_rows(rows, (0.0, 0.0), "closer", 8))
        rows = [self.row(0.9, None)] * 6                               # camera without a pose estimate
        self.assertIsNotNone(cal.select_rows(rows, (-20.0, 0.0), "left", 8))

    def test_dark_frames_enrol_nothing(self):
        rows = [self.row(0.9, (0.0, 0.0), light="dark")] * 10         # RGB blind in a dim room
        self.assertIsNone(cal.select_rows(rows, (0.0, 0.0), "centre", 8))


class ReenrolKeepsSkippedPoses(unittest.TestCase):
    def test_a_pose_without_new_samples_keeps_its_old_model(self):
        from ward.engine.detect import fuse
        rng = np.random.default_rng(5)
        st = store.Store(user="keep", path=store.path_for("keep"))
        old = rng.normal(size=128); old /= np.linalg.norm(old)
        st.add("L1/level", "ir", old, 0.9, (-20.0, 0.0), samples=[old] * 3)
        poses = [("C/level", "centre", "", (0.0, 0.0)), ("L1/level", "left", "", (-20.0, 0.0))]
        base = rng.normal(size=128); base /= np.linalg.norm(base)
        caps = {0: {"ir": [(0.9, base, [], "ok", (0.0, 0.0)) for _ in range(9)]},
                1: {"ir": [(0.9, base, [], "ok", (-3.0, 0.0)) for _ in range(9)]}}   # left never reached
        cal.build_from_captures(st, caps, poses, 8, fuse)
        self.assertEqual({m.label for m in st.models}, {"C/level", "L1/level"})
        self.assertTrue(any(m.label == "L1/level" and np.allclose(m.embedding, old) for m in st.models))


class OtherFaces(unittest.TestCase):
    def test_a_second_person_is_not_enrolled(self):
        from ward.engine.detect import fuse
        rng = np.random.default_rng(9)
        me = rng.normal(size=128); me /= np.linalg.norm(me)
        other = rng.normal(size=128); other /= np.linalg.norm(other)
        jit = lambda b: (lambda v: v / np.linalg.norm(v))(b + rng.normal(scale=0.012, size=128))
        poses = [("C/level", "centre", "", (0.0, 0.0)), ("L1/level", "left", "", (-20.0, 0.0))]
        caps = {0: {"ir": [(0.9, jit(me), [], "ok", (0.0, 0.0)) for _ in range(12)]},
                1: {"ir": [(0.9, jit(me), [], "ok", (-20.0, 0.0)) for _ in range(6)]
                    + [(0.99, jit(other), [], "ok", (-20.0, 0.0)) for _ in range(6)]}}   # someone leans in
        st = store.Store(user="two", path=store.path_for("two"))
        cal.build_from_captures(st, caps, poses, 6, fuse)
        left = next(m for m in st.models if m.label == "L1/level")
        self.assertLess(float(np.linalg.norm(left.embedding - me)), 0.2)
        self.assertGreater(float(np.linalg.norm(left.embedding - other)), 0.6)


class SetupSummary(unittest.TestCase):
    def test_summary_names_missed_poses_and_the_dark(self):
        from ward import gui
        res = cal.derive_from_holdout({})
        txt = gui._setup_summary(res, 7, ["Look up"], ir=True)
        self.assertIn("No model for: Look up", txt)
        self.assertIn("dark", txt)
        self.assertNotIn("dark", gui._setup_summary(res, 3, [], ir=True, condition="glasses"))


class Holdout(unittest.TestCase):
    """Unified setup: the gate comes from HELD-OUT frames (roadmap: fused enroll+calibrate)."""

    def test_derive_from_holdout(self):
        rng = np.random.default_rng(2)
        # genuine held-out distances: user frames cluster ~0.25 raw, whitened ~8
        rows_ir = [(float(abs(rng.normal(0.25, 0.05))), float(abs(rng.normal(8, 1.5))), "ok") for _ in range(30)]
        rows_ir += [(0.22, 7.0, "dim") for _ in range(6)]
        res = cal.derive_from_holdout({"ir": rows_ir})
        self.assertIn("ir", res.gate)
        self.assertGreaterEqual(res.gate["ir"], cal.GATE_MIN); self.assertLessEqual(res.gate["ir"], cal.GATE_MAX)
        self.assertIn("ir", res.w_threshold)
        self.assertEqual(res.conditions["ir"], {"ok": 30, "dim": 6})
        # too few frames → warned, no gate
        res2 = cal.derive_from_holdout({"ir": rows_ir[:4]})
        self.assertNotIn("ir", res2.gate); self.assertTrue(any("not enough" in w for w in res2.warnings))

    def test_need_samples_floor(self):
        # ring pools the gate over more poses so each holds fewer; neither style drops below the model-4 floor
        self.assertLessEqual(cal.RING_NEED_SAMPLES, cal.NEED_SAMPLES)
        self.assertGreaterEqual(cal.RING_NEED_SAMPLES, 6)

    def test_setup_poses_cover_the_grid_and_target_signs(self):
        keys = [p[1] for p in cal.SETUP_POSES]
        self.assertEqual(keys[:5], ["centre", "up", "down", "left", "right"])
        # left target has NEGATIVE display yaw (user's left = lower yaw, matches enrollment hint), right positive
        left = next(p for p in cal.SETUP_POSES if p[1] == "left")
        right = next(p for p in cal.SETUP_POSES if p[1] == "right")
        self.assertLess(left[3][0], 0); self.assertGreater(right[3][0], 0)
        up = next(p for p in cal.SETUP_POSES if p[1] == "up")
        down = next(p for p in cal.SETUP_POSES if p[1] == "down")
        self.assertGreater(up[3][1], 0); self.assertLess(down[3][1], 0)      # up = higher pitch
