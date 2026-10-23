"""Unit tests that need no camera and no face: run with  python3 -m unittest discover tests"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

os.environ.setdefault("FACEUP_ETC", tempfile.mkdtemp(prefix="faceup-test-"))
os.environ.setdefault("FACEUP_LOG", os.path.join(os.environ["FACEUP_ETC"], "log"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from faceup.engine import config, pam, quality, store  # noqa: E402
from faceup.engine.detect import Box  # noqa: E402
from faceup.engine.liveness import Result, cross_spectrum, motion, shadow  # noqa: E402


def textured(h=360, w=640, seed=1):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w), dtype=np.uint8)
    return cv2.GaussianBlur(img, (0, 0), 1.2)


class ConfigTests(unittest.TestCase):
    def test_set_key_types_and_validation(self):
        cfg = config.load()
        self.assertEqual(config.set_key(cfg, "recognition.certainty", "0.5"), 0.5)
        self.assertEqual(config.set_key(cfg, "liveness.motion", "false"), False)
        self.assertEqual(config.set_key(cfg, "timing.window_ms", "900"), 900)
        with self.assertRaises(ValueError):
            config.set_key(cfg, "liveness.bogus", "1")
        with self.assertRaises(ValueError):
            config.set_key(cfg, "timing.window_ms", "fast")

    def test_roundtrip(self):
        cfg = config.load()
        cfg["cameras"]["ir"] = "/dev/video9"
        config.save(cfg)
        self.assertEqual(config.load()["cameras"]["ir"], "/dev/video9")


class StoreTests(unittest.TestCase):
    def test_roundtrip_and_distance(self):
        st = store.load("tester")
        e1 = np.ones(128) / np.sqrt(128)
        e2 = -e1
        st.add("straight", "ir", e1, 0.9)
        st.add("straight", "rgb", e2, 0.8)
        st.calib = {"dx": 0.01, "dy": 0.0, "s": 1.05}
        st.save()
        self.assertEqual(oct(os.stat(st.path).st_mode & 0o777), "0o600")
        st2 = store.load("tester")
        self.assertEqual({m.spectrum for m in st2.models}, {"ir", "rgb"})
        d, label = st2.min_distance(e1, "ir")
        self.assertAlmostEqual(d, 0.0, places=5)
        self.assertEqual(label, "straight")
        d2, _ = st2.min_distance(e1, "rgb")
        self.assertAlmostEqual(d2, 2.0, places=5)
        self.assertEqual(st2.calib["s"], 1.05)

    def test_refuses_bad_user(self):
        with self.assertRaises(store.StoreError):
            store.load("../etc")


class QualityTests(unittest.TestCase):
    def test_dark_and_blur(self):
        dark = np.full((360, 640), 10, np.uint8)
        self.assertEqual(quality.measure(dark).reason(25), "dark")
        flat = np.full((360, 640), 120, np.uint8)
        self.assertEqual(quality.measure(flat).reason(25), "blurred")
        self.assertIsNone(quality.measure(textured()).reason(25))


class CrossSpectrumTests(unittest.TestCase):
    calib = {"dx": 0.0, "dy": 0.0, "s": 1.0}
    size = (640, 360)

    def test_screen_replay_fails(self):
        v = cross_spectrum.vote(None, Box(100, 80, 150, 150), False, False, self.calib, self.size, self.size)
        self.assertEqual(v.result, Result.FAIL)

    def test_ir_print_fails(self):
        v = cross_spectrum.vote(Box(100, 80, 150, 150), None, False, False, self.calib, self.size, self.size)
        self.assertEqual(v.result, Result.FAIL)

    def test_night_abstains(self):
        v = cross_spectrum.vote(Box(100, 80, 150, 150), None, False, True, self.calib, self.size, self.size)
        self.assertEqual(v.result, Result.ABSTAIN)

    def test_aligned_passes_and_calibration_applies(self):
        v = cross_spectrum.vote(Box(100, 80, 150, 150), Box(104, 78, 150, 150), False, False, self.calib, self.size, self.size)
        self.assertEqual(v.result, Result.PASS)
        # RGB camera at 1280x720: same normalised place must still pass
        v = cross_spectrum.vote(Box(100, 80, 150, 150), Box(200, 160, 300, 300), False, False, self.calib, self.size, (1280, 720))
        self.assertEqual(v.result, Result.PASS)
        # far apart → fail
        v = cross_spectrum.vote(Box(100, 80, 150, 150), Box(400, 80, 150, 150), False, False, self.calib, self.size, self.size)
        self.assertEqual(v.result, Result.FAIL)


class MotionTests(unittest.TestCase):
    box = Box(200, 80, 220, 220)

    def test_still_abstains(self):
        g = textured()
        self.assertEqual(motion.vote(g, g.copy(), self.box).result, Result.ABSTAIN)

    def test_planar_motion_fails(self):
        g0 = textured()
        H = np.array([[1.01, 0.004, 8.0], [-0.003, 1.008, 6.0], [0.00001, 0.00002, 1.0]])
        g1 = cv2.warpPerspective(g0, H, (g0.shape[1], g0.shape[0]))
        v = motion.vote(g0, g1, self.box)
        self.assertEqual(v.result, Result.FAIL, v.reason)

    def test_small_planar_motion_abstains(self):
        """A live head that shifts 2–3 px also fits a plane (sub-pixel parallax): never a FAIL."""
        g0 = textured()
        H = np.array([[1.0, 0.0, 2.5], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
        g1 = cv2.warpPerspective(g0, H, (g0.shape[1], g0.shape[0]))
        v = motion.vote(g0, g1, self.box)
        self.assertEqual(v.result, Result.ABSTAIN, v.reason); self.assertIn("small motion", v.reason)

    def test_non_rigid_motion_passes(self):
        g0 = textured()
        h, w = g0.shape
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        # a bulge moving differently from its surroundings (nose-vs-ears parallax caricature)
        cx, cy = 310, 190
        r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        bump = np.exp(-(r / 60.0) ** 2)
        # thresholds are fractions of the face width: a 220 px face (vs the 140 px reference) needs
        # a proportionally larger nose-vs-ears parallax to count as non-rigid
        k = self.box.w / 140.0
        map_x = xs - 4.0 * k * bump - 2.0 * k
        map_y = ys + 3.0 * k * bump - 1.5 * k
        g1 = cv2.remap(g0, map_x, map_y, cv2.INTER_LINEAR)
        v = motion.vote(g0, g1, self.box)
        self.assertEqual(v.result, Result.PASS, v.reason)


class ShadowTests(unittest.TestCase):
    box = Box(200, 80, 200, 200)

    def test_uniform_change_fails_and_gradient_passes(self):
        off = np.full((360, 640), 60, np.uint8)
        on_uniform = np.full((360, 640), 100, np.uint8)
        self.assertEqual(shadow.vote(off, on_uniform, self.box).result, Result.FAIL)
        ys = np.mgrid[0:360, 0:640][0]
        on_grad = np.clip(60 + (ys - 80) * 0.6, 0, 255).astype(np.uint8)   # strong top→bottom falloff
        self.assertEqual(shadow.vote(off, on_grad, self.box).result, Result.PASS)
        self.assertEqual(shadow.vote(off, off, self.box).result, Result.ABSTAIN)


class PamTests(unittest.TestCase):
    SAMPLE = ("# comment\nauth\t[success=3 default=ignore]\tpam_howdy.so\n"
              "auth\t[success=2 default=ignore]\tpam_unix.so nullok try_first_pass\n"
              "auth\trequisite\tpam_deny.so\n")

    def test_insert_before_first_real_module_and_remove(self):
        out = pam.render_enabled(self.SAMPLE, native=False)    # pam_exec flavour, whatever this machine has installed
        lines = out.splitlines()
        self.assertIn("faceup-auth", lines[1])
        self.assertTrue(lines[2].endswith("pam_howdy.so"))
        self.assertEqual(pam.render_enabled(out, native=False), out)         # idempotent
        self.assertIn("seteuid", lines[1])                         # runs as root, not as the sudo caller
        self.assertEqual(pam.render_disabled(out), self.SAMPLE)
        old = self.SAMPLE.replace("auth\t[success=3", "auth\tsufficient\t\t\tpam_exec.so quiet /usr/local/bin/faceup-auth\t# faceup\nauth\t[success=3", 1)
        upgraded = pam.render_enabled(old, native=False)
        self.assertEqual(upgraded.count("faceup-auth"), 1)
        self.assertIn("seteuid", upgraded)
        self.assertIn("stdout", upgraded)                          # PAM_TEXT_INFO feedback for the greeters
        # a line with seteuid but without stdout (pre-feedback install) is upgraded too
        older = out.replace("quiet stdout seteuid", "quiet seteuid")
        self.assertEqual(pam.render_enabled(older, native=False), out)

    def test_native_module_line_jumps_to_pam_permit(self):
        """pam_faceup.so uses [success=N default=ignore] like pam_unix, so the caller's
        pam_gnome_keyring line after @include still runs and receives the token."""
        ubuntu = ("auth\t[success=2 default=ignore]\tpam_unix.so nullok try_first_pass\n"
                  "auth\t[success=1 default=ignore]\tpam_sss.so use_first_pass\n"
                  "auth\trequisite\t\t\tpam_deny.so\n"
                  "auth\trequired\t\t\tpam_permit.so\n"
                  "auth\toptional\t\t\tpam_cap.so\n")
        out = pam.render_enabled(ubuntu, native=True)
        first = out.splitlines()[0]
        self.assertIn("[success=3 default=ignore]", first); self.assertIn("pam_faceup.so", first)   # unix, sss, deny → permit
        self.assertEqual(pam.render_enabled(out, native=True), out)                                # idempotent
        # switching pam_exec ↔ native rewrites the line in place, disable removes either
        exec_line = pam.render_enabled(ubuntu, native=False)
        self.assertEqual(pam.render_enabled(exec_line, native=True), out)
        self.assertEqual(pam.render_enabled(out, native=False), exec_line)
        self.assertEqual(pam.render_disabled(out), ubuntu)
        self.assertEqual(pam.render_enabled(self.SAMPLE, native=True).count("pam_faceup.so"), 1)

    def test_status_ignores_commented_howdy(self):
        import os, tempfile
        f = tempfile.NamedTemporaryFile("w", delete=False, suffix=".pam"); f.write("#auth\t[success=3 default=ignore]\tpam_howdy.so\nauth\trequired\tpam_unix.so\n"); f.close()
        old = pam.COMMON_AUTH; pam.COMMON_AUTH = pam.Path(f.name)
        try:
            self.assertFalse(pam.status()["howdy"]); self.assertFalse(pam.status()["enabled"])
        finally:
            pam.COMMON_AUTH = old; os.unlink(f.name)


class PamPortability(unittest.TestCase):
    """Fedora/Arch system-auth stacks, openSUSE symlinked common-auth, authselect-managed files."""
    FEDORA = ("auth        required      pam_env.so\n"
              "auth        required      pam_faildelay.so delay=2000000\n"
              "auth        sufficient    pam_unix.so nullok\n"
              "auth        requisite     pam_succeed_if.so uid >= 1000 quiet_success\n"
              "auth        sufficient    pam_sss.so forward_pass\n"
              "auth        required      pam_deny.so\n")

    def setUp(self):
        import tempfile
        self.dir = pam.Path(tempfile.mkdtemp(prefix="faceup-pam-"))
        self.saved = (pam.COMMON_AUTH, pam.BACKUP)

    def tearDown(self):
        pam.COMMON_AUTH, pam.BACKUP = self.saved

    def use(self, path):
        pam.COMMON_AUTH = path; pam.BACKUP = path.with_name(path.name + ".faceup-bak")

    def test_fedora_system_auth_native_jump_lands_after_deny(self):
        f = self.dir / "system-auth"; f.write_text(self.FEDORA); self.use(f)
        pam.enable(native=True)
        lines = f.read_text().splitlines()
        i = next(k for k, ln in enumerate(lines) if "pam_faceup.so" in ln)
        self.assertIn("pam_unix", lines[i + 1])                                # before the first real module
        self.assertIn("[success=4 default=ignore]", lines[i])                  # unix, succeed_if, sss, deny
        pam.disable()
        self.assertEqual(f.read_text(), self.FEDORA)                           # byte-for-byte restore

    def test_symlinked_stack_stays_a_symlink(self):
        import os
        real = self.dir / "common-auth-pc"; real.write_text("auth\trequired\tpam_unix.so\n")
        link = self.dir / "common-auth"; os.symlink(real, link); self.use(link)
        with unittest.mock.patch.object(pam, "manager", return_value="pam-config"):
            pam.enable(native=False)
        self.assertTrue(link.is_symlink())
        self.assertIn("faceup-auth", real.read_text())

    def test_authselect_managed_file_is_refused_untouched(self):
        import os
        (self.dir / "authselect").mkdir()
        real = self.dir / "authselect" / "system-auth"; real.write_text(self.FEDORA)
        link = self.dir / "system-auth"; os.symlink(real, link); self.use(link)
        with self.assertRaises(pam.PamError):
            pam.enable(native=True)
        self.assertEqual(real.read_text(), self.FEDORA)
        self.assertFalse((self.dir / "system-auth.faceup-bak").exists())

    def test_missing_stack_is_a_clear_error(self):
        self.use(self.dir / "nope")
        with self.assertRaises(pam.PamError):
            pam.enable()


class DetectorPortability(unittest.TestCase):
    def test_yunet_failure_falls_back_to_hog(self):
        import cv2
        from faceup.engine import detect
        if detect.dlib is None:
            self.skipTest("dlib not installed")
        m = detect.Models("yunet", 320)
        with unittest.mock.patch.object(m, "_yunet_detect", side_effect=cv2.error("dnn: unsupported layer")):
            m.detect(np.zeros((240, 320), np.uint8))
        self.assertEqual(m.detector_kind, "hog")
        self.assertIn("HOG", m.fallback_reason)

    def test_dlib_dir_override_is_searched_first(self):
        import tempfile
        from faceup.engine import detect
        d = tempfile.mkdtemp(prefix="faceup-dlib-")
        open(os.path.join(d, "shape_predictor_5_face_landmarks.dat"), "w").close()
        with unittest.mock.patch.dict(os.environ, {"FACEUP_DLIB_DIR": d}):
            self.assertEqual(str(detect.model_path("shape_predictor_5_face_landmarks.dat").parent), d)


class FastPath(unittest.TestCase):
    """Roadmap R1/R2/R3/R7/L3: what can be checked without a camera."""

    def test_async_open_reports_missing_device_without_blocking_the_caller(self):
        from faceup.engine.capture import Capture, CaptureError
        pending = Capture.open_async({"ir": "/dev/video-nope", "rgb": ""})
        with self.assertRaises(CaptureError):
            pending.result(timeout=5)
        with self.assertRaises(CaptureError):
            Capture.open_async({"ir": "", "rgb": ""}).result(timeout=5)

    def test_cancel_releases_devices_opened_after_it(self):
        # R2/R3: close()/a failed attempt must never leave a late-opening camera held by nobody
        import threading as _th
        import time as _t
        from faceup.engine import capture as capmod
        released = []

        class FakeCap:
            def __init__(self, s): self.s = s
            def release(self): released.append(self.s)
        go = _th.Event()

        def slow_open(spectrum, path, *a, **k):
            go.wait(2); return FakeCap(spectrum)
        mock = unittest.mock
        with mock.patch.object(capmod, "_open_one", slow_open), mock.patch.object(capmod, "_light_filter", lambda d, n: None):
            p = capmod.PendingCapture({"ir": "/dev/a", "rgb": "/dev/b"})
            t = _t.monotonic(); p.cancel(); self.assertLess(_t.monotonic() - t, 0.5)   # never blocks
            go.set(); p._thread.join(2)
        self.assertEqual(released, ["ir"])                                       # opened after cancel → released, RGB never opened

    def test_hw_apply_is_cheap_and_needs_no_subprocess(self):
        import time
        from faceup import hw
        t = time.monotonic(); n = hw.apply(); dt = time.monotonic() - t
        self.assertGreaterEqual(n, 1); self.assertLess(dt, 0.05)
        self.assertEqual(hw.threads_for(12), 4); self.assertEqual(hw.threads_for(2), 1)

    def test_confident_reject_needs_enough_far_frames(self):
        from faceup.engine.auth import Engine, Obs
        rec = {"early_reject": 0.65, "fusion_frames": 4}
        def o(d):
            return Obs(None, None, None, np.ones(128), False, "ok", d)
        far = [o(0.7) for _ in range(6)]
        self.assertTrue(Engine._confident_reject(far, rec))
        self.assertFalse(Engine._confident_reject(far[:5], rec))                       # not enough frames
        self.assertTrue(Engine._confident_reject(far[:5] + [o(0.5)], rec))             # one lucky frame no longer blocks it
        self.assertFalse(Engine._confident_reject(far[:3] + [o(0.4)] * 3, rec))          # half close: keep looking
        self.assertFalse(Engine._confident_reject(far, {"early_reject": 0, "fusion_frames": 4}))   # disabled

    def test_learning_stores_per_frame_samples(self):
        from faceup.engine import store
        from faceup.engine.auth import Engine, Obs
        from faceup.engine.liveness import PASS, ABSTAIN
        rng = np.random.default_rng(1)
        base = rng.normal(size=128); base /= np.linalg.norm(base)
        st = store.Store(user="t")
        st.add("C/level", "ir", base, 0.9, None, samples=[base])
        eng = Engine.__new__(Engine)                      # no models, no hw: only _learn is exercised
        eng.models = None; eng.verbose = False; eng.events = None
        frames = []
        for _ in range(4):
            v = base + rng.normal(scale=0.02, size=128); frames.append(v / np.linalg.norm(v))
        fused = np.mean(frames, axis=0); fused /= np.linalg.norm(fused)
        eng._last_fused = {"ir": fused}; eng._last_samples = {"ir": frames}
        beh = {"learn": True, "learn_max_dist": 0.30, "learn_min_novelty": 0.25, "learn_per_bin": 2}
        d = float(np.linalg.norm(fused - base))
        self.assertLess(d, 0.25); self.assertLess(d, beh["learn_max_dist"])   # a confident accept that is not novel
        # below the novelty bar: nothing learned
        self.assertEqual(eng._learn(st, {"ir": [Obs(None, None, None, fused, False, "ok")]}, {"ir": d}, [PASS("cross")], beh, {}), "")
        beh["learn_min_novelty"] = 0.0
        st.save = lambda: None
        learned = eng._learn(st, {"ir": [Obs(None, None, None, fused, False, "ok")]}, {"ir": d}, [PASS("cross"), ABSTAIN("motion")], beh, {})
        self.assertEqual(learned, "ir:unknown")
        m = [m for m in st.models if m.learned][0]
        self.assertEqual(len(m.samples), 4)                # the frames, not just their mean
        # no PASS at all → never learn
        self.assertEqual(eng._learn(st, {"ir": []}, {"ir": d}, [ABSTAIN("motion")], beh, {}), "")


if __name__ == "__main__":
    unittest.main()
