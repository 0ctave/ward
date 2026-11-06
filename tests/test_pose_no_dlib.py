"""Pose from YuNet's 5 landmarks — the dlib-less + ArcFace path (most distros).

Without python3-dlib there is no 68-point predictor, so pose (and therefore pose-guided
enrollment) must come from YuNet's 5 points. Regression for the shipped crash where
`ward setup`/`ward enroll` hit `models.sp68` and raised on a machine without dlib.
"""
import unittest

import numpy as np

from ward.engine import detect, pose


def _lm5(eye_l, eye_r, nose, mouth_l, mouth_r):
    return np.array([eye_l, eye_r, nose, mouth_l, mouth_r], dtype=np.float64)


# a plausible frontal face in full-frame px (image axes: +x right, +y down)
FRONTAL = _lm5((40, 50), (60, 50), (50, 62), (43, 76), (57, 76))


class Stub68(unittest.TestCase):
    def test_shape_and_filled_indices(self):
        lm = pose.stub68_from5(FRONTAL)
        self.assertEqual(lm.shape, (68, 2))
        # eyes sorted to image-left (36:42) / image-right (42:48)
        np.testing.assert_allclose(lm[36], (40, 50))
        np.testing.assert_allclose(lm[42], (60, 50))
        np.testing.assert_allclose(lm[30], (50, 62))          # nose tip
        np.testing.assert_allclose(lm[48], (43, 76))          # mouth corners sorted by x
        np.testing.assert_allclose(lm[54], (57, 76))

    def test_wings_cue_self_skips(self):
        # nose-wings (31,35) left at the nose tip -> zero span -> ratios() drops the cue, no div-by-0
        lm = pose.stub68_from5(FRONTAL)
        np.testing.assert_allclose(lm[31], lm[30])
        np.testing.assert_allclose(lm[35], lm[30])
        self.assertNotIn("yaw_wings", pose.ratios(lm, 0.0))

    def test_eye_order_independent(self):
        # swapping the two eye points must not change the result (sorted by x internally)
        swapped = _lm5((60, 50), (40, 50), (50, 62), (43, 76), (57, 76))
        np.testing.assert_allclose(pose.stub68_from5(FRONTAL), pose.stub68_from5(swapped))


class PoseSigns(unittest.TestCase):
    def _yaw(self, lm5):
        p = pose.estimate(pose.stub68_from5(lm5), 640, 360)
        self.assertIsNotNone(p)
        return p

    def test_frontal_is_near_level(self):
        p = self._yaw(FRONTAL)
        self.assertAlmostEqual(p.yaw, 0.0, delta=6.0)
        self.assertAlmostEqual(p.roll, 0.0, delta=3.0)

    def test_yaw_sign_flips_with_nose(self):
        # nose shifted toward image-right vs image-left must give opposite-signed yaw
        right = self._yaw(_lm5((40, 50), (60, 50), (56, 62), (43, 76), (57, 76)))
        left = self._yaw(_lm5((40, 50), (60, 50), (44, 62), (43, 76), (57, 76)))
        self.assertLess(right.yaw * left.yaw, 0.0)                        # opposite signs

    def test_roll_sign(self):
        # right eye lower in the image (larger y) -> positive roll (matches roll_from_eyes)
        tilted = _lm5((40, 46), (60, 54), (50, 62), (43, 76), (57, 76))
        self.assertGreater(pose.estimate(pose.stub68_from5(tilted), 640, 360).roll, 1.0)


@unittest.skipIf(detect.dlib is not None, "exercises the dlib-absent fallback only")
class Landmarks68Fallback(unittest.TestCase):
    def test_landmarks68_uses_stub_without_dlib(self):
        m = detect.Models(recogniser="arcface")
        box = detect.Box(10, 10, 80, 80, FRONTAL)
        out = m.landmarks68(np.zeros((360, 640), np.uint8), box)
        np.testing.assert_allclose(out, pose.stub68_from5(FRONTAL))

    def test_landmarks68_without_lm5_raises_clear_error(self):
        m = detect.Models(recogniser="arcface")
        with self.assertRaises(RuntimeError) as cm:
            m.landmarks68(np.zeros((360, 640), np.uint8), detect.Box(10, 10, 80, 80, None))
        self.assertIn("dlib", str(cm.exception))

    def test_preload_pose_is_noop(self):
        detect.Models(recogniser="arcface").preload_pose()   # must not raise


class CoarseSmoothing(unittest.TestCase):
    def test_coarse_pose_filter_lags_more_on_jitter(self):
        # a still signal with frame-to-frame noise: the coarse filter must damp it harder
        rng = [0.0, 3.0, -2.5, 2.0, -3.0, 2.5, -2.0, 3.0, -2.5, 2.0]
        _, normal = pose.enroll_filters(coarse=False)
        _, coarse = pose.enroll_filters(coarse=True)
        no, co = [], []
        for i, v in enumerate(rng):
            t = i / 30.0
            no.append(normal(pose.Pose(v, 0, 0), t).yaw)
            co.append(coarse(pose.Pose(v, 0, 0), t).yaw)
        self.assertLess(np.std(co[2:]), np.std(no[2:]))     # coarse output is steadier

    def test_coarse_lowers_cutoffs(self):
        lmf, posef = pose.enroll_filters(coarse=True)
        self.assertLess(lmf.mc, 1.2)                         # LandmarkFilter default
        self.assertLess(posef.min_cutoff, 0.6)               # PoseFilter default


class SmoothBoxKeepsLm5(unittest.TestCase):
    def test_smooth_box_preserves_lm5(self):
        f = pose.LandmarkFilter()
        box = detect.Box(10, 10, 80, 80, FRONTAL)
        sm = f.smooth_box(box, 0.0)
        self.assertIsNotNone(sm.lm5)
        np.testing.assert_allclose(sm.lm5, FRONTAL)


if __name__ == "__main__":
    unittest.main()
