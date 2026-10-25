"""Pluggable recogniser (DESIGN.md §6.4a): alignment, distance scaling, store identity, installer."""
import hashlib
import io
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("WARD_ETC", tempfile.mkdtemp())
os.environ.setdefault("WARD_LOG", os.path.join(os.environ["WARD_ETC"], "log"))

from ward.engine import recognise, store
from ward.engine.detect import Box, Models


class Alignment(unittest.TestCase):
    def test_template_points_land_on_the_template(self):
        # a face drawn at 2x scale, rotated 10°, shifted: its 5 points must map back onto TEMPLATE
        th = np.deg2rad(10); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float32)
        pts = (recognise.TEMPLATE @ R.T) * 2.0 + np.array([40, 30], dtype=np.float32)
        img = np.zeros((400, 400, 3), np.uint8)
        for (x, y) in pts:
            img[int(y) - 2:int(y) + 3, int(x) - 2:int(x) + 3] = 255
        crop = recognise.align(img, pts)
        self.assertEqual(crop.shape, (112, 112, 3))
        for (x, y) in recognise.TEMPLATE:
            self.assertGreater(crop[int(round(y)), int(round(x))].max(), 0, (x, y))

    def test_three_point_alignment_uses_eyes_and_nose(self):
        pts = recognise.TEMPLATE[:3] * 1.5 + 10
        img = np.zeros((300, 300, 3), np.uint8)
        crop = recognise.align(img, pts)
        self.assertEqual(crop.shape, (112, 112, 3))

    def test_box_scaling_keeps_landmarks(self):
        b = Box(10, 10, 20, 20, np.array([[1.0, 2.0]] * 5, dtype=np.float32)).scaled(2.0)
        self.assertEqual((b.x, b.w), (20, 40)); self.assertEqual(b.lm5[0].tolist(), [2.0, 4.0])
        self.assertIsNone(Box(1, 1, 1, 1).scaled(2.0).lm5)


class Scaling(unittest.TestCase):
    def test_scale_factor_maps_radius_to_dlib(self):
        self.assertAlmostEqual(recognise.scale_factor(recognise.ARCFACE_RADIUS) * recognise.ARCFACE_RADIUS, 0.6)
        self.assertEqual(recognise.scale_factor(0.6), 1.0)

    def test_store_scales_distances(self):
        a = np.zeros(512); a[0] = 1.0
        b = np.zeros(512); b[1] = 1.0                         # raw distance sqrt(2)
        st = store.Store(user="t", recogniser="arcface:mbf", scale=0.5)
        st.add("C/level", "ir", a, 0.9, None, samples=[a])
        d, label = st.min_distance(b, "ir")
        self.assertAlmostEqual(d, np.sqrt(2) * 0.5); self.assertEqual(label, "C/level")
        self.assertAlmostEqual(st.distance(a, b), np.sqrt(2) * 0.5)
        self.assertEqual(st.dim, 512)

    def test_models_default_is_dlib_and_scale_one(self):
        m = Models("yunet", 320)
        self.assertEqual(m.recogniser_kind, "dlib"); self.assertEqual(m.recogniser_name, recognise.DLIB_NAME)
        self.assertEqual(m.distance_scale, 1.0)
        self.assertEqual(Models("yunet", 320, recogniser="bogus").recogniser_kind, "dlib")


class StoreIdentity(unittest.TestCase):
    def test_store_refuses_another_recogniser(self):
        os.environ["WARD_ETC"]  # set at import
        st = store.Store(user="ident", path=store.config.MODELS_DIR / "ident.json", recogniser="arcface:mbf")
        st.add("C/level", "ir", np.ones(512) / np.sqrt(512), 0.9, None)
        st.save()
        with self.assertRaises(store.StoreError):
            store.load("ident")                               # default = dlib space
        back = store.load("ident", "arcface:mbf", 0.55)
        self.assertEqual(back.recogniser, "arcface:mbf"); self.assertEqual(back.scale, 0.55); self.assertEqual(back.dim, 512)


class Installer(unittest.TestCase):
    def test_checksums_are_enforced(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("w600k_mbf.onnx", b"not a model")
        data = buf.getvalue()
        with self.assertRaises(RuntimeError):
            recognise.install("mbf", fetch=lambda url: data, say=lambda *_: None)   # zip hash differs from the pin
        with self.assertRaises(ValueError):
            recognise.install("nope", fetch=lambda url: data, say=lambda *_: None)
        self.assertFalse(recognise.available())

    def test_real_model_when_present(self):
        """If the pinned pack was already downloaded into the scratchpad, run it end to end."""
        cand = [p for p in Path("/tmp/claude-1000").glob("**/buffalo_sc/w600k_mbf.onnx")] if Path("/tmp/claude-1000").exists() else []
        if not cand:
            self.skipTest("model not on disk")
        self.assertEqual(recognise.sha256_of(cand[0]), recognise.PACKS["mbf"]["sha256"])
        arc = recognise.ArcFace(cand[0])
        a = arc.embed_aligned(np.full((112, 112, 3), 120, np.uint8))
        b = arc.embed_aligned(np.full((112, 112, 3), 121, np.uint8))
        self.assertEqual(a.shape, (512,)); self.assertAlmostEqual(float(np.linalg.norm(a)), 1.0, places=5)
        self.assertLess(float(np.linalg.norm(a - b)), 0.3)         # near-identical inputs → near-identical embeddings


if __name__ == "__main__":
    unittest.main()
