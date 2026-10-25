"""Recognisers (DESIGN.md §6.4a): the network that turns an aligned face into an embedding.

Two backends behind one interface, chosen by `recognition.recogniser`:

- **dlib** (default, shipped by the distro): ResNet-34, 128-d, trained on colour photos.
  Vendor's same-person radius 0.6 (99.38 % on LFW). On the IR camera the user's own
  frames spread 0.12–0.47 because the network never saw infrared skin.
- **arcface** (opt-in, `ward recogniser install`): insightface's ArcFace family run by
  OpenCV DNN (no new dependency). `w600k_mbf` = MobileFaceNet, 512-d, 6 ms on a laptop CPU;
  `w600k_r50` = ResNet-50, 512-d, 57 ms. ArcFace-margin training gives much tighter
  identity clusters than dlib (IJB-C TAR ≈ 95 % at FAR 1e-4 for r50 vs. far less for dlib),
  which is what a Windows-Hello-class false-accept target needs. Licence: insightface's code
  is MIT but the *model packs* are for non-commercial research use — hence not bundled;
  the installer shows the notice.

Distances: both backends return the Euclidean distance between L2-normalised embeddings,
rescaled so that the backend's nominal same-person radius maps to dlib's 0.6. Every gate in
the config (certainty, hard_cap, early_reject, learn_*) therefore keeps one meaning across
recognisers. ArcFace's radius is provisional (`ARCFACE_RADIUS`) until the lab calibrates it.
"""
from __future__ import annotations

import hashlib
import io
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np

from . import config

DLIB_NAME = "dlib_face_recognition_resnet_model_v1"
ARCFACE_MODEL = config.ETC / "arcface.onnx"
ARCFACE_INFO = config.ETC / "arcface.txt"          # which variant is installed
DLIB_RADIUS = 0.6
# Euclidean distance between unit vectors at the cosine similarity insightface treats as
# "same person" for these packs (≈ 0.40): sqrt(2 − 2·0.40) ≈ 1.10. Provisional — calibrate in
# the lab with an impostor subject. Measured 2026-09-07 on the reference IR camera (face 116 px,
# YuNet landmarks, single frames): the user's own frames sit at median 0.36 / p95 0.57 raw from
# their mean (r50; mbf 0.39 / 0.66), i.e. 0.20 / 0.31 in dlib units after ×0.545 — comparable to
# dlib's 0.10 / 0.29 on the same frames; multi-frame fusion halves both. dlib's 68-landmark
# alignment was worse for ArcFace (p95 0.78) and flip-averaging gained < 0.02, so neither is used.
ARCFACE_RADIUS = 1.10

# Pinned model packs (insightface v0.7 release assets): zip sha256, member, member sha256.
PACKS = {
    "mbf": {"url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip",
            "zip_sha256": "57d31b56b6ffa911c8a73cfc1707c73cab76efe7f13b675a05223bf42de47c72",
            "member": "w600k_mbf.onnx", "sha256": "9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f", "size_mb": 16, "name": "w600k_mbf (MobileFaceNet, 5 ms)"},
    "r50": {"url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
            "zip_sha256": "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f",
            "member": "w600k_r50.onnx", "sha256": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43", "size_mb": 281, "name": "w600k_r50 (ResNet-50, 57 ms)"},
}
LICENCE_NOTE = ("insightface model packs are provided for non-commercial research use "
                "(https://github.com/deepinsight/insightface#license); ward downloads them for you, it does not ship them.")

# ArcFace alignment template: 112×112 crop, points = image-left eye, image-right eye, nose tip,
# image-left mouth corner, image-right mouth corner (the standard insightface template)
TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def available() -> bool:
    return ARCFACE_MODEL.exists()


def installed_variant() -> str:
    try:
        return ARCFACE_INFO.read_text().strip()
    except OSError:
        return "?" if available() else ""


def align(rgb: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Similarity-warp the face to the 112×112 ArcFace template from 5 (or 3) landmarks
    ordered like TEMPLATE. With 3 points (eyes + nose, e.g. from dlib's 5-point predictor)
    the mouth rows of the template are dropped."""
    pts = np.asarray(pts, dtype=np.float32)
    n = min(len(pts), 5)
    src, dst = pts[:n], TEMPLATE[:n]
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:                                  # degenerate points: scale the box instead
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    return cv2.warpAffine(rgb, M, (112, 112), borderValue=(0, 0, 0))


class ArcFace:
    name = "arcface"

    def __init__(self, path: Path = ARCFACE_MODEL, backend_target: tuple[int, int] | None = None):
        if not path.exists():
            raise FileNotFoundError(f"{path} — run `sudo ward recogniser install`")
        self.net = cv2.dnn.readNetFromONNX(str(path))
        if backend_target:
            self.net.setPreferableBackend(backend_target[0]); self.net.setPreferableTarget(backend_target[1])
        self.variant = installed_variant()
        self.dim = 512
        self.radius = ARCFACE_RADIUS
        self.store_name = f"arcface:{self.variant or 'unknown'}"

    def embed_aligned(self, crop112_rgb: np.ndarray) -> np.ndarray:
        blob = cv2.dnn.blobFromImage(crop112_rgb, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=False)
        self.net.setInput(blob)
        vec = self.net.forward().reshape(-1).astype(np.float64)
        return vec / (np.linalg.norm(vec) or 1.0)

    def embed(self, rgb: np.ndarray, pts: np.ndarray) -> np.ndarray:
        return self.embed_aligned(align(rgb, pts))


def scale_factor(radius: float) -> float:
    """Multiply a backend's raw distance by this to express it in dlib-equivalent units."""
    return DLIB_RADIUS / radius


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def install(variant: str = "mbf", fetch=None, say=print) -> Path:
    """Download the pinned pack, verify the checksums, extract the ONNX to /etc/ward.
    `fetch(url) -> bytes` can be injected (tests)."""
    if variant not in PACKS:
        raise ValueError(f"unknown variant {variant!r}; choose from {', '.join(PACKS)}")
    pack = PACKS[variant]
    if fetch is None:
        import urllib.request

        def fetch(url: str) -> bytes:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
    say(f"downloading {pack['name']} ({pack['size_mb']} MB) from {pack['url']}")
    data = fetch(pack["url"])
    got = hashlib.sha256(data).hexdigest()
    if pack["zip_sha256"] and got != pack["zip_sha256"]:
        raise RuntimeError(f"checksum mismatch for the download: {got} (expected {pack['zip_sha256']})")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        member = z.read(pack["member"])
    got = hashlib.sha256(member).hexdigest()
    if pack["sha256"] and got != pack["sha256"]:
        raise RuntimeError(f"checksum mismatch for {pack['member']}: {got}")
    ARCFACE_MODEL.parent.mkdir(parents=True, exist_ok=True)
    tmp = ARCFACE_MODEL.with_suffix(".onnx.tmp")
    tmp.write_bytes(member)
    os.chmod(tmp, 0o644)
    os.replace(tmp, ARCFACE_MODEL)
    ARCFACE_INFO.write_text(variant + "\n")
    # load once to prove the distro OpenCV can run it
    ArcFace(ARCFACE_MODEL).embed_aligned(np.zeros((112, 112, 3), np.uint8))
    say(f"installed {ARCFACE_MODEL} ({pack['member']}, sha256 {got[:12]}…)")
    return ARCFACE_MODEL
