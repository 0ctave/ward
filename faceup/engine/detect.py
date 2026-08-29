"""Face detection, landmarks and embeddings on top of dlib (DESIGN.md §6.4).

One `Models` object per process; everything is loaded lazily so the fast path
(HOG + 5-point landmarks + ResNet) never pays for the 68-point predictor or the CNN
unless a voter or the config asks for them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import dlib
import numpy as np

DLIB_DIRS = [Path("/usr/share/dlib"), Path("/usr/lib/howdy/dlib-data"), Path("/usr/share/dlib-models")]
YUNET_MODEL = Path(__file__).resolve().parents[1] / "models" / "face_detection_yunet_2022mar.onnx"
YUNET_WIDTH = 480          # working width: 4 ms/frame on a laptop CPU, faces ≥ ~40 px
YUNET_SCORE = 0.6          # small/far faces score lower; false positives are filtered by size and landmarks


def model_path(name: str) -> Path:
    for d in DLIB_DIRS:
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(f"dlib model {name} not found in {[str(d) for d in DLIB_DIRS]} (install dlib-models)")


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def rect(self) -> dlib.rectangle:
        return dlib.rectangle(self.x, self.y, self.x + self.w, self.y + self.h)

    def scaled(self, s: float) -> "Box":
        return Box(int(self.x * s), int(self.y * s), int(self.w * s), int(self.h * s))

    def iou(self, o: "Box") -> float:
        ix = max(0, min(self.x + self.w, o.x + o.w) - max(self.x, o.x))
        iy = max(0, min(self.y + self.h, o.y + o.h) - max(self.y, o.y))
        inter = ix * iy
        union = self.w * self.h + o.w * o.h - inter
        return inter / union if union else 0.0

    def normalised(self, width: int, height: int) -> tuple[float, float, float, float]:
        return (self.x / width, self.y / height, self.w / width, self.h / height)


def yunet_available() -> bool:
    return YUNET_MODEL.exists() and hasattr(cv2, "FaceDetectorYN_create")


class Models:
    """detector: "yunet" (default: fast AND pose-robust), "hog" (frontal only), "cnn" (slow)."""

    def __init__(self, detector: str = "yunet", max_height: int = 320, accel: str = "cpu"):
        if detector == "yunet" and not yunet_available():
            detector = "hog"
        self.detector_kind = detector
        self.max_height = max_height
        self.accel = accel
        self._hog = dlib.get_frontal_face_detector()
        self._yunet = None
        self._yunet_size = None
        self._misses = 0          # consecutive full-frame misses; the 2x-upsampled search runs on every 4th
        self._cnn = None
        self._sp5 = None
        self._sp68 = None
        self._rec = None

    # -- lazy loaders ---------------------------------------------------------
    @property
    def cnn(self):
        if self._cnn is None:
            self._cnn = dlib.cnn_face_detection_model_v1(str(model_path("mmod_human_face_detector.dat")))
        return self._cnn

    @property
    def sp5(self):
        if self._sp5 is None:
            self._sp5 = dlib.shape_predictor(str(model_path("shape_predictor_5_face_landmarks.dat")))
        return self._sp5

    @property
    def sp68(self):
        if self._sp68 is None:
            self._sp68 = dlib.shape_predictor(str(model_path("shape_predictor_68_face_landmarks.dat")))
        return self._sp68

    @property
    def rec(self):
        if self._rec is None:
            self._rec = dlib.face_recognition_model_v1(str(model_path("dlib_face_recognition_resnet_model_v1.dat")))
        return self._rec

    def _yunet_detect(self, gray: np.ndarray) -> Box | None:
        H, W = gray.shape[:2]
        s = min(1.0, YUNET_WIDTH / W)
        size = (int(round(W * s)), int(round(H * s)))
        if self._yunet is None:
            from .. import hw
            backend, target = hw.dnn_target(self.accel)
            self._yunet = cv2.FaceDetectorYN_create(str(YUNET_MODEL), "", size, YUNET_SCORE, 0.3, 500, backend, target)
            self._yunet_size = size
        elif size != self._yunet_size:
            self._yunet.setInputSize(size); self._yunet_size = size
        img = cv2.cvtColor(gray if s == 1.0 else cv2.resize(gray, size), cv2.COLOR_GRAY2BGR)
        _, faces = self._yunet.detect(img)
        if faces is None or len(faces) == 0:
            return None
        f = max(faces, key=lambda r: r[2] * r[3])
        x, y, w, h = (float(v) / s for v in f[:4])
        # dlib's landmark predictors were trained on square HOG-style boxes (brow to chin);
        # YuNet's are taller (forehead to chin). Normalise so the 68/5-point fits stay accurate.
        side = max(w, h * 0.85)
        cx, cy = x + w / 2, y + h * 0.55
        return Box(int(cx - side / 2), int(cy - side / 2), int(side), int(side))

    # -- operations -------------------------------------------------------------
    def detect(self, gray: np.ndarray, near: Box | None = None) -> Box | None:
        if self.detector_kind == "yunet":
            return self._yunet_detect(gray)
        return self._detect_dlib(gray, near)

    def _detect_dlib(self, gray: np.ndarray, near: Box | None = None) -> Box | None:
        """Largest face in the frame, in full-frame coordinates.

        `near`: the face box from the previous frame. Detection then runs only on a region
        around it without upsampling (~5 ms instead of ~60 ms); if that misses, it falls back
        to the full frame. HOG needs ~80 px faces, so the full-frame pass first tries without
        upsampling (fast, fine for a laptop user) and only then with 2x (small/far faces).
        """
        H, W = gray.shape[:2]
        if near is not None and self.detector_kind != "cnn":
            mx, my = int(near.w * 0.4), int(near.h * 0.4)
            x0, y0 = max(0, near.x - mx), max(0, near.y - my)
            x1, y1 = min(W, near.x + near.w + mx), min(H, near.y + near.h + my)
            roi = gray[y0:y1, x0:x1]
            if roi.size:
                dets = list(self._hog(roi, 0))
                if dets:
                    r = max(dets, key=lambda d: d.width() * d.height())
                    return Box(r.left() + x0, r.top() + y0, r.width(), r.height())
        s = min(1.0, self.max_height / H)
        img = gray if s == 1.0 else cv2.resize(gray, (int(W * s), int(H * s)))
        if self.detector_kind == "cnn":
            dets = [d.rect for d in self.cnn(img, 0)]
        else:
            dets = list(self._hog(img, 0))
            if not dets and self._misses % 4 == 0:      # expensive 2x search only now and then
                dets = list(self._hog(img, 1))
        if not dets:
            self._misses += 1
            return None
        self._misses = 0
        r = max(dets, key=lambda d: d.width() * d.height())
        return Box(r.left(), r.top(), r.width(), r.height()).scaled(1.0 / s)

    def landmarks5(self, rgb: np.ndarray, box: Box):
        return self.sp5(rgb, box.rect)

    def landmarks68(self, gray: np.ndarray, box: Box) -> np.ndarray:
        shape = self.sp68(gray, box.rect)
        return np.array([(p.x, p.y) for p in shape.parts()], dtype=np.float64)

    def embed(self, rgb: np.ndarray, box: Box) -> np.ndarray:
        shape = self.landmarks5(rgb, box)
        vec = np.asarray(self.rec.compute_face_descriptor(rgb, shape, 1), dtype=np.float64)
        return vec / (np.linalg.norm(vec) or 1.0)


def fuse(embeddings: list[np.ndarray]) -> np.ndarray:
    """Average of L2-normalised embeddings, re-normalised (DESIGN.md §6.4)."""
    m = np.mean(np.stack(embeddings), axis=0)
    return m / (np.linalg.norm(m) or 1.0)
