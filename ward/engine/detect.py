"""Face detection, landmarks and embeddings on top of dlib (DESIGN.md §6.4).

One `Models` object per process; everything is loaded lazily so the fast path
(HOG + 5-point landmarks + ResNet) never pays for the 68-point predictor or the CNN
unless a voter or the config asks for them.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import cv2
import numpy as np

try:
    import dlib                                   # distro package; optional with recogniser = arcface + YuNet
except ImportError:                               # doctor still runs and explains; ArcFace + YuNet auth works
    dlib = None

# where distros / Howdy / hand installs put dlib's model files; WARD_DLIB_DIR is searched first
DLIB_DIRS = [
    Path("/usr/share/dlib"), Path("/usr/share/dlib-models"), Path("/usr/local/share/dlib"),
    Path("/usr/lib/howdy/dlib-data"), Path("/usr/share/howdy/dlib-data"), Path("/usr/lib/security/howdy/dlib-data"),
    Path("/usr/lib64/security/howdy/dlib-data")]
_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
YUNET_2022 = _MODELS_DIR / "face_detection_yunet_2022mar.onnx"   # runs on OpenCV 4.5.4–4.7 (Ubuntu 22.04/24.04, Debian 12)
YUNET_2023 = _MODELS_DIR / "face_detection_yunet_2023mar.onnx"   # OpenCV >= 4.8 (Fedora, Arch, Tumbleweed): the
                                                                # 2022 graph fails there in forward() (measured 4.13, 5.0)


def _cv_version() -> tuple[int, int]:
    try:
        return tuple(int(x) for x in cv2.__version__.split(".")[:2])  # type: ignore[return-value]
    except (AttributeError, ValueError):
        return (0, 0)


def yunet_candidates() -> list[Path]:
    """Bundled YuNet graphs, the one this OpenCV is expected to run first; the other is the retry."""
    order = [YUNET_2023, YUNET_2022] if _cv_version() >= (4, 8) else [YUNET_2022, YUNET_2023]
    return [p for p in order if p.exists()]


YUNET_MODEL = (yunet_candidates() or [YUNET_2022])[0]
YUNET_WIDTH = 480          # working width: 4 ms/frame on a laptop CPU, faces ≥ ~40 px
YUNET_SCORE = 0.6          # small/far faces score lower; false positives are filtered by size and landmarks


def dlib_dirs() -> list[Path]:
    """WARD_DLIB_DIR first (read at call time), then the known locations."""
    extra = os.environ.get("WARD_DLIB_DIR")
    return ([Path(extra)] if extra else []) + DLIB_DIRS


def model_path(name: str) -> Path:
    for d in dlib_dirs():
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(f"dlib model {name} not found in {[str(d) for d in dlib_dirs()]} (install dlib-models, or set WARD_DLIB_DIR)")


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int

    lm5: np.ndarray | None = None      # 5 landmarks from YuNet (image-left eye, image-right eye, nose, mouth corners), full-frame px

    @property
    def rect(self) -> dlib.rectangle:
        return dlib.rectangle(self.x, self.y, self.x + self.w, self.y + self.h)

    def scaled(self, s: float) -> "Box":
        return Box(int(self.x * s), int(self.y * s), int(self.w * s), int(self.h * s),
                   None if self.lm5 is None else self.lm5 * s)

    def iou(self, o: "Box") -> float:
        ix = max(0, min(self.x + self.w, o.x + o.w) - max(self.x, o.x))
        iy = max(0, min(self.y + self.h, o.y + o.h) - max(self.y, o.y))
        inter = ix * iy
        union = self.w * self.h + o.w * o.h - inter
        return inter / union if union else 0.0

    def normalised(self, width: int, height: int) -> tuple[float, float, float, float]:
        return (self.x / width, self.y / height, self.w / width, self.h / height)


def yunet_available() -> bool:
    return bool(yunet_candidates()) and hasattr(cv2, "FaceDetectorYN_create")


class Models:
    """detector: "yunet" (default: fast AND pose-robust), "hog" (frontal only), "cnn" (slow)."""

    def __init__(self, detector: str = "yunet", max_height: int = 320, accel: str = "cpu", recogniser: str = "dlib"):
        if detector == "yunet" and not yunet_available():
            detector = "hog"
        self.detector_kind = detector
        self.max_height = max_height
        self.accel = accel
        self.recogniser_kind = recogniser if recogniser in ("dlib", "arcface") else "dlib"
        self._arc = None
        # no dlib: fail at the first dlib use (rec/sp5/sp68), not here — the detector, YuNet, ArcFace
        # and every replay path with a patched embedder work without it (portability: Arch/SUSE/Debian)
        self._hog = dlib.get_frontal_face_detector() if dlib is not None else None
        self._yunet = None
        self._yunet_size = None
        self._yunet_models = yunet_candidates()      # popped on a graph this OpenCV cannot run
        self._misses = 0          # consecutive full-frame misses; the 2x-upsampled search runs on every 4th
        self._cnn = None
        self._sp5 = None
        self._sp68 = None
        self._rec = None

    # -- lazy loaders ---------------------------------------------------------
    @staticmethod
    def _need_dlib(what: str) -> None:
        if dlib is None:
            raise RuntimeError(f"{what} needs python3-dlib (not installed)")

    @property
    def cnn(self):
        if self._cnn is None:
            self._need_dlib("the CNN detector")
            self._cnn = dlib.cnn_face_detection_model_v1(str(model_path("mmod_human_face_detector.dat")))
        return self._cnn

    @property
    def sp5(self):
        if self._sp5 is None:
            self._need_dlib("the 5-point landmarks")
            self._sp5 = dlib.shape_predictor(str(model_path("shape_predictor_5_face_landmarks.dat")))
        return self._sp5

    @property
    def sp68(self):
        if self._sp68 is None:
            self._need_dlib("the 68-point landmarks (pose)")
            self._sp68 = dlib.shape_predictor(str(model_path("shape_predictor_68_face_landmarks.dat")))
        return self._sp68

    @property
    def rec(self):
        if self._rec is None:
            self._need_dlib("the dlib recogniser (or `ward recogniser install` + `recogniser use arcface`)")
            self._rec = dlib.face_recognition_model_v1(str(model_path("dlib_face_recognition_resnet_model_v1.dat")))
        return self._rec

    @property
    def arc(self):
        if self._arc is None:
            from .. import hw
            from . import recognise
            self._arc = recognise.ArcFace(backend_target=hw.dnn_target(self.accel))
        return self._arc

    @property
    def recogniser_name(self) -> str:
        """Identity of the embedding space, recorded in the store (a change forces re-enrol)."""
        from . import recognise
        return self.arc.store_name if self.recogniser_kind == "arcface" else recognise.DLIB_NAME

    @property
    def distance_scale(self) -> float:
        """Raw embedding distance × this = dlib-equivalent units used by every config gate."""
        from . import recognise
        return recognise.scale_factor(self.arc.radius) if self.recogniser_kind == "arcface" else 1.0

    def preload(self) -> None:
        """Load what `embed` will need, so the cost overlaps the camera open rather than the first face."""
        if self.recogniser_kind == "arcface":
            self.arc
            if dlib is not None:
                try:
                    self.sp5                          # only the fallback alignment (HOG/CNN boxes) needs it
                except OSError:
                    pass
        elif dlib is not None:
            try:
                self.sp5; self.rec
            except OSError:                           # dlib without its model files (Tumbleweed has no
                pass                                  # dlib-models): the first embed raises the clear error
        # Build the YuNet FaceDetectorYN now too: it is constructed lazily on the first detect()
        # (~50–150 ms to allocate its buffers), which otherwise lands on the first captured frame
        # on the cold in-process path. A one-shot dummy detect overlaps that with the camera open.
        try:
            self.detect(np.zeros((360, 640), np.uint8))
        except Exception:
            pass                                      # fail-soft: preload is a speed-up, never a dependency

    def _yunet_detect(self, gray: np.ndarray) -> Box | None:
        H, W = gray.shape[:2]
        s = min(1.0, YUNET_WIDTH / W)
        size = (int(round(W * s)), int(round(H * s)))
        if self._yunet is None:
            from .. import hw
            backend, target = hw.dnn_target(self.accel)
            self._yunet = cv2.FaceDetectorYN_create(str(self._yunet_models[0]), "", size, YUNET_SCORE, 0.3, 500, backend, target)
            self._yunet_size = size
        elif size != self._yunet_size:
            self._yunet.setInputSize(size); self._yunet_size = size
        img = cv2.cvtColor(gray if s == 1.0 else cv2.resize(gray, size), cv2.COLOR_GRAY2BGR)
        _, faces = self._yunet.detect(img)
        if faces is None or len(faces) == 0:
            return None
        f = max(faces, key=lambda r: r[2] * r[3])
        x, y, w, h = (float(v) / s for v in f[:4])
        lm5 = (np.asarray(f[4:14], dtype=np.float32).reshape(5, 2) / s) if len(f) >= 14 else None
        # dlib's landmark predictors were trained on square HOG-style boxes (brow to chin);
        # YuNet's are taller (forehead to chin). Normalise so the 68/5-point fits stay accurate.
        side = max(w, h * 0.85)
        cx, cy = x + w / 2, y + h * 0.55
        return Box(int(cx - side / 2), int(cy - side / 2), int(side), int(side), lm5)

    # -- operations -------------------------------------------------------------
    def detect(self, gray: np.ndarray, near: Box | None = None) -> Box | None:
        if self.detector_kind == "yunet":
            try:
                return self._yunet_detect(gray)
            except cv2.error as e:
                if len(self._yunet_models) > 1:      # the other bundled graph (OpenCV version mismatch)
                    self._yunet_models.pop(0); self._yunet = None
                    try:
                        return self._yunet_detect(gray)
                    except cv2.error as e2:
                        e = e2
                # an OpenCV build whose DNN cannot run the bundled model (old 4.5.x, odd builds):
                # use dlib's HOG for the rest of this process instead of failing every frame
                if dlib is None:
                    raise
                self.detector_kind = "hog"
                self.fallback_reason = f"YuNet unavailable ({str(e).splitlines()[0][:80]}); using HOG"
        if dlib is None:
            return None
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
        if self.recogniser_kind == "arcface":
            return self.arc.embed(rgb, self.points5(rgb, box))
        shape = self.landmarks5(rgb, box)
        vec = np.asarray(self.rec.compute_face_descriptor(rgb, shape, 1), dtype=np.float64)
        return vec / (np.linalg.norm(vec) or 1.0)

    def points5(self, rgb: np.ndarray, box: Box) -> np.ndarray:
        """Alignment points in ArcFace template order. YuNet's own 5 landmarks when the box came
        from it; otherwise eye centres + nose from dlib's 5-point predictor (3 points)."""
        if box.lm5 is not None:
            return box.lm5
        self._need_dlib("alignment without YuNet landmarks")
        shape = self.landmarks5(rgb, box)
        p = np.array([(pt.x, pt.y) for pt in shape.parts()], dtype=np.float32)
        # dlib order: outer left-eye corner, inner left-eye corner, outer right-eye corner, inner right-eye corner, nose bottom
        eye_a, eye_b, nose = (p[0] + p[1]) / 2, (p[2] + p[3]) / 2, p[4]
        left, right = (eye_a, eye_b) if eye_a[0] < eye_b[0] else (eye_b, eye_a)
        return np.array([left, right, nose], dtype=np.float32)


def fuse(embeddings: list[np.ndarray]) -> np.ndarray:
    """Average of L2-normalised embeddings, re-normalised (DESIGN.md §6.4)."""
    m = np.mean(np.stack(embeddings), axis=0)
    return m / (np.linalg.norm(m) or 1.0)
