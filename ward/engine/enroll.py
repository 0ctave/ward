"""Enrollment (DESIGN.md §8): calibration + per-pose, per-spectrum fused embeddings."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .capture import Capture
from .detect import Box, Models, fuse
from .quality import MIN_FACE_PX, measure
from .store import Store

MIN_ENROLL_FACE_PX = 60      # ≈ 90 cm from a 640 px-wide laptop camera; dlib's chip is upscaled from here
WRONG_PERSON_DIST = 0.6
BEST_FRAMES = 3


@dataclass
class PoseResult:
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)   # spectrum -> fused embedding
    quality: dict[str, float] = field(default_factory=dict)
    boxes: dict[str, Box] = field(default_factory=dict)
    frames: int = 0
    problems: list[str] = field(default_factory=list)


def pair_offset(bi: Box, br: Box, ir_shape, rgb_shape) -> tuple[float, float, float]:
    """(scale, dx, dy) mapping the RGB face box onto the IR one, in IR-frame fractions."""
    ih, iw = ir_shape[:2]
    rh, rw = rgb_shape[:2]
    nb = Box(int(br.x * iw / rw), int(br.y * ih / rh), int(br.w * iw / rw), int(br.h * ih / rh))
    sc = bi.w / nb.w if nb.w else 1.0
    return sc, (bi.x - nb.x * sc) / iw, (bi.y - nb.y * sc) / ih


def summarise_pairs(pairs: list) -> dict | None:
    if not pairs:
        return None
    s, dx, dy = zip(*pairs)
    return {"dx": round(float(np.median(dx)), 4), "dy": round(float(np.median(dy)), 4), "s": round(float(np.median(s)), 4)}


def calibrate(cap: Capture, models: Models, cams: dict, n_pairs: int = 10, timeout: float = 4.0) -> dict | None:
    """Offset/scale between IR and RGB face boxes (only meaningful with both cameras)."""
    if set(cap.spectra) != {"ir", "rgb"}:
        return None
    pairs = []
    t_end = time.monotonic() + timeout
    while len(pairs) < n_pairs and time.monotonic() < t_end:
        fs = cap.read_set()
        if set(fs) != {"ir", "rgb"}:
            continue
        bi = models.detect(fs["ir"].gray)
        br = models.detect(fs["rgb"].gray)
        if bi is None or br is None:
            continue
        pairs.append(pair_offset(bi, br, fs["ir"].gray.shape, fs["rgb"].gray.shape))
    return summarise_pairs(pairs)


def capture_pose(cap: Capture, models: Models, cams: dict, seconds: float = 1.2) -> PoseResult:
    res = PoseResult()
    per: dict[str, list[tuple[float, np.ndarray, Box]]] = {s: [] for s in cap.spectra}
    dark = {s: 0 for s in cap.spectra}
    small = {s: 0 for s in cap.spectra}
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        fs = cap.read_set()
        for s, fr in fs.items():
            res.frames += 1
            q = measure(fr.gray, models.max_height)
            if q.reason(cams[f"{s}_dark_threshold"]):
                if q.mean < cams[f"{s}_dark_threshold"]:
                    dark[s] += 1
                continue
            box = models.detect(fr.gray)
            if box is None:
                continue
            if box.w < MIN_ENROLL_FACE_PX:
                small[s] += 1
                continue
            per[s].append((q.score(), models.embed(fr.rgb, box), box))
    for s, lst in per.items():
        if not lst:
            if dark[s]:
                res.problems.append(f"{s}: frames too dark")
            elif small[s]:
                res.problems.append(f"{s}: face too small — move closer")
            else:
                res.problems.append(f"{s}: no face found")
            continue
        lst.sort(key=lambda t: t[0], reverse=True)
        best = lst[:BEST_FRAMES]
        res.embeddings[s] = fuse([e for _, e, _ in best])
        res.quality[s] = float(np.mean([q for q, _, _ in best]))
        res.boxes[s] = best[0][2]
    return res


def wrong_person(store: Store, res: PoseResult) -> str | None:
    for s, emb in res.embeddings.items():
        if store.by_spectrum(s):
            d, _ = store.min_distance(emb, s)
            if d > WRONG_PERSON_DIST:
                return f"{s}: this face is {d:.2f} away from every enrolled model — is it the same person?"
    return None
