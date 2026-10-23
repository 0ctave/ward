"""LFW verification benchmark: the recogniser's impostor/genuine separation and the match rates
at faceup's gate. A reliability instrument, never on the auth path.

    python3 -m faceup.lfw --dir ~/lfw --pairs ~/lfw/pairs.txt            # full 6000 pairs
    python3 -m faceup.lfw --dir ~/lfw --pairs ~/lfw/pairs.txt --limit 1000

LFW (Labeled Faces in the Wild) is RGB-only still images, so this measures ONE thing: how well the
embedding network separates same-person from different-person faces, and therefore roughly how
often a single frame of a stranger would fall inside the gate (FMR) and how often you would be
rejected (FNMR). It does NOT exercise the IR / cross-spectrum path, the liveness voters, the
multi-frame fusion, or the whitened template match — the live decision compares a frame against
your fused, whitened enrolment, not two single photos. Read the numbers as the recogniser's raw
discriminative power and a single-frame upper bound on impostor acceptance, not the deployed FAR.

The image load → detect(gray) → embed(RGB) path mirrors engine/capture + detect exactly, so the
distances are in the same dlib units as the config gate. Dataset images are research-only; keep
them out of the repo.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np


def read_pairs(pairs_file: str, image_root: str, ext: str = ".jpg") -> list[tuple[str, str, bool]]:
    """Parse an LFW pairs file into (pathA, pathB, same_person). Handles the standard format:
    a header line (fold/count), 3-field lines `name i j` (same person, images i and j), and
    4-field lines `nameA i nameB j` (different people). Missing files are left for the caller to
    skip, so this does no disk I/O beyond reading the list."""
    def img(name: str, idx: str) -> str:
        return os.path.join(image_root, name, f"{name}_{int(idx):04d}{ext}")

    pairs: list[tuple[str, str, bool]] = []
    with open(pairs_file) as f:
        for line in f:
            p = line.split()
            if len(p) == 3:
                pairs.append((img(p[0], p[1]), img(p[0], p[2]), True))
            elif len(p) == 4:
                pairs.append((img(p[0], p[1]), img(p[2], p[3]), False))
            # 1- or 2-field lines are the header (fold count / pairs-per-fold): ignored
    return pairs


def balanced_subset(pairs: list[tuple[str, str, bool]], limit: int, seed: int = 0) -> list[tuple[str, str, bool]]:
    """At most `limit` pairs, split evenly between same and different (LFW groups them per fold, so
    a plain head() would be all one class). Deterministic with `seed`."""
    if limit <= 0 or limit >= len(pairs):
        return pairs
    import random
    rng = random.Random(seed)
    same = [p for p in pairs if p[2]]
    diff = [p for p in pairs if not p[2]]
    rng.shuffle(same); rng.shuffle(diff)
    k = limit // 2
    out = same[:k] + diff[:k]
    rng.shuffle(out)
    return out


class Embedder:
    """Path → face embedding, cached (an LFW image appears in several pairs). Load BGR, detect on
    grey, embed on RGB — the exact engine path. Returns None when no face is found."""

    def __init__(self, models):
        self.models = models
        self._cache: dict[str, np.ndarray | None] = {}
        self.misses = 0

    def __call__(self, path: str) -> np.ndarray | None:
        if path in self._cache:
            return self._cache[path]
        import cv2
        emb = None
        img = cv2.imread(path, cv2.IMREAD_COLOR)        # BGR, like a camera frame
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            box = self.models.detect(gray)
            if box is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                try:
                    emb = self.models.embed(rgb, box)
                except Exception:
                    emb = None
        if emb is None:
            self.misses += 1
        self._cache[path] = emb
        return emb


def evaluate(embed_fn, pairs, scale: float = 1.0, progress=None) -> tuple[np.ndarray, np.ndarray, int]:
    """Distances for every usable pair. Returns (genuine, impostor, skipped). `embed_fn(path)`
    returns a unit embedding or None; a pair with either image unreadable/faceless is skipped."""
    genuine: list[float] = []
    impostor: list[float] = []
    skipped = 0
    for i, (a, b, same) in enumerate(pairs):
        ea, eb = embed_fn(a), embed_fn(b)
        if ea is None or eb is None:
            skipped += 1
        else:
            d = float(np.linalg.norm(ea - eb)) * scale
            (genuine if same else impostor).append(d)
        if progress and (i + 1) % 500 == 0:
            progress(i + 1, len(pairs), skipped)
    return np.array(genuine), np.array(impostor), skipped


def rates_at(genuine: np.ndarray, impostor: np.ndarray, t: float) -> tuple[float, float]:
    """(FMR, FNMR) if the engine accepts at distance ≤ t: FMR = impostors accepted,
    FNMR = genuine rejected."""
    fmr = float(np.mean(impostor <= t)) if len(impostor) else float("nan")
    fnmr = float(np.mean(genuine > t)) if len(genuine) else float("nan")
    return fmr, fnmr


def eer(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """(threshold, rate) where FMR ≈ FNMR — the equal-error point."""
    if not len(genuine) or not len(impostor):
        return float("nan"), float("nan")
    ts = np.unique(np.concatenate([genuine, impostor]))
    best_t, best_gap, best_rate = ts[0], float("inf"), float("nan")
    for t in ts:
        fmr, fnmr = rates_at(genuine, impostor, t)
        gap = abs(fmr - fnmr)
        if gap < best_gap:
            best_t, best_gap, best_rate = float(t), gap, (fmr + fnmr) / 2
    return best_t, best_rate


def auc(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """P(impostor distance > genuine distance): 1.0 = perfect separation (Mann–Whitney)."""
    if not len(genuine) or not len(impostor):
        return float("nan")
    g = np.sort(genuine)
    # for each impostor distance, how many genuine distances are strictly below it
    below = np.searchsorted(g, impostor, side="left")
    ties = np.searchsorted(g, impostor, side="right") - below
    return float(np.sum(below + 0.5 * ties) / (len(genuine) * len(impostor)))


def metrics(genuine: np.ndarray, impostor: np.ndarray, gates: dict[str, float]) -> dict:
    et, er = eer(genuine, impostor)
    out = {
        "n_genuine": int(len(genuine)), "n_impostor": int(len(impostor)),
        "genuine_mean": _f(np.mean(genuine)), "genuine_p95": _pct(genuine, 95),
        "impostor_mean": _f(np.mean(impostor)), "impostor_p5": _pct(impostor, 5),
        "auc": auc(genuine, impostor), "eer": er, "eer_threshold": et,
        "at": {name: dict(zip(("fmr", "fnmr"), rates_at(genuine, impostor, g))) | {"gate": g}
               for name, g in gates.items()},
    }
    return out


def _f(x) -> float:
    return float(x) if len(np.atleast_1d(x)) and np.isfinite(x) else float("nan")


def _pct(a: np.ndarray, p: float) -> float:
    return float(np.percentile(a, p)) if len(a) else float("nan")


def format_report(m: dict, skipped: int, seconds: float) -> str:
    L = ["LFW verification benchmark (recogniser only — no IR, liveness or fusion)", ""]
    L.append(f"  pairs scored : {m['n_genuine']} genuine, {m['n_impostor']} impostor"
             + (f"  ({skipped} skipped: no face / unreadable)" if skipped else ""))
    L.append(f"  genuine dist : mean {m['genuine_mean']:.3f}, p95 {m['genuine_p95']:.3f}")
    L.append(f"  impostor dist: mean {m['impostor_mean']:.3f}, p5  {m['impostor_p5']:.3f}")
    L.append(f"  separation   : AUC {m['auc']:.4f}   EER {m['eer'] * 100:.2f}% at distance {m['eer_threshold']:.3f}")
    L.append("")
    L.append("  at each gate (accept if distance ≤ gate):")
    L.append(f"    {'gate':<16} {'threshold':>9} {'FMR (impostor in)':>18} {'FNMR (you out)':>16}")
    for name, r in m["at"].items():
        L.append(f"    {name:<16} {r['gate']:>9.3f} {r['fmr'] * 100:>17.3f}% {r['fnmr'] * 100:>15.2f}%")
    L.append("")
    L.append(f"  FMR = how often a stranger's single frame lands inside the gate (lower = safer).")
    L.append(f"  FNMR = how often your own frame is rejected (lower = smoother). The live engine")
    L.append(f"  fuses frames and matches a whitened template, so deployed rates are better than FMR here.")
    L.append(f"  [{seconds:.0f}s]")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="faceup.lfw", description="LFW impostor/genuine benchmark for the faceup recogniser.")
    ap.add_argument("--dir", required=True, help="LFW image root (contains one folder per person)")
    ap.add_argument("--pairs", required=True, help="LFW pairs file (e.g. pairs.txt)")
    ap.add_argument("--ext", default=".jpg", help="image extension (default .jpg; .png for some mirrors)")
    ap.add_argument("--limit", type=int, default=0, help="score at most this many pairs (balanced, deterministic)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true", help="emit the metrics as JSON")
    ap.add_argument("--csv", help="write per-pair distances to this CSV")
    a = ap.parse_args(argv)

    from .engine import config as cfgmod
    from .engine.detect import Models
    cfg = cfgmod.load()["recognition"]
    gates = {"certainty (IR gate)": float(cfg["certainty"]),
             "rgb_certainty": float(cfg["rgb_certainty"]),
             "dlib classic 0.6": 0.6}

    pairs = read_pairs(a.pairs, a.dir, a.ext)
    if not pairs:
        print(f"no pairs parsed from {a.pairs}", file=sys.stderr); return 1
    pairs = balanced_subset(pairs, a.limit, a.seed)
    models = Models(cfg["detector"], cfg["max_height"], "cpu", cfg.get("recogniser", "dlib"))
    emb = Embedder(models)
    t0 = time.monotonic()

    def progress(done, total, skipped):
        print(f"  … {done}/{total} pairs ({skipped} skipped, {emb.misses} faceless images)", file=sys.stderr)

    if a.csv:
        genuine, impostor, skipped = [], [], 0
        with open(a.csv, "w") as fh:
            fh.write("distance,same,a,b\n")
            for i, (pa, pb, same) in enumerate(pairs):
                ea, eb = emb(pa), emb(pb)
                if ea is None or eb is None:
                    skipped += 1
                else:
                    d = float(np.linalg.norm(ea - eb)) * models.distance_scale
                    (genuine if same else impostor).append(d)
                    fh.write(f"{d:.4f},{int(same)},{os.path.basename(pa)},{os.path.basename(pb)}\n")
                if (i + 1) % 500 == 0:
                    progress(i + 1, len(pairs), skipped)
        genuine, impostor = np.array(genuine), np.array(impostor)
    else:
        genuine, impostor, skipped = evaluate(emb, pairs, scale=models.distance_scale, progress=progress)

    if not len(genuine) or not len(impostor):
        print("no scorable pairs (no faces detected?) — check --dir/--ext", file=sys.stderr); return 1
    m = metrics(genuine, impostor, gates)
    if a.json:
        import json
        print(json.dumps({"metrics": m, "skipped": skipped}))
    else:
        print(format_report(m, skipped, time.monotonic() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
