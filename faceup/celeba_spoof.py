"""CelebA-Spoof benchmark: APCER / BPCER / ACER for the anti-spoof CNN voter. Never on the auth path.

    python3 -m faceup.celeba_spoof --dir ~/CelebA_Spoof/Data/test
    python3 -m faceup.celeba_spoof --dir ~/CelebA_Spoof/Data/test --limit 4000 --labels ~/CelebA_Spoof/metas/intra_test/test_label.json

What it measures (ISO/IEC 30107-3 presentation-attack metrics):
  APCER — attack presentations classified as bona fide (a SPOOF the voter would call live; lower safer)
  BPCER — bona fide presentations classified as attack (a REAL face the voter would reject; lower smoother)
  ACER  — (APCER + BPCER) / 2 at a threshold.
It scores the SAME `antispoof_cnn.score()` the engine uses, on the engine's own detector box, so the
p_real values and the operating point are exactly the deployed voter's — on RGB stills only. It does
NOT exercise the multi-frame median/sticky-FAIL logic of `decide()` (that needs a live frame stream),
the IR/cross-spectrum voter, or the recogniser. Read it as the CNN voter's standalone quality.

Binary live/spoof comes from the `/live/` vs `/spoof/` folder segment (CelebA-Spoof's own layout).
A `--labels …_label.json` adds a per-attack-type APCER breakdown. Dataset is research-only; keep it
out of the repo.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

IMG_EXT = (".png", ".jpg", ".jpeg")
# CelebA-Spoof "Spoof Type" annotation (index 40 of the 43-value label list)
SPOOF_TYPES = {0: "live", 1: "photo", 2: "poster", 3: "a4", 4: "face-mask", 5: "upper-body-mask",
               6: "region-mask", 7: "pc-screen", 8: "pad-screen", 9: "phone-screen", 10: "3d-mask"}
SPOOF_TYPE_INDEX = 40


def collect_images(root: str, labels: dict | None = None) -> list[tuple[str, bool, str]]:
    """(path, is_spoof, type_name) for every image under `root`. is_spoof from the folder segment;
    type_name from `labels` (relative-path → 43-value list) when given, else 'live'/'spoof'."""
    out = []
    for dirpath, _dirs, files in os.walk(root):
        seg = dirpath.replace("\\", "/").lower().split("/")
        is_spoof = "spoof" in seg and "live" not in seg
        for fn in files:
            if not fn.lower().endswith(IMG_EXT) or fn.endswith("_BB.txt"):
                continue
            path = os.path.join(dirpath, fn)
            kind = ("spoof" if is_spoof else "live")
            if labels is not None:
                v = _label_for(path, labels)
                if v is not None and 0 <= SPOOF_TYPE_INDEX < len(v):
                    kind = SPOOF_TYPES.get(int(v[SPOOF_TYPE_INDEX]), kind)
            out.append((path, is_spoof, kind))
    return out


def _label_for(path: str, labels: dict) -> list | None:
    """Match an absolute image path against a labels dict keyed by dataset-relative paths."""
    p = path.replace("\\", "/")
    if p in labels:
        return labels[p]
    for n in range(1, 6):                                   # try the last n path components
        suffix = "/".join(p.split("/")[-n:])
        for key in (suffix, "Data/" + suffix):
            if key in labels:
                return labels[key]
    return None


def balanced_subset(items, limit: int, seed: int = 0):
    """At most `limit` images, split evenly live/spoof, deterministic."""
    if limit <= 0 or limit >= len(items):
        return items
    import random
    rng = random.Random(seed)
    live = [x for x in items if not x[1]]
    spoof = [x for x in items if x[1]]
    rng.shuffle(live); rng.shuffle(spoof)
    k = limit // 2
    out = live[:k] + spoof[:k]
    rng.shuffle(out)
    return out


def read_bb(path: str, w: int, h: int):
    """CelebA-Spoof `<img>_BB.txt` face box (coords are for a 224×224 image → scale to w×h)."""
    bb = os.path.splitext(path)[0] + "_BB.txt"
    try:
        with open(bb) as f:
            x, y, bw, bh = (float(v) for v in f.read().split()[:4])
    except (OSError, ValueError):
        return None
    sx, sy = w / 224.0, h / 224.0
    return int(x * sx), int(y * sy), int(bw * sx), int(bh * sy)


class Scorer:
    """Image path → p_real from the engine's anti-spoof voter, using the engine's detector box
    (or the dataset bounding box with --use-bb). None when no face/score is available."""

    def __init__(self, models, antispoof, use_bb: bool = False):
        self.models = models
        self.anti = antispoof
        self.use_bb = use_bb
        self.net = antispoof._load()
        self.no_face = 0

    def __call__(self, path: str) -> float | None:
        import cv2
        from .engine.detect import Box
        img = cv2.imread(path, cv2.IMREAD_COLOR)            # BGR
        if img is None:
            self.no_face += 1
            return None
        box = None
        if self.use_bb:
            bb = read_bb(path, img.shape[1], img.shape[0])
            if bb:
                box = Box(*bb)
        if box is None:
            box = self.models.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if box is None:
            self.no_face += 1
            return None
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        s = self.anti.score(rgb, box, self.net)
        if s is None:
            self.no_face += 1
        return s


def evaluate(score_fn, items, progress=None):
    """Returns (live_scores, spoof_scores, by_type{name:[scores]}, skipped)."""
    live, spoof, by_type, skipped = [], [], {}, 0
    for i, (path, is_spoof, kind) in enumerate(items):
        s = score_fn(path)
        if s is None:
            skipped += 1
        else:
            (spoof if is_spoof else live).append(s)
            by_type.setdefault(kind, []).append(s)
        if progress and (i + 1) % 500 == 0:
            progress(i + 1, len(items), skipped)
    return (np.array(live), np.array(spoof),
            {k: np.array(v) for k, v in by_type.items()}, skipped)


# --- metrics (predict LIVE when p_real ≥ t; predict ATTACK otherwise) ---

def apcer(spoof: np.ndarray, t: float) -> float:
    """Attacks classified as bona fide: spoof frames scoring live (p_real ≥ t)."""
    return float(np.mean(spoof >= t)) if len(spoof) else float("nan")


def bpcer(live: np.ndarray, t: float) -> float:
    """Bona fide classified as attack: real frames scoring not-live (p_real < t)."""
    return float(np.mean(live < t)) if len(live) else float("nan")


def eer(live: np.ndarray, spoof: np.ndarray) -> tuple[float, float]:
    """(threshold, rate) where APCER ≈ BPCER."""
    if not len(live) or not len(spoof):
        return float("nan"), float("nan")
    ts = np.unique(np.concatenate([live, spoof, [0.0, 1.0]]))
    best = (ts[0], float("inf"), float("nan"))
    for t in ts:
        a, b = apcer(spoof, t), bpcer(live, t)
        if abs(a - b) < best[1]:
            best = (float(t), abs(a - b), (a + b) / 2)
    return best[0], best[2]


def auc(live: np.ndarray, spoof: np.ndarray) -> float:
    """P(live p_real > spoof p_real): 1.0 = perfect separation."""
    if not len(live) or not len(spoof):
        return float("nan")
    s = np.sort(spoof)
    below = np.searchsorted(s, live, side="left")
    ties = np.searchsorted(s, live, side="right") - below
    return float(np.sum(below + 0.5 * ties) / (len(live) * len(spoof)))


def metrics(live, spoof, by_type, thresholds: dict[str, float]) -> dict:
    et, er = eer(live, spoof)
    at = {}
    for name, t in thresholds.items():
        a, b = apcer(spoof, t), bpcer(live, t)
        at[name] = {"threshold": t, "apcer": a, "bpcer": b, "acer": (a + b) / 2}
    # per-attack-type APCER at the PASS threshold (how often each PAI species would be called live)
    pass_t = thresholds.get("PASS_P (live vote)")
    per_type = ({k: {"n": int(len(v)), "apcer": apcer(v, pass_t)}
                 for k, v in sorted(by_type.items()) if k not in ("live",)}
                if pass_t is not None else {})
    return {
        "n_live": int(len(live)), "n_spoof": int(len(spoof)),
        "live_p_real_mean": _f(np.mean(live)), "live_p_real_p5": _pct(live, 5),
        "spoof_p_real_mean": _f(np.mean(spoof)), "spoof_p_real_p95": _pct(spoof, 95),
        "auc": auc(live, spoof), "eer": er, "eer_threshold": et,
        "at": at, "per_type_apcer": per_type,
    }


def _f(x) -> float:
    a = np.atleast_1d(x)
    return float(x) if len(a) and np.isfinite(x) else float("nan")


def _pct(a: np.ndarray, p: float) -> float:
    return float(np.percentile(a, p)) if len(a) else float("nan")


def format_report(m: dict, skipped: int, seconds: float) -> str:
    L = ["CelebA-Spoof anti-spoof benchmark (CNN voter only — RGB stills, single frame)", ""]
    L.append(f"  images scored : {m['n_live']} live, {m['n_spoof']} spoof"
             + (f"  ({skipped} skipped: no face / unreadable)" if skipped else ""))
    L.append(f"  live  p_real  : mean {m['live_p_real_mean']:.3f}, p5  {m['live_p_real_p5']:.3f}  (want high)")
    L.append(f"  spoof p_real  : mean {m['spoof_p_real_mean']:.3f}, p95 {m['spoof_p_real_p95']:.3f}  (want low)")
    L.append(f"  separation    : AUC {m['auc']:.4f}   EER {m['eer'] * 100:.2f}% at p_real {m['eer_threshold']:.3f}")
    L.append("")
    L.append(f"    {'operating point':<22} {'thresh':>7} {'APCER':>9} {'BPCER':>9} {'ACER':>9}")
    for name, r in m["at"].items():
        L.append(f"    {name:<22} {r['threshold']:>7.2f} {r['apcer'] * 100:>8.2f}% {r['bpcer'] * 100:>8.2f}% {r['acer'] * 100:>8.2f}%")
    if m["per_type_apcer"]:
        L.append("")
        L.append("  APCER by attack type at the live-vote threshold (spoofs the voter would call live):")
        for k, r in m["per_type_apcer"].items():
            L.append(f"    {k:<18} {r['apcer'] * 100:>7.2f}%  (n={r['n']})")
    L.append("")
    L.append("  APCER = spoofs the voter would pass as live (lower safer); BPCER = real faces it would")
    L.append("  not pass (they fall back to the other voters or the password). faceup only lets this voter")
    L.append("  FAIL an attempt when several frames agree ≤ 0.02, so the deployed false-reject is milder.")
    L.append(f"  [{seconds:.0f}s]")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="faceup.celeba_spoof", description="CelebA-Spoof APCER/BPCER for the faceup anti-spoof voter.")
    ap.add_argument("--dir", required=True, help="dataset split root (walks live/ and spoof/ folders)")
    ap.add_argument("--labels", help="…_label.json for per-attack-type APCER")
    ap.add_argument("--limit", type=int, default=0, help="score at most this many images (balanced, deterministic)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-bb", action="store_true", help="use the dataset _BB.txt box instead of faceup's detector")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    from .engine import config as cfgmod
    from .engine.detect import Models
    from .engine.liveness import antispoof_cnn as anti
    if not anti.available():
        print("anti-spoof model not installed", file=sys.stderr); return 1
    ok, _ = anti.verify()
    if not ok:
        print("anti-spoof model checksum mismatch — refusing to benchmark a tampered model", file=sys.stderr); return 1

    if not os.path.isdir(a.dir):
        print(f"not a directory: {a.dir}\n  download CelebA-Spoof (research-only) and point --dir at e.g. CelebA_Spoof/Data/test",
              file=sys.stderr); return 1
    labels = None
    if a.labels:
        import json
        try:
            with open(a.labels) as f:
                labels = json.load(f)
        except (OSError, ValueError) as e:
            print(f"cannot read --labels {a.labels}: {e}\n  (it's optional — omit it to skip the per-attack-type breakdown)",
                  file=sys.stderr); return 1
    items = collect_images(a.dir, labels)
    if not items:
        print(f"no images under {a.dir} — expected live/ and spoof/ subfolders (CelebA-Spoof layout)", file=sys.stderr); return 1
    items = balanced_subset(items, a.limit, a.seed)

    rc = cfgmod.load()["recognition"]
    models = Models(rc["detector"], rc["max_height"], "cpu", rc.get("recogniser", "dlib"))
    scorer = Scorer(models, anti, use_bb=a.use_bb)
    t0 = time.monotonic()

    def progress(done, total, skipped):
        print(f"  … {done}/{total} images ({skipped} skipped)", file=sys.stderr)

    live, spoof, by_type, skipped = evaluate(scorer, items, progress)
    if not len(live) or not len(spoof):
        print("need both live and spoof images with detectable faces — check --dir layout", file=sys.stderr); return 1
    thresholds = {"PASS_P (live vote)": anti.PASS_P, "FAIL_P (spoof vote)": anti.FAIL_P, "mid 0.50": 0.5}
    m = metrics(live, spoof, by_type, thresholds)
    if a.json:
        import json
        print(json.dumps({"metrics": m, "skipped": skipped}))
    else:
        print(format_report(m, skipped, time.monotonic() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
