"""IR presentation-attack benchmark: does the cross-spectrum voter catch attacks? Not on the auth path.

faceup's IR-specific liveness is `cross_spectrum`: the face must appear in BOTH the IR and the RGB
frame at the same place. A phone/monitor emits ~no IR, so a screen replay has no face in IR and is
caught; an IR-tuned print can still show in both. This harness runs that exact voter over a PAD
dataset that provides paired colour + IR (CASIA-SURF, WMCA, CASIA-SURF CeFA — all access-gated).

    # CASIA-SURF file list (columns: color depth ir label — label 1=live, 0=spoof):
    python3 -m faceup.ir_pad --root ~/CASIA-SURF --list ~/CASIA-SURF/test_private_list.txt
    # or a folder tree with matching colour/ir files split into live/ and spoof/:
    python3 -m faceup.ir_pad --color ~/set/color --ir ~/set/ir

It reports, per class, how the voter decided (PASS / FAIL / ABSTAIN):
  live  → should PASS;  a FAIL is a false reject (BPCER-like)
  spoof → should FAIL;  a PASS is a missed attack (APCER-like)
Dataset colour+IR are assumed aligned (same resolution), so calibration is identity. This tests ONE
voter: the recogniser and the RGB anti-spoof CNN (faceup.celeba_spoof) are separate layers. Datasets
are research-only; keep them out of the repo.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

IDENTITY_CALIB = {"dx": 0.0, "dy": 0.0, "s": 1.0}
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def read_list(list_file: str, root: str) -> list[tuple[str, str, bool]]:
    """CASIA-SURF-style list → (color_path, ir_path, is_spoof). Columns are whitespace-separated:
    3 = `color ir label`, 4 = `color depth ir label` (CASIA-SURF). label 1 = live, 0 = spoof."""
    out = []
    with open(list_file) as f:
        for line in f:
            p = line.split()
            if len(p) == 3:
                color, ir, label = p
            elif len(p) >= 4:
                color, _depth, ir, label = p[0], p[1], p[2], p[3]
            else:
                continue
            out.append((os.path.join(root, color), os.path.join(root, ir), int(label) == 0))
    return out


def pair_by_name(color_dir: str, ir_dir: str) -> list[tuple[str, str, bool]]:
    """Pair colour and IR images by identical relative filename; label from a live/ or spoof/
    path segment on the colour side."""
    out = []
    for dp, _d, files in os.walk(color_dir):
        for fn in files:
            if not fn.lower().endswith(IMG_EXT):
                continue
            rel = os.path.relpath(os.path.join(dp, fn), color_dir)
            ir = os.path.join(ir_dir, rel)
            if not os.path.exists(ir):
                continue
            seg = os.path.join(dp, fn).replace("\\", "/").lower().split("/")
            out.append((os.path.join(dp, fn), ir, "spoof" in seg and "live" not in seg))
    return out


def balanced_subset(items, limit: int, seed: int = 0):
    if limit <= 0 or limit >= len(items):
        return items
    import random
    rng = random.Random(seed)
    live = [x for x in items if not x[2]]
    spoof = [x for x in items if x[2]]
    rng.shuffle(live); rng.shuffle(spoof)
    k = limit // 2
    out = live[:k] + spoof[:k]
    rng.shuffle(out)
    return out


class CrossVoter:
    """One paired (colour, IR) sample → cross-spectrum Vote result string ('PASS'|'FAIL'|'ABSTAIN'),
    or None when neither image loads. Uses faceup's detector and the real `cross_spectrum.vote`."""

    def __init__(self, models, cross, dark_mean: float = 12.0):
        self.models = models
        self.cross = cross
        self.dark_mean = dark_mean
        self.unreadable = 0

    def __call__(self, color_path: str, ir_path: str) -> str | None:
        import cv2
        color = cv2.imread(color_path, cv2.IMREAD_GRAYSCALE)
        ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        if color is None or ir is None:
            self.unreadable += 1
            return None
        rgb_box = self.models.detect(color)
        ir_box = self.models.detect(ir)
        rgb_dark = float(color.mean()) < self.dark_mean
        ir_dark = float(ir.mean()) < self.dark_mean
        v = self.cross.vote(ir_box, rgb_box, ir_dark, rgb_dark, IDENTITY_CALIB,
                            (ir.shape[1], ir.shape[0]), (color.shape[1], color.shape[0]))
        return v.result.value


def evaluate(voter, items, progress=None) -> dict:
    """Confusion counts per class. Returns {'live': {PASS,FAIL,ABSTAIN}, 'spoof': {...}, 'skipped'}."""
    res = {"live": {"PASS": 0, "FAIL": 0, "ABSTAIN": 0}, "spoof": {"PASS": 0, "FAIL": 0, "ABSTAIN": 0}, "skipped": 0}
    for i, (color, ir, is_spoof) in enumerate(items):
        r = voter(color, ir)
        if r is None:
            res["skipped"] += 1
        else:
            res["spoof" if is_spoof else "live"][r] += 1
        if progress and (i + 1) % 500 == 0:
            progress(i + 1, len(items), res["skipped"])
    return res


def metrics(res: dict) -> dict:
    live, spoof = res["live"], res["spoof"]
    nl, ns = sum(live.values()), sum(spoof.values())
    return {
        "n_live": nl, "n_spoof": ns,
        "live_pass_rate": _r(live["PASS"], nl),
        "bpcer": _r(live["FAIL"], nl),                 # live wrongly flagged as attack
        "live_abstain": _r(live["ABSTAIN"], nl),
        "spoof_catch_rate": _r(spoof["FAIL"], ns),
        "apcer": _r(spoof["PASS"], ns),                # attack wrongly passed as live
        "spoof_abstain": _r(spoof["ABSTAIN"], ns),
        "counts": {"live": live, "spoof": spoof},
    }


def _r(n, d) -> float:
    return n / d if d else float("nan")


def format_report(m: dict, skipped: int, seconds: float) -> str:
    L = ["IR cross-spectrum benchmark (one liveness voter — needs paired colour + IR)", ""]
    L.append(f"  samples: {m['n_live']} live, {m['n_spoof']} spoof" + (f"  ({skipped} unreadable)" if skipped else ""))
    L.append("")
    L.append(f"  live  → PASS {m['live_pass_rate'] * 100:5.1f}%   FAIL {m['bpcer'] * 100:5.1f}% (false reject)   ABSTAIN {m['live_abstain'] * 100:5.1f}%")
    L.append(f"  spoof → FAIL {m['spoof_catch_rate'] * 100:5.1f}% (caught)   PASS {m['apcer'] * 100:5.1f}% (MISSED)   ABSTAIN {m['spoof_abstain'] * 100:5.1f}%")
    L.append("")
    L.append("  APCER here = attacks the cross-spectrum voter alone lets through. It catches screen")
    L.append("  replays (no face in IR) but not necessarily IR-reflective prints — those lean on the")
    L.append("  RGB anti-spoof CNN (faceup.celeba_spoof) and the recogniser. ABSTAIN = no IR/RGB pair to")
    L.append("  judge (e.g. a dark frame), which in deployment means this voter simply doesn't vote.")
    L.append(f"  [{seconds:.0f}s]")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="faceup.ir_pad", description="IR cross-spectrum PAD benchmark (CASIA-SURF / WMCA).")
    ap.add_argument("--root", default="", help="prefix for the paths in --list")
    ap.add_argument("--list", help="CASIA-SURF-style file list (color [depth] ir label)")
    ap.add_argument("--color", help="colour image dir (with --ir; paired by filename)")
    ap.add_argument("--ir", help="IR image dir (with --color)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    if a.list:
        if not os.path.isfile(a.list):
            print(f"not a file: {a.list}", file=sys.stderr); return 1
        items = read_list(a.list, a.root)
    elif a.color and a.ir:
        if not (os.path.isdir(a.color) and os.path.isdir(a.ir)):
            print("--color and --ir must both be directories", file=sys.stderr); return 1
        items = pair_by_name(a.color, a.ir)
    else:
        print("give --list FILE (CASIA-SURF) or --color DIR --ir DIR", file=sys.stderr); return 1
    if not items:
        print("no paired colour/IR samples found", file=sys.stderr); return 1
    items = balanced_subset(items, a.limit, a.seed)

    from .engine import config as cfgmod
    from .engine.detect import Models
    from .engine.liveness import cross_spectrum as cross
    rc = cfgmod.load()["recognition"]
    models = Models(rc["detector"], rc["max_height"], "cpu", rc.get("recogniser", "dlib"))
    voter = CrossVoter(models, cross)
    t0 = time.monotonic()

    def progress(done, total, skipped):
        print(f"  … {done}/{total} samples ({skipped} unreadable)", file=sys.stderr)

    res = evaluate(voter, items, progress)
    if not sum(res["live"].values()) or not sum(res["spoof"].values()):
        print("need both live and spoof samples — check the list/labels", file=sys.stderr); return 1
    m = metrics(res)
    if a.json:
        import json
        print(json.dumps({"metrics": m, "skipped": res["skipped"]}))
    else:
        print(format_report(m, res["skipped"], time.monotonic() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
