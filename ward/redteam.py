"""Red-team harness: digitally attack real frames and see what the RGB layers of ward let through.

Runs NOW on recorded colour frames (no dataset, no second person needed):

    python3 -m ward.redteam --dir <dir of your recorded colour frames>

It builds a reference template from clean frames (standing in for enrolment), then throws a battery
of digital transforms at held-out frames and, for each, measures the two RGB-visible layers:
  recognised  — does the attacked frame still match the template within the gate? (recogniser)
  antispoof   — does the anti-spoof CNN still call it live (p_real ≥ PASS_P)? (liveness)
A row is a *bypass* only when BOTH hold for a spoof-style transform (screen replay, print): the frame
would be recognised AND pass the CNN. Degradation rows (blur, noise, light) instead test whether YOU
stay recognised under bad conditions — reliability, not security.

HONEST SCOPE — this is one of three layers, digitally simulated:
  • A real printed photo or phone screen also loses the IR channel, which the cross-spectrum voter
    catches (ward.ir_pad) — this harness cannot simulate that, so a "bypass" here is NOT a full
    system bypass, only evidence the RGB CNN alone is or isn't enough.
  • Impostor acceptance (a different person) needs ward.lfw or a real second face — not tested here.
  • Digital transforms approximate, but are not, physical presentation attacks.
Read it as: how robust is recognition to degradation, and how much does the anti-spoof CNN add on its
own. Keep any dataset frames out of the repo.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

IMG_EXT = (".png", ".jpg", ".jpeg")


# --- digital transforms on a BGR uint8 frame ---

def _jpeg(img, q):
    import cv2
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def _rescale(img, f):
    import cv2
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _gamma(img, g):
    lut = (np.clip((np.arange(256) / 255.0) ** g, 0, 1) * 255).astype(np.uint8)
    return lut[img]


def _noise(img, sigma, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.int16) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def _moire(img, amp=10, period=4):
    h, w = img.shape[:2]
    yy = np.sin(2 * np.pi * np.arange(h) / period)[:, None]
    return np.clip(img.astype(np.float32) + (amp * yy)[..., None], 0, 255).astype(np.uint8)


def _screen_replay(img):
    """Crude digital replay: resolution loss + compression + scanline moiré + display gamma."""
    return _gamma(_moire(_jpeg(_rescale(img, 0.5), 30), amp=8, period=3), 1.1)


def _print_sim(img):
    """Crude print: desaturate, paper grain, soft focus, warm tint."""
    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= 0.5
    desat = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return _noise(cv2.GaussianBlur(desat, (3, 3), 0), 6, seed=1)


def _blur(img, k):
    import cv2
    return cv2.GaussianBlur(img, (k, k), 0)


ATTACKS = {
    # name: (fn, is_spoof_simulation) — spoof sims are the security test; the rest are robustness
    "clean": (lambda x: x, False),
    "jpeg q20": (lambda x: _jpeg(x, 20), False),
    "jpeg q8": (lambda x: _jpeg(x, 8), False),
    "blur 5px": (lambda x: _blur(x, 5), False),
    "downscale 4x": (lambda x: _rescale(x, 0.25), False),
    "bright +50": (lambda x: np.clip(x.astype(np.int16) + 50, 0, 255).astype(np.uint8), False),
    "dark -50": (lambda x: np.clip(x.astype(np.int16) - 50, 0, 255).astype(np.uint8), False),
    "gamma 2.0": (lambda x: _gamma(x, 2.0), False),
    "noise σ25": (lambda x: _noise(x, 25), False),
    "SCREEN REPLAY": (_screen_replay, True),
    "PRINT": (_print_sim, True),
}


def load_frames(path: str):
    import cv2
    if os.path.isdir(path):
        paths = sorted(os.path.join(dp, f) for dp, _d, fs in os.walk(path) for f in fs if f.lower().endswith(IMG_EXT))
    else:
        paths = [path]
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is not None:
            yield p, img


def build_template(models, frames, n):
    """Mean of the first `n` clean embeddings (proxy for the fused enrolment template)."""
    import cv2
    embs = []
    used = []
    for p, img in frames:
        box = models.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if box is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        embs.append(models.embed(rgb, box))
        used.append((p, img))
        if len(embs) >= n:
            break
    if not embs:
        return None, []
    t = np.mean(embs, axis=0)
    return t / (np.linalg.norm(t) or 1.0), used


def run(models, anti, template, held, gate: float, pass_p: float):
    import cv2
    net = anti._load()
    rows = {}
    for name, (fn, is_spoof) in ATTACKS.items():
        dists, preals, nodet = [], [], 0
        bypass = 0
        for _p, img in held:
            att = fn(img)
            box = models.detect(cv2.cvtColor(att, cv2.COLOR_BGR2GRAY))
            if box is None:
                nodet += 1                          # the attack defeated detection: not recognised, not a bypass
                continue
            rgb = cv2.cvtColor(att, cv2.COLOR_BGR2RGB)
            d = float(np.linalg.norm(models.embed(rgb, box) - template)) * models.distance_scale
            s = anti.score(rgb, box, net)
            dists.append(d)
            if s is not None:
                preals.append(s)
            if d <= gate and s is not None and s >= pass_p:
                bypass += 1                          # recognised AND passed the CNN
        n = len(held)
        rows[name] = {
            "is_spoof": is_spoof, "n": n, "no_detect": nodet,
            "dist_mean": float(np.mean(dists)) if dists else float("nan"),
            "recognised": sum(d <= gate for d in dists) / n if n else float("nan"),
            "p_real_mean": float(np.mean(preals)) if preals else float("nan"),
            "antispoof_pass": sum(s >= pass_p for s in preals) / n if n else float("nan"),
            "bypass_rate": bypass / n if (is_spoof and n) else None,
        }
    return rows


def verdict(rows: dict) -> list[str]:
    out = []
    robust = [r for n, r in rows.items() if not r["is_spoof"] and n != "clean"]
    if robust:
        worst = min(robust, key=lambda r: r["recognised"])
        wn = next(n for n, r in rows.items() if r is worst)
        out.append(f"recognition robustness: you stay recognised on {min(r['recognised'] for r in robust) * 100:.0f}–"
                   f"{max(r['recognised'] for r in robust) * 100:.0f}% of degraded frames (worst: {wn}).")
    for name, r in rows.items():
        if r["is_spoof"]:
            br = r["bypass_rate"]
            if br is None:
                continue
            tag = "GOOD" if br <= 0.05 else "CONCERN" if br <= 0.3 else "WEAK"
            out.append(f"{name}: the RGB CNN+recogniser alone would accept {br * 100:.0f}% as live-you [{tag}] "
                       f"(a real one also loses IR → cross-spectrum; see ward.ir_pad).")
    return out


def format_report(rows, template_n, held_n, seconds) -> str:
    L = [f"Red-team: {template_n} frames as the template, {held_n} held-out frames attacked", ""]
    L.append(f"  {'transform':<16}{'recognised':>11}{'dist':>7}{'antispoof pass':>16}{'p_real':>8}{'bypass':>9}")
    for name, r in rows.items():
        bp = "" if r["bypass_rate"] is None else f"{r['bypass_rate'] * 100:>7.0f}%"
        nod = f"  ({r['no_detect']} no-detect)" if r["no_detect"] else ""
        L.append(f"  {name:<16}{r['recognised'] * 100:>10.0f}%{r['dist_mean']:>7.2f}"
                 f"{r['antispoof_pass'] * 100:>15.0f}%{r['p_real_mean']:>8.2f}{bp:>9}{nod}")
    L.append("")
    for line in verdict(rows):
        L.append("  • " + line)
    L.append("")
    L.append("  Scope: RGB recogniser + anti-spoof CNN only, on digital transforms. The IR cross-spectrum")
    L.append("  voter (catches screen replays) and impostor acceptance are NOT in this number — use")
    L.append("  ward.ir_pad and ward.lfw for those. A 'bypass' here means the RGB layers alone are not")
    L.append("  sufficient, not that the whole system is defeated.")
    L.append(f"  [{seconds:.0f}s]")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ward.redteam", description="Digitally attack real frames; measure the RGB recogniser + anti-spoof layers.")
    ap.add_argument("--dir", required=True, help="directory of recorded colour frames of ONE person")
    ap.add_argument("--template", type=int, default=0, help="frames to build the template from (default: half)")
    ap.add_argument("--gate", type=float, default=0.0, help="match gate (default: config recognition.certainty)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    if not os.path.exists(a.dir):
        print(f"no such path: {a.dir}", file=sys.stderr); return 1
    from .engine import config as cfgmod
    from .engine.detect import Models
    from .engine.liveness import antispoof_cnn as anti
    rc = cfgmod.load()["recognition"]
    gate = a.gate or float(rc["certainty"])
    models = Models(rc["detector"], rc["max_height"], "cpu", rc.get("recogniser", "dlib"))

    frames = list(load_frames(a.dir))
    if len(frames) < 4:
        print(f"need at least 4 readable frames in {a.dir} (found {len(frames)})", file=sys.stderr); return 1
    n_tmpl = a.template or max(2, len(frames) // 2)
    t0 = time.monotonic()
    template, used = build_template(models, iter(frames), n_tmpl)
    if template is None:
        print("no face detected in any frame — is this a face dataset?", file=sys.stderr); return 1
    used_paths = {p for p, _ in used}
    held = [(p, img) for p, img in frames if p not in used_paths] or frames[len(used):]
    if not held:
        held = frames
    rows = run(models, anti, template, held, gate, anti.PASS_P)
    if a.json:
        import json
        print(json.dumps({"rows": rows, "template_n": len(used), "held_n": len(held), "gate": gate}))
    else:
        print(format_report(rows, len(used), len(held), time.monotonic() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
