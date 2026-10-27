"""Threshold calibration (DESIGN.md §6.7): the gate and the whitening threshold come from
measurements of THIS user on THIS machine, never from constants — and keep following the
user's real conditions afterwards.

Two mechanisms:

1. **The session** (`ward calibrate`, and the end of enrollment). A few guided passes
   (look at the screen, at the keyboard, lean back, lean in, tilt, turn, change the light)
   while the lab evaluator measures every frame set: the fused live embedding's raw and
   whitened distance to the store, per spectrum. From those genuine distances:
       gate            = clamp(p95(raw) + MARGIN, GATE_MIN, GATE_MAX)
       hard_cap        = min(gate + 0.15, 0.55)
       early_reject    = max(0.65, hard_cap + 0.10)
       learn_max_dist  = gate − 0.10
       whitening thr   = p95(whitened) × 1.15   (per spectrum; the store's own threshold)
   The session's fused embeddings also enter the whitening scatter as real, varied samples,
   which is what the leave-one-out threshold over correlated enrollment frames lacked.
   Coverage of light states (ok / dim / bright) is recorded so the report can say which
   conditions have not been seen yet.

2. **Rolling refresh** (automatic, store-side, guarded). Every confident accept appends its
   raw and whitened distances to a ring in the store (`calib.recent`). Every REFRESH_EVERY
   accepts the whitening threshold is recomputed from the recent window — it may move by at
   most ±20 % per refresh and never below the session's floor — so a new condition (night,
   glasses) that first passed through the whitened path or online learning is absorbed
   instead of forcing a manual retune. The raw gate is config, not store: the engine never
   rewrites config; `doctor` recommends `ward calibrate refresh` when the recent window says
   the gate is off.

Guardrails on purpose: the gate can never exceed GATE_MAX (a look-alike's territory), and a
session whose p95 sits above GATE_MAX reports "re-enroll" rather than loosening.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

GATE_MIN, GATE_MAX, MARGIN = 0.33, 0.45, 0.05
WHITEN_MARGIN = 1.15
MIN_SAMPLES = 8                      # per spectrum before anything is derived
RECENT_MAX = 60                      # ring size of accepted attempts kept in the store
REFRESH_EVERY = 20                   # accepts between automatic whitening-threshold refreshes
REFRESH_STEP = 0.20                  # max relative change of the whitening threshold per refresh

# (key, text, cue, condition) — the condition is checked on the DISPLAY-space pose (gui.to_display
# applied once by the caller: display yaw > 0 = the user turned to their right as they see it in
# the mirrored preview) and on the face width relative to the "screen" pass. None = any pose.
#   cue: what the window draws — "centre" | "down" | "away" | "closer" | "tilt" | "left" | "right" | "light"
PROMPTS = [
    ("screen",   "Look at the middle of the screen, as you normally sit.",                 "centre", {"yaw": (-10, 10), "pitch": (-10, 10)}),
    ("keyboard", "Look down at the keyboard, the way you do when a sudo prompt appears.",   "down",   {"pitch": (-90, -7)}),
    ("back",     "Sit back, further from the screen.",                                      "away",   {"width": (0.0, 0.88)}),
    ("close",    "Lean in close, like reading small text.",                                 "closer", {"width": (1.12, 9.0)}),
    ("tilt",     "Tilt your head a little to one side.",                                    "tilt",   {"roll_abs": (7, 90)}),
    ("left",     "Turn your head to the left, eyes on the screen.",                         "left",   {"yaw": (-90, -9)}),
    ("right",    "Turn your head to the right.",                                            "right",  {"yaw": (9, 90)}),
    ("light",    "If you can, change the light now: lamp on or off, or shade the window.",  "light",  None),
]
# a CONDITION (glasses, headset, night…) needs only a short sweep on top of the full enrolment
NEED_FRAMES = 20                     # frame sets satisfying a pass before it is complete
PASS_TIMEOUT_S = 15.0                # then move on with what was measured

# Unified enroll+calibrate poses (roadmap: one guided app). Each is (label, cue, text, target).
# `target` is the DISPLAY-space (yaw, pitch) the pose should reach — the window guides the head
# there and derives the arrow from current→target, so left/right can never be mixed up. Display
# yaw < 0 = the user turns to THEIR left (matches the enrollment hint the user already likes).
SETUP_POSES = [
    ("C/level", "centre", "Look here",      (0.0,   0.0)),
    ("C/up",    "up",     "Look up",        (0.0,  18.0)),    # was +15 — deeper so an off-angle look-up matches
    ("C/down",  "down",   "Look down",      (0.0, -24.0)),    # was -18 — people look down at a laptop; go further
    ("L1/level","left",   "Turn left",      (-28.0, 0.0)),    # was -20 — a real turn, still below the dlib ~30-35° knee
    ("R1/level","right",  "Turn right",     (28.0,  0.0)),    # was +20
    ("C/close", "closer", "Lean in — fill the circle",   (0.0, 0.0)),
    ("C/back",  "away",   "Lean back",                        (0.0, 0.0)),
    ("C/level2","light",  "Change your light — lamp on/off, or turn to a window", (0.0, 0.0)),
]
CONDITION_POSES = [p for p in SETUP_POSES if p[1] in ("centre", "down", "left", "right")]   # glasses/headset: 4 short passes


def satisfied(cond: dict | None, yaw, pitch, roll, face_w, baseline_w) -> bool | None:
    """Does this frame meet the pass's condition? None = pose unknown (counts as satisfied so the
    text mode and cameras without a pose estimate still finish)."""
    if cond is None:
        return True
    for key, (lo, hi) in cond.items():
        if key == "width":
            if face_w is None or not baseline_w:
                return None
            v = face_w / baseline_w
        elif key == "roll_abs":
            if roll is None:
                return None
            v = abs(roll)
        else:
            v = {"yaw": yaw, "pitch": pitch}[key]
            if v is None:
                return None
        if not (lo <= v <= hi):
            return False
    return True


class Collector:
    """Walks the passes; `feed()` gets one lab sample (+ fused embeddings) per frame set and
    returns the pass state the UI shows. A pass completes after NEED_FRAMES satisfying frames
    or PASS_TIMEOUT_S; every frame with a face is measured regardless (transitions are genuine
    data too). `skip()` ends the current pass early."""

    def __init__(self, prompts=PROMPTS, need: int = NEED_FRAMES, timeout: float = PASS_TIMEOUT_S, sgn: float = -1.0):
        import time as _t
        self._t = _t
        self.prompts = list(prompts)
        self.need, self.timeout, self.sgn = need, timeout, sgn
        self.i = 0
        self.hits = 0
        self.started = None
        self.baseline_w: float | None = None
        self._widths: list[int] = []
        self.measures: list[Measure] = []
        self.finished = False
        self.last_ok: bool | None = None

    @property
    def current(self):
        return self.prompts[self.i] if self.i < len(self.prompts) else None

    def start_pass(self) -> None:
        self.started = self._t.monotonic(); self.hits = 0; self.last_ok = None

    def skip(self) -> None:
        self._advance()

    def _advance(self) -> None:
        if self.current and self.current[0] == "screen" and self._widths:
            self.baseline_w = float(np.median(self._widths))
        self.i += 1
        if self.i >= len(self.prompts):
            self.finished = True
        else:
            self.start_pass()

    def display_pose(self, smp):
        """(yaw, pitch, roll) in display space, or None."""
        yaw = getattr(smp, "yaw", None)
        if smp is None or yaw is None:
            return None
        roll = getattr(smp, "roll", None)
        return (self.sgn * yaw, getattr(smp, "pitch", 0.0), self.sgn * (roll if roll is not None else 0.0))

    def feed(self, smp, fused: dict | None = None) -> dict:
        if self.finished or self.current is None:
            return {"finished": True}
        if self.started is None:
            self.start_pass()
        key, text, cue, cond = self.current
        if smp is not None and any(smp.raw.get(sp) is not None for sp in ("ir", "rgb")):
            for sp in ("ir", "rgb"):
                if smp.raw.get(sp) is None:
                    continue
                self.measures.append(Measure(key, sp, float(smp.raw[sp]), (getattr(smp, "wdist", None) or {}).get(sp),
                                             (getattr(smp, "light", None) or {}).get(sp, ""), (fused or {}).get(sp),
                                             (getattr(smp, "label", None) or {}).get(sp, "")))
            if key == "screen" and getattr(smp, "face_w", None):
                self._widths.append(smp.face_w)
            dp = self.display_pose(smp)
            ok = satisfied(cond, dp[0] if dp else None, dp[1] if dp else None, dp[2] if dp else None,
                           getattr(smp, "face_w", None), self.baseline_w)
            self.last_ok = ok
            if ok is None or ok:
                self.hits += 1
        else:
            self.last_ok = None
        elapsed = self._t.monotonic() - self.started
        done = self.hits >= self.need or elapsed >= self.timeout
        state = {"finished": False, "index": self.i, "total": len(self.prompts), "key": key, "text": text, "cue": cue,
                 "hits": self.hits, "need": self.need, "progress": min(1.0, self.hits / self.need),
                 "ok": self.last_ok, "elapsed": elapsed, "timeout": self.timeout, "timed_out": elapsed >= self.timeout and self.hits < self.need}
        if done:
            self._advance()
            state["pass_done"] = True
            state["finished"] = self.finished
        return state


@dataclass
class Measure:
    """One frame set of the session: what the engine would have compared."""
    prompt: str
    spectrum: str
    raw: float
    wdist: float | None
    light: str
    fused: np.ndarray | None = None
    label: str = ""


@dataclass
class Result:
    n: dict = field(default_factory=dict)             # spectrum -> samples
    p50: dict = field(default_factory=dict)
    p95: dict = field(default_factory=dict)
    max: dict = field(default_factory=dict)
    w_p95: dict = field(default_factory=dict)
    gate: dict = field(default_factory=dict)          # spectrum -> derived gate
    w_threshold: dict = field(default_factory=dict)   # spectrum -> derived whitening threshold
    conditions: dict = field(default_factory=dict)    # spectrum -> {light state: count}
    prompts_with_face: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def ok(self) -> bool:
        return bool(self.gate)


def percentile(values, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def derive_gate(raws) -> tuple[float, str]:
    """(gate, note). p95 + margin, clamped; a note when the clamp bit."""
    p95 = percentile(raws, 95)
    want = p95 + MARGIN
    if want > GATE_MAX:
        return GATE_MAX, f"p95 {p95:.2f} + margin exceeds {GATE_MAX}: gate capped — expect rejects in the worst light; re-enroll with more poses/conditions"
    if want < GATE_MIN:
        return GATE_MIN, ""
    return round(want, 3), ""


def derive_whitening_threshold(wds) -> float | None:
    wds = [w for w in wds if w is not None and np.isfinite(w)]
    if len(wds) < MIN_SAMPLES:
        return None
    return round(percentile(wds, 95) * WHITEN_MARGIN, 3)


def derived_config(gate_ir: float | None, gate_rgb: float | None) -> dict:
    """The recognition/behaviour keys implied by the gates (dlib units; ArcFace is scaled to them)."""
    g = gate_ir if gate_ir is not None else gate_rgb
    if g is None:
        return {}
    hard_cap = round(min(g + 0.15, 0.55), 3)
    out = {"recognition": {"certainty": g, "hard_cap": hard_cap, "early_reject": round(max(0.65, hard_cap + 0.10), 3)},
           "behaviour": {"learn_max_dist": round(max(0.20, g - 0.10), 3)}}
    if gate_rgb is not None:
        out["recognition"]["rgb_certainty"] = gate_rgb
    return out


def summarise(measures: list[Measure]) -> Result:
    res = Result()
    by: dict[str, list[Measure]] = {}
    for m in measures:
        by.setdefault(m.spectrum, []).append(m)
        res.prompts_with_face.setdefault(m.prompt, set()).add(m.spectrum)
    for sp, lst in by.items():
        raws = [m.raw for m in lst if m.raw is not None and np.isfinite(m.raw)]
        res.n[sp] = len(raws)
        res.conditions[sp] = {}
        for m in lst:
            res.conditions[sp][m.light or "?"] = res.conditions[sp].get(m.light or "?", 0) + 1
        if len(raws) < MIN_SAMPLES:
            res.warnings.append(f"{sp}: only {len(raws)} frame sets with a face — not enough to calibrate (need {MIN_SAMPLES})")
            continue
        res.p50[sp], res.p95[sp], res.max[sp] = percentile(raws, 50), percentile(raws, 95), max(raws)
        gate, note = derive_gate(raws)
        res.gate[sp] = gate
        if note:
            res.warnings.append(f"{sp}: {note}")
        thr = derive_whitening_threshold([m.wdist for m in lst])
        if thr is not None:
            res.w_threshold[sp] = thr
            res.w_p95[sp] = percentile([m.wdist for m in lst if m.wdist is not None], 95)
    seen = set().union(*res.prompts_with_face.values()) if res.prompts_with_face else set()
    missing = [p[0] for p in PROMPTS if p[0] not in res.prompts_with_face]
    if missing:
        res.warnings.append("no face during: " + ", ".join(missing))
    for sp, cond in res.conditions.items():
        if "dim" not in cond and "bright" not in cond:
            res.warnings.append(f"{sp}: only one light condition seen ({', '.join(cond)}); night/bright rely on learning until they have been seen (an IR camera covers the dark on its own)")
    return res


def apply(st, cfg: dict, res: Result, measures: list[Measure], save=True) -> dict:
    """Write the derived thresholds: gates into cfg (returned as the changed keys), the whitening
    threshold + real samples into the store, and the evidence into store.calib['calibration']."""
    changed: dict = {}
    if not res.ok():
        return changed
    new = derived_config(res.gate.get("ir"), res.gate.get("rgb"))
    for section, keys in new.items():
        for k, v in keys.items():
            if cfg[section].get(k) != v:
                changed[f"{section}.{k}"] = (cfg[section].get(k), v)
            cfg[section][k] = v
    # real varied samples into the whitening scatter, then the session's threshold
    for sp in res.gate:
        extra = [(m.label or "session", m.fused) for m in measures if m.spectrum == sp and m.fused is not None]
        try:
            st.fit_whitening(sp, extra=extra)
        except Exception:
            pass
        if sp in res.w_threshold and st.whitening.get(sp):
            st.whitening[sp]["threshold"] = float(res.w_threshold[sp])
            st.whitening[sp]["floor"] = float(res.w_threshold[sp])       # refresh never goes below the session
    st.calib["calibration"] = {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"), "n": res.n,
        "raw_p50": {k: round(v, 3) for k, v in res.p50.items()}, "raw_p95": {k: round(v, 3) for k, v in res.p95.items()},
        "gate": res.gate, "w_threshold": res.w_threshold, "conditions": res.conditions,
        "prompts": {p: sorted(s) for p, s in res.prompts_with_face.items()},
    }
    st.calib["recent"] = {}
    st.calib["accepts_since_refresh"] = 0
    if save:
        st.save()
    return changed


# -- rolling adaptation --------------------------------------------------------------------------
def record_accept(st, dists: dict, wdist: dict, light: dict | None = None) -> bool:
    """Append a confident accept's distances to the store's ring; True when a refresh happened."""
    recent = st.calib.setdefault("recent", {})
    now = time.time()
    for sp, d in dists.items():
        if d is None:
            continue
        w = wdist.get(sp, (None, None))[0] if isinstance(wdist.get(sp), tuple) else wdist.get(sp)
        entry = [round(now), round(float(d), 4), (round(float(w), 4) if w is not None else None), (light or {}).get(sp, "")]
        lst = recent.setdefault(sp, [])
        lst.append(entry)
        del lst[:-RECENT_MAX]
    st.calib["accepts_since_refresh"] = int(st.calib.get("accepts_since_refresh", 0)) + 1
    if st.calib["accepts_since_refresh"] >= REFRESH_EVERY:
        refresh(st)
        st.calib["accepts_since_refresh"] = 0
        return True
    return False


def refresh(st) -> dict:
    """Recompute each spectrum's whitening threshold from the recent accepts, bounded by
    ±REFRESH_STEP per call and by the session floor. Returns {spectrum: (old, new)}."""
    out = {}
    for sp, lst in (st.calib.get("recent") or {}).items():
        ent = st.whitening.get(sp)
        if not ent:
            continue
        wds = [e[2] for e in lst if len(e) > 2 and e[2] is not None]
        new = derive_whitening_threshold(wds)
        if new is None:
            continue
        old = float(ent["threshold"])
        lo, hi = old * (1 - REFRESH_STEP), old * (1 + REFRESH_STEP)
        new = min(max(new, lo), hi)
        floor = float(ent.get("floor", 0.0))
        new = max(new, floor)
        if abs(new - old) > 1e-6:
            ent["threshold"] = round(new, 4)
            out[sp] = (old, ent["threshold"])
    if out:
        st.calib["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return out


POSE_TOLERANCE_DEG = 8.0            # arrival band per axis: a frame belongs to a pose only once the head is this close
DIRECTIONAL_CUES = ("centre", "up", "down", "left", "right")


NEED_SAMPLES = 12                    # usable (pose-true, lit) embeddings a setup pass collects: ≤6 model + 4 held out
LIGHT_PASS_MIN_S = 4.0               # the "change the light" pass stays open long enough to actually do it


def pose_tolerance(target) -> tuple[float, float]:
    """Flat per-axis arrival band around a display-space target. With the deeper SETUP_POSES
    targets a resting head can no longer satisfy an angular pose (frontal vs ±28 yaw = 28° >> 8;
    level pitch ≈ -4 vs +18 = 22° >> 8), so the old 0.6*|target| guard — which let a 15° look-up
    be satisfied by a level head, and let the user stop at ~40 % of the turn — is no longer needed.
    8° sits above the pose estimator's own ~±4° noise, and stays above the neutral head's pitch
    (≈ -4°) so the centre pose still accepts a level head (6° would make it flaky)."""
    return (POSE_TOLERANCE_DEG, POSE_TOLERANCE_DEG)


def pose_ok(pose, target) -> bool:
    """The one predicate for 'this display pose is at the target' (window progress and model selection)."""
    if pose is None:
        return False
    ty, tp = pose_tolerance(target)
    return abs(pose[0] - target[0]) <= ty and abs(pose[1] - target[1]) <= tp


def usable_rows(rows: list, target, cue: str) -> list:
    """The frames that may enrol this pose: not dark, and (directional poses) within
    POSE_TOLERANCE_DEG of the target when a pose was measured. Arrival order is kept."""
    rows = [r for r in rows if not (len(r) > 3 and r[3] == "dark")]   # a camera blind in this light enrols nothing
    posed = [r for r in rows if len(r) > 4 and r[4] is not None]
    if cue in DIRECTIONAL_CUES and posed:
        rows = [r for r in posed if pose_ok(r[4], target)]
    return rows


def select_rows(rows: list, target, cue: str, samples_per_cell: int) -> tuple[list, list] | None:
    """Split one pose's captured frames into (model rows, held-out rows), or None if too few.

    rows = (quality, embedding, aug, light[, (yaw, pitch) display pose]) in ARRIVAL order.
    - Pose-true: for a directional pose, only frames whose measured pose is within
      POSE_TOLERANCE_DEG of the target count — frames taken while still turning toward it would
      enrol a "left" model from a half-turned head (review A1).
    - Honest holdout: the LAST third in time is held out, so the gate is measured on frames a
      few seconds after the model's, not interleaved with them (interleaved p95 0.20 vs temporal
      0.25 on the owner's samples, review A2). The model takes the best-quality of the rest."""
    rows = usable_rows(rows, target, cue)
    if len(rows) < 3:
        return None
    n_held = len(rows) // 3 if len(rows) >= 6 else 0
    cand, held = (rows[:-n_held], rows[-n_held:]) if n_held else (rows, [])
    # quality first, then SPREAD IN TIME: the top-quality frames cluster in one moment, and 3
    # consecutive frames gave a held-out p95 of 0.34 vs ~0.23 for 6 spread over the pass (review A9)
    keep = max(3, len(cand) - len(cand) // 3)
    best = sorted(range(len(cand)), key=lambda i: cand[i][0], reverse=True)[:keep]
    best.sort()                                              # back to arrival order
    k = max(3, min(samples_per_cell, len(best)))
    pick = [best[round(j * (len(best) - 1) / (k - 1))] for j in range(k)] if k > 1 else best[:1]
    return [cand[i] for i in dict.fromkeys(pick)], held


OTHER_FACE_DIST = 0.6        # dlib units: the same person sits ≤ ~0.45 from their own median, others ≥ ~0.6


def drop_other_faces(caps: dict, limit: float = OTHER_FACE_DIST, scale: float = 1.0) -> dict:
    """Remove frames of someone else (YuNet tracks the LARGEST face: a person leaning into view
    takes over the box). Per camera, frames farther than `limit` from the median embedding of the
    whole sweep are dropped; the owner dominates a sweep, so the median is theirs (review A14)."""
    from collections import defaultdict
    by_sp = defaultdict(list)
    for i in caps:
        for sp, rows in caps[i].items():
            by_sp[sp].extend(r[1] for r in rows)
    med = {}
    for sp, v in by_sp.items():
        if len(v) < 6:
            continue
        m = np.median(np.array(v), axis=0)
        far = sum(float(np.linalg.norm(np.asarray(e) - m)) * scale > limit for e in v)
        if far <= 0.4 * len(v):              # a clear majority defines the owner; otherwise don't guess
            med[sp] = m
    out = {}
    for i in caps:
        out[i] = defaultdict(list)
        for sp, rows in caps[i].items():
            m = med.get(sp)
            out[i][sp] = [r for r in rows if m is None or float(np.linalg.norm(np.asarray(r[1]) - m)) * scale <= limit]
    return out


def build_from_captures(st, caps: dict, poses: list, samples_per_cell: int, fuse,
                        identity: str = "", condition: str = "") -> tuple[int, dict, dict]:
    """Turn per-pose captured frames into enrolled models, holding out frames to measure the gate
    honestly (roadmap: fused setup). `caps[pose_index][spectrum]` = list of rows in arrival order
    (see `select_rows`). Returns (models_added, dist_by_spectrum, extra_for_whitening).
    dist_by_spectrum[sp] = [(raw, whitened, light)] measured on the held-out frames against the
    finished store."""
    from collections import defaultdict
    holdout = defaultdict(list); extra = defaultdict(list); added = 0
    caps = drop_other_faces(caps, scale=float(getattr(st, "scale", 1.0) or 1.0))   # limit is in dlib units
    splits = {(i, sp): select_rows(rows, poses[i][3], poses[i][1], samples_per_cell)
              for i in caps for sp, rows in caps[i].items()}
    # clear each target label ONCE up front (only THIS person's, THIS condition's) — and only labels
    # that produced a new model: a skipped or timed-out pose keeps its old models
    for label in {poses[i][0].replace("2", "") for (i, _sp), sp_split in splits.items() if sp_split is not None}:
        st.remove_label(label, identity=identity, condition=condition)
    for pose_i in sorted(caps):
        label = poses[pose_i][0].replace("2", "")            # a "…2" pass folds into its base label
        for sp in caps[pose_i]:
            split = splits[(pose_i, sp)]
            if split is None:
                continue
            model_rows, held = split
            embs = [r[1] for r in model_rows]
            st.add(label, sp, fuse(embs), float(np.mean([r[0] for r in model_rows])), poses[pose_i][3], samples=embs,
                   identity=identity, condition=condition)
            added += 1
            for r in model_rows:
                extra[sp].extend((label, a) for a in (r[2] or []))
            for r in held:
                holdout[sp].append((r[1], r[3] if len(r) > 3 else ""))
    for sp in st.spectra():
        try:
            st.fit_whitening(sp, extra.get(sp))
        except Exception:
            pass
    dist_by = defaultdict(list)
    for sp, rows in holdout.items():
        if not st.by_spectrum(sp):
            continue
        for e, lt in rows:
            d, _ = st.min_distance(e, sp)
            wd, _thr = st.wdist(e, sp)
            dist_by[sp].append((d, wd, lt))
    return added, dict(dist_by), dict(extra)


def derive_from_holdout(dist_by_spectrum: dict) -> Result:
    """Gate + whitening threshold from HELD-OUT frames (roadmap: unified setup). `dist_by_spectrum`
    maps spectrum -> list of (raw, whitened, light) tuples measured on frames that did NOT go into
    the model — i.e. genuine fresh-frame distances, the same situation as a live unlock. This
    replaces measuring against a store the frame helped build (self-match ≈ 0) or transition frames
    (which inflate p95)."""
    res = Result()
    for sp, rows in dist_by_spectrum.items():
        raws = [r[0] for r in rows if r[0] is not None and np.isfinite(r[0])]
        res.n[sp] = len(raws)
        res.conditions[sp] = {}
        for r in rows:
            lt = r[2] if len(r) > 2 else ""
            res.conditions[sp][lt or "?"] = res.conditions[sp].get(lt or "?", 0) + 1
        if len(raws) < MIN_SAMPLES:
            res.warnings.append(f"{sp}: only {len(raws)} held-out frames — not enough to calibrate")
            continue
        res.p50[sp], res.p95[sp], res.max[sp] = percentile(raws, 50), percentile(raws, 95), max(raws)
        gate, note = derive_gate(raws)
        res.gate[sp] = gate
        if note:
            res.warnings.append(f"{sp}: {note}")
        thr = derive_whitening_threshold([r[1] for r in rows])
        if thr is not None:
            res.w_threshold[sp] = thr
            res.w_p95[sp] = percentile([r[1] for r in rows if r[1] is not None], 95)
    for sp, cond in res.conditions.items():
        if sp == "rgb" and "dim" not in cond and "bright" not in cond and "dark" not in cond:
            res.warnings.append("rgb: registered in good light only — in a dark room the IR camera carries the unlock; learning adapts to the rest")
    return res


def gate_advice(st, cfg: dict) -> str | None:
    """For doctor: does the recent window disagree with the configured gate?"""
    recent = st.calib.get("recent") or {}
    raws = [e[1] for e in recent.get("ir", []) if len(e) > 1 and e[1] is not None]
    if len(raws) < MIN_SAMPLES:
        return None
    gate = float(cfg["recognition"]["certainty"])
    p95 = percentile(raws, 95)
    if p95 + MARGIN < gate - 0.08:
        return f"recent accepts sit at p95 {p95:.2f}: the gate {gate} is looser than needed — `sudo ward calibrate refresh` would tighten it to {max(GATE_MIN, round(p95 + MARGIN, 2))}"
    if p95 + MARGIN > gate + 0.03:
        return f"recent accepts reach p95 {p95:.2f}, close to the gate {gate}: expect rejects in worse light — `sudo ward calibrate` (session) or add conditions"
    return None


# -- the session itself ---------------------------------------------------------------------------
def collect(ev, seconds: float = PASS_TIMEOUT_S, prompts=PROMPTS, on_prompt=None, on_tick=None, pause=None,
            abort_after_faceless: int = 2, need: int = NEED_FRAMES) -> list[Measure]:
    """Text-mode session: drive a lab.LiveEvaluator through the passes with the Collector.
    `on_prompt(key, text)` announces, `pause()` may wait for the user, `on_tick(sample)` reports.
    `seconds` caps each pass. Stops early when the first `abort_after_faceless` passes saw no face."""
    col = Collector(prompts, need=need, timeout=seconds)
    faceless = 0
    while not col.finished:
        key, text, _, _ = col.current
        if on_prompt:
            on_prompt(key, text)
        if pause:
            pause()
        col.start_pass()
        had_face = False
        while True:
            smp = ev.step()
            if smp is None:
                continue
            if on_tick:
                on_tick(smp)
            st = col.feed(smp, getattr(ev, "last_fused", None))
            if st.get("ok") is not None or any(smp.raw.get(sp) is not None for sp in ("ir", "rgb")):
                had_face = True
            if st.get("pass_done") or st.get("finished"):
                break
        faceless = 0 if had_face else faceless + 1
        if abort_after_faceless and faceless >= abort_after_faceless and not col.measures:
            break
    return col.measures
