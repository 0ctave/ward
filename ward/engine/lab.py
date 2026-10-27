"""Live evaluation ("lab"): run the recognition + liveness pipeline continuously and expose
every intermediate number, with the decision under several unlock policies at once.

Used by `ward lab` (GTK window or text mode) to test reliability with other people, poses,
lighting, and spoof attempts, and to compare policies on the same frames. Nothing here
authenticates anything; it only observes.
"""
from __future__ import annotations

import csv
import os
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from . import brightness, config, pose as posemod, store as storemod
from .auth import Engine, Obs
from .capture import Capture, CaptureError
from .liveness import Result, Vote

def _lab_dir() -> Path:
    """Recordings dir: WARD_LAB, else under WARD_ETC when that override is set (tests,
    unprivileged runs), else the system location."""
    if os.environ.get("WARD_LAB"):
        return Path(os.environ["WARD_LAB"])
    if os.environ.get("WARD_ETC"):
        return Path(os.environ["WARD_ETC"]) / "lab"
    return Path("/var/lib/ward/lab")


LAB_DIR = _lab_dir()

# Unlock policies compared side by side. Each is a pure function of a Sample.
POLICIES = ("strict", "whitened", "whitened+liveness")
POLICY_TEXT = {
    "strict": "raw distance ≤ certainty",
    "whitened": "raw ≤ certainty, or raw ≤ hard cap and whitened ≤ threshold",
    "whitened+liveness": "whitened path and the liveness vote (≥ required PASS, no FAIL)",
}


@dataclass
class Sample:
    t: float
    subject: str
    primary: str
    faces: dict                      # spectrum -> bool
    raw: dict                        # spectrum -> raw distance | None
    label: dict                      # spectrum -> nearest model label
    wdist: dict                      # spectrum -> whitened distance | None
    wthr: dict                       # spectrum -> whitened threshold | None
    votes: list                      # [(voter, result, reason)]
    light: dict                      # spectrum -> face light state
    yaw: float | None
    pitch: float | None
    decisions: dict = field(default_factory=dict)   # policy -> bool
    reason: str = ""
    roll: float | None = None
    face_w: int | None = None                        # primary face box width, px (distance proxy)

    def row(self) -> dict:
        r = {"time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.t)), "subject": self.subject, "primary": self.primary}
        for sp in ("ir", "rgb"):
            r[f"{sp}_face"] = int(bool(self.faces.get(sp)))
            r[f"{sp}_raw"] = "" if self.raw.get(sp) is None else round(self.raw[sp], 3)
            r[f"{sp}_label"] = self.label.get(sp, "")
            r[f"{sp}_w"] = "" if self.wdist.get(sp) is None else round(self.wdist[sp], 3)
            r[f"{sp}_wthr"] = "" if self.wthr.get(sp) is None else round(self.wthr[sp], 3)
            r[f"{sp}_light"] = self.light.get(sp, "")
        r["votes"] = " ".join(f"{v}:{res}" for v, res, _ in self.votes)
        r["yaw"] = "" if self.yaw is None else round(self.yaw, 1)
        r["pitch"] = "" if self.pitch is None else round(self.pitch, 1)
        for p in POLICIES:
            r[p] = int(bool(self.decisions.get(p)))
        r["reason"] = self.reason
        return r


class LiveEvaluator:
    """Rolling window over the last few frame-sets; call step() at camera rate."""

    def __init__(self, user: str, cfg: dict | None = None, window: int = 6):
        self.cfg = cfg or config.load()
        self.user = user
        self.engine = Engine(self.cfg)
        self.models = self.engine.models
        self.store = storemod.load(user, self.models.recogniser_name, self.models.distance_scale)
        self.window = window
        self.obs: dict[str, deque] = {}
        self.cap: Capture | None = None
        self.devices: dict[str, str] = {}
        self.primary = "ir"
        self.subject = "me"
        self.thresholds = {"certainty": float(self.cfg["recognition"]["certainty"]),
                           "rgb_certainty": float(self.cfg["recognition"]["rgb_certainty"]),
                           "hard_cap": float(self.cfg["recognition"].get("hard_cap", 0.55)),
                           "required_votes": int(self.cfg["liveness"]["required_votes"])}
        self.recording = False
        self._csv = None
        self._writer = None
        self.records: list[Sample] = []
        self.recorder = None                       # replay.Recorder when --record-frames is on
        self.fps = 0.0
        self._fps_t, self._fps_n = time.monotonic(), 0

    # -- lifecycle ------------------------------------------------------------
    def open(self) -> None:
        paths, notes = config.camera_paths(self.cfg)
        self.notes = notes
        self.devices = {s: paths[s] for s in ("ir", "rgb") if paths.get(s)}
        if not self.devices:
            raise CaptureError("no configured camera connected")
        self.primary = "ir" if "ir" in self.devices else "rgb"
        self.cap = Capture(self.devices)
        self.cap.warm_up(5)
        self.obs = {s: deque(maxlen=self.window) for s in self.devices}
        self.models.preload()

    def record_frames(self, directory) -> Path:
        from .replay import Recorder
        self.recorder = Recorder(Path(directory))
        return self.recorder.dir

    def close(self) -> None:
        self.stop_recording()
        if self.recorder is not None:
            self.recorder.close(); self.recorder = None
        if self.cap:
            self.cap.close(); self.cap = None

    # -- recording --------------------------------------------------------------
    def start_recording(self, subject: str) -> Path:
        self.stop_recording()
        LAB_DIR.mkdir(parents=True, exist_ok=True)
        path = LAB_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{''.join(c if c.isalnum() else '_' for c in subject)}.csv"
        self._csv = open(path, "w", newline="")
        self._writer = None
        self.subject = subject
        self.recording = True
        self.record_path = path
        return path

    def stop_recording(self) -> None:
        if self._csv:
            self._csv.close()
        self._csv = self._writer = None
        self.recording = False

    def _record(self, s: Sample) -> None:
        self.records.append(s)
        if self._csv:
            row = s.row()
            if self._writer is None:
                self._writer = csv.DictWriter(self._csv, fieldnames=list(row))
                self._writer.writeheader()
            self._writer.writerow(row)
            self._csv.flush()

    # -- evaluation ---------------------------------------------------------------
    def step(self) -> Sample | None:
        if self.cap is None:
            return None
        fs = self.cap.read_set()
        if not fs:
            return None
        if self.recorder is not None:
            self.recorder.write(fs)
        cams, rec, live = self.cfg["cameras"], self.cfg["recognition"], self.cfg["liveness"]
        for s, fr in fs.items():
            prev = next((o.box for o in reversed(self.obs[s]) if o.box is not None), None)
            self.obs[s].append(self.engine._observe(fr, s, cams, rec, prev, self.cap))
        self._fps_n += 1
        if time.monotonic() - self._fps_t >= 1.0:
            self.fps = self._fps_n / (time.monotonic() - self._fps_t); self._fps_n, self._fps_t = 0, time.monotonic()

        obs = {s: list(d) for s, d in self.obs.items()}
        st = self.store
        faces, raw, label, wd, wthr, light = {}, {}, {}, {}, {}, {}
        fused = {}
        for s, lst in obs.items():
            last = lst[-1] if lst else None
            faces[s] = bool(last and last.box is not None)
            light[s] = last.light if last else ""
            f = self.engine._fused(lst, rec["fusion_frames"])
            fused[s] = f
            if f is not None and st.by_spectrum(s):
                d, lab = st.min_distance(f, s)
                raw[s], label[s] = d, lab or ""
                w, thr = st.wdist(f, s)
                wd[s], wthr[s] = w, thr
            else:
                raw[s] = None; label[s] = ""; wd[s] = None; wthr[s] = None
        self.engine._shadow_done = None   # each lab sample is an independent attempt: don't freeze the first shadow vote (authenticate resets this per attempt; the lab reuses one Engine across samples)
        votes_v = self.engine._liveness(st, obs, self.devices, live) if any(faces.values()) else []
        votes = [(v.voter, v.result.value, v.reason) for v in votes_v]
        # pose of the primary face (display convention not needed here: camera angles)
        yaw = pitch = roll = None
        face_w = None
        lst = obs.get(self.primary, [])
        last = lst[-1] if lst else None
        if last is not None and last.box is not None:
            face_w = int(last.box.w)
            try:
                lm = self.models.landmarks68(brightness.normalise(last.frame.gray, last.quality.mean), last.box)
                shape = np.array(st.calib["shape3d"]) if st.calib.get("shape3d") else None
                neutral = posemod.Neutral(st.calib["neutral_ratios"]) if st.calib.get("neutral_ratios") else None
                p = posemod.estimate(lm, last.frame.gray.shape[1], last.frame.gray.shape[0], shape3d=shape, neutral=neutral)
                if p is not None:
                    yaw, pitch, roll = p.yaw, p.pitch, p.roll
                    if shape is not None and st.calib.get("pose_offset"):
                        yaw -= float(st.calib["pose_offset"][0]); pitch -= float(st.calib["pose_offset"][1])
            except Exception:
                pass
        sample = Sample(time.time(), self.subject, self.primary, faces, raw, label, wd, wthr, votes, light, yaw, pitch, roll=roll, face_w=face_w)
        sample.decisions, sample.reason = self.decide(sample)
        if self.recording and any(faces.values()):
            self._record(sample)
        self.last = sample
        self.last_frames = fs
        self.last_obs = obs
        self.last_fused = fused
        return sample

    def decide(self, s: Sample) -> tuple[dict, str]:
        """All policies on one sample, with the current (possibly slider-adjusted) thresholds."""
        th = self.thresholds
        prim = s.primary if s.raw.get(s.primary) is not None else ("rgb" if s.raw.get("rgb") is not None else s.primary)
        gate = th["certainty"] if prim == "ir" else th["rgb_certainty"]
        r = s.raw.get(prim)
        if r is None:
            return {p: False for p in POLICIES}, "no face"
        strict = r <= gate
        whitened = strict or (r <= th["hard_cap"] and s.wdist.get(prim) is not None and s.wthr.get(prim) is not None
                              and s.wdist[prim] <= s.wthr[prim])
        n_pass = sum(1 for _, res, _ in s.votes if res == Result.PASS.value)
        n_fail = sum(1 for _, res, _ in s.votes if res == Result.FAIL.value)
        enabled = len(s.votes)
        required = min(th["required_votes"], enabled)
        live = n_fail == 0 and (enabled == 0 or n_pass >= required or n_pass >= 1)
        reason = ("match" if strict else "whitened match" if whitened else "no match") + ("" if live else " · liveness: " + ("FAIL" if n_fail else "no PASS"))
        return {"strict": strict, "whitened": whitened, "whitened+liveness": whitened and live}, reason

    # -- summary ------------------------------------------------------------------
    def summary(self) -> list[dict]:
        """Per subject: samples, accept rate per policy, raw/whitened distance stats of the primary camera."""
        by: dict[str, list[Sample]] = {}
        for s in self.records:
            by.setdefault(s.subject, []).append(s)
        out = []
        for subject, lst in by.items():
            row = {"subject": subject, "n": len(lst)}
            for p in POLICIES:
                row[p] = sum(1 for s in lst if s.decisions.get(p)) / len(lst)
            prim = lst[0].primary
            raws = [s.raw[prim] for s in lst if s.raw.get(prim) is not None]
            wds = [s.wdist[prim] / s.wthr[prim] for s in lst if s.wdist.get(prim) is not None and s.wthr.get(prim)]
            row["raw_median"] = float(np.median(raws)) if raws else None
            row["raw_min"] = float(np.min(raws)) if raws else None
            row["w_median"] = float(np.median(wds)) if wds else None      # whitened / threshold: < 1 accepts
            out.append(row)
        return out


def format_summary(rows: list[dict]) -> str:
    if not rows:
        return "no recorded samples"
    lines = [f"{'subject':<14}{'n':>5}  {'strict':>7} {'whiten':>7} {'w+live':>7}   {'raw med':>8} {'raw min':>8} {'w/thr':>6}"]
    for r in rows:
        lines.append(f"{r['subject']:<14}{r['n']:>5}  {r['strict']*100:>6.0f}% {r['whitened']*100:>6.0f}% {r['whitened+liveness']*100:>6.0f}%   "
                     f"{(r['raw_median'] if r['raw_median'] is not None else float('nan')):>8.2f} {(r['raw_min'] if r['raw_min'] is not None else float('nan')):>8.2f} "
                     f"{(r['w_median'] if r['w_median'] is not None else float('nan')):>6.2f}")
    return "\n".join(lines)
