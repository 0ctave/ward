"""The authentication pipeline (DESIGN.md §6): capture → quality → detect → embed → fuse →
recognise → liveness votes → decide. Returns an Outcome; never raises for expected failures.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import config, store as storemod
from .capture import Capture, CaptureError, Frame
from .detect import Box, Models, fuse
from .liveness import Result, Vote, antispoof_cnn, cross_spectrum, motion, shadow
from .quality import MIN_FACE_PX, Quality, measure


@dataclass
class Obs:
    frame: Frame
    quality: Quality
    box: Box | None
    embedding: np.ndarray | None
    dark: bool


@dataclass
class Outcome:
    accepted: bool
    reason: str                      # accept | no-face | dark | no-match | spoof:<voter> | camera | no-models | ssh | timeout | store
    user: str = ""
    ir_dist: float | None = None
    rgb_dist: float | None = None
    votes: list[Vote] = field(default_factory=list)
    elapsed_ms: int = 0
    frames: int = 0
    detail: str = ""

    def logline(self) -> str:
        parts = [time.strftime("%Y-%m-%dT%H:%M:%S"), f"user={self.user}", f"result={'accept' if self.accepted else self.reason}"]
        if self.ir_dist is not None:
            parts.append(f"ir={self.ir_dist:.2f}")
        if self.rgb_dist is not None:
            parts.append(f"rgb={self.rgb_dist:.2f}")
        if self.votes:
            parts.append("votes=" + ",".join(str(v) for v in self.votes))
        parts.append(f"frames={self.frames}")
        parts.append(f"t={self.elapsed_ms}ms")
        if self.detail:
            parts.append(f"detail={self.detail!r}")
        return " ".join(parts)


def log(outcome: Outcome) -> None:
    try:
        with open(config.LOG_PATH, "a") as f:
            f.write(outcome.logline() + "\n")
    except OSError:
        pass


class Engine:
    def __init__(self, cfg: dict | None = None, models: Models | None = None, verbose: bool = False):
        self.cfg = cfg or config.load()
        rec = self.cfg["recognition"]
        self.models = models or Models(rec["detector"], rec["max_height"])
        self.verbose = verbose

    def say(self, msg: str) -> None:
        if self.verbose:
            print("  " + msg, flush=True)

    # ------------------------------------------------------------------ run ----
    def authenticate(self, user: str) -> Outcome:
        t0 = time.monotonic()
        cams, rec, live, timing, beh = (self.cfg[k] for k in ("cameras", "recognition", "liveness", "timing", "behaviour"))

        if beh["abort_if_ssh"] and os.environ.get("SSH_CONNECTION"):
            return Outcome(False, "ssh", user)
        try:
            st = storemod.load(user)
        except (storemod.StoreError, OSError, ValueError) as e:
            return Outcome(False, "store", user, detail=str(e))
        if not st.models:
            return Outcome(False, "no-models", user)

        devices = {s: cams[s] for s in ("ir", "rgb") if cams.get(s) and s in st.spectra()}
        if not devices:
            return Outcome(False, "no-models", user, detail="no camera matches the enrolled spectra")
        primary = "ir" if "ir" in devices else "rgb"
        # load the dlib models before the camera opens so their cost overlaps nothing user-visible
        self.models.sp5; self.models.rec
        try:
            cap = Capture(devices)
        except CaptureError as e:
            return Outcome(False, "camera", user, detail=str(e))

        obs: dict[str, list[Obs]] = {s: [] for s in devices}
        last_votes: list[Vote] = []
        saw_face = False
        all_dark = True
        hard = t0 + timing["hard_timeout_ms"] / 1000
        n_sets = 0
        ir_means: list[float] = []
        try:
            settle_ms = self._settle(cap, primary, timing["settle_ms"])
            self.say(f"exposure settled in {settle_ms} ms")
            deadline = min(hard, time.monotonic() + timing["window_ms"] / 1000)   # window starts now
            while time.monotonic() < deadline:
                fs = cap.read_set()
                if not fs:
                    continue
                n_sets += 1
                for s, fr in fs.items():
                    prev = next((o.box for o in reversed(obs[s]) if o.box is not None), None)
                    obs[s].append(self._observe(fr, s, cams, rec, prev))
                for s in fs:
                    if obs[s][-1].box is not None:
                        saw_face = True
                    if not obs[s][-1].dark:
                        all_dark = False
                if "ir" in fs:
                    ir_means.append(obs["ir"][-1].quality.mean)
                # emitter dead at night: every IR frame dark → RGB carries this attempt (DESIGN §10)
                if primary == "ir" and "rgb" in devices and n_sets >= 3 and all(o.dark for o in obs["ir"]):
                    primary = "rgb"
                self.say(f"set {n_sets}: " + ", ".join(
                    f"{s}: {'dark' if obs[s][-1].dark else ('face' if obs[s][-1].box else 'no-face')}" for s in fs))

                decision = self._decide(st, obs, primary, devices, rec, live)
                if decision is None:
                    continue
                accepted, votes, dists = decision
                last_votes = votes
                if accepted:
                    out = Outcome(True, "accept", user, dists.get("ir"), dists.get("rgb"), votes,
                                  int((time.monotonic() - t0) * 1000), n_sets)
                    log(out)
                    return out
                if time.monotonic() > hard:
                    break
        finally:
            cap.close()

        elapsed = int((time.monotonic() - t0) * 1000)
        dists = self._best_dists(st, obs, rec)
        fails = [v for v in last_votes if v.result == Result.FAIL]
        if fails:
            reason, detail = f"spoof:{fails[0].voter}", fails[0].reason
        elif not saw_face:
            reason, detail = ("dark", "") if all_dark and n_sets else ("no-face", "")
        elif dists.get(primary) is None and any(d is not None for d in dists.values()):
            n_ir = sum(o.box is not None for o in obs.get("ir", []))
            rng = f"{min(ir_means):.0f}..{max(ir_means):.0f}" if ir_means else "-"
            reason, detail = f"no-face:{primary}", f"{primary} faces {n_ir}/{len(obs.get(primary, []))}, ir brightness {rng}"
        elif any(d is not None for d in dists.values()):
            reason, detail = "no-match", ""
        else:
            reason, detail = "no-face", ""
        if n_sets == 0:
            reason, detail = "camera", "no frames"
        out = Outcome(False, reason, user, dists.get("ir"), dists.get("rgb"), last_votes, elapsed, n_sets, detail)
        log(out)
        return out

    # ---------------------------------------------------------------- steps ----
    @staticmethod
    def _settle(cap: Capture, primary: str, max_ms: int) -> int:
        """Read frames until the primary camera's auto-exposure stops drifting (or max_ms)."""
        t0 = time.monotonic()
        last: list[float] = []
        while (time.monotonic() - t0) * 1000 < max_ms:
            fs = cap.read_set()
            fr = fs.get(primary)
            if fr is None:
                continue
            last.append(float(fr.gray.mean()))
            if len(last) >= 4 and max(last[-4:]) - min(last[-4:]) < 3.0:
                break
        return int((time.monotonic() - t0) * 1000)

    def _observe(self, fr: Frame, s: str, cams: dict, rec: dict, near: Box | None = None) -> Obs:
        q = measure(fr.gray, rec["max_height"])
        dark = q.mean < cams[f"{s}_dark_threshold"]
        if q.reason(cams[f"{s}_dark_threshold"]):
            return Obs(fr, q, None, None, dark)
        box = self.models.detect(fr.gray, near)
        if box is None or box.w < MIN_FACE_PX:
            return Obs(fr, q, None, None, dark)
        emb = self.models.embed(fr.rgb, box)
        return Obs(fr, q, box, emb, dark)

    def _fused(self, obs: list[Obs], n: int) -> np.ndarray | None:
        good = [o for o in obs if o.embedding is not None]
        if not good:
            return None
        good.sort(key=lambda o: o.quality.score(), reverse=True)
        return fuse([o.embedding for o in good[:n]])

    def _best_dists(self, st, obs, rec) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for s, lst in obs.items():
            f = self._fused(lst, rec["fusion_frames"])
            out[s] = st.min_distance(f, s)[0] if f is not None and st.by_spectrum(s) else None
        return out

    def _decide(self, st, obs, primary, devices, rec, live):
        """None = not enough evidence yet; else (accepted, votes, dists)."""
        dists = self._best_dists(st, obs, rec)
        if dists.get(primary) is None:
            return None
        rec_ir = dists.get("ir") is not None and dists["ir"] <= rec["certainty"]
        rec_rgb = dists.get("rgb") is not None and dists["rgb"] <= rec["rgb_certainty"]
        recognised = rec_ir if primary == "ir" else rec_rgb
        if primary == "ir" and rec["require_rgb_match"]:
            recognised = recognised and rec_rgb
        if not recognised:
            return None   # keep collecting; a later frame may match

        votes = self._liveness(st, obs, devices, live)
        n_pass = sum(v.result == Result.PASS for v in votes)
        n_fail = sum(v.result == Result.FAIL for v in votes)
        enabled = len(votes)
        required = min(live["required_votes"], enabled)
        if n_fail:
            ok = False
        elif enabled == 0:
            ok = True
        elif n_pass >= required:
            ok = True
        elif n_pass >= 1:
            ok = True            # everything else abstained (degraded profile)
        else:
            ok = not live["require_liveness_when_available"]
        self.say("recognised " + " ".join(f"{k}={v:.2f}" for k, v in dists.items() if v is not None)
                 + " votes=" + ",".join(f"{v}({v.reason})" for v in votes) + f" → {'ACCEPT' if ok else 'hold'}")
        return ok, votes, dists

    def _liveness(self, st, obs, devices, live) -> list[Vote]:
        votes: list[Vote] = []
        ir = obs.get("ir", []); rgb = obs.get("rgb", [])
        li = ir[-1] if ir else None
        lr = rgb[-1] if rgb else None
        if live["cross_spectrum"] and "ir" in devices and "rgb" in devices and li and lr:
            votes.append(cross_spectrum.vote(
                li.box, lr.box, li.dark, lr.dark, st.calib,
                (li.frame.gray.shape[1], li.frame.gray.shape[0]), (lr.frame.gray.shape[1], lr.frame.gray.shape[0])))
        if live["antispoof_cnn"] and "rgb" in devices and antispoof_cnn.available():
            votes.append(antispoof_cnn.vote(lr.frame.rgb if lr else None, lr.box if lr else None, lr.dark if lr else True))
        if live["shadow_test"] is True:   # "auto" stays off until toggle-bench proves the hardware (§15.1)
            votes.append(shadow.vote(None, None, None))
        if live["motion"]:
            seq = [o for o in (ir if ir else rgb) if o.box is not None]
            if len(seq) >= 2:
                votes.append(motion.vote(seq[0].frame.gray, seq[-1].frame.gray, seq[-1].box))
            else:
                from .liveness import ABSTAIN
                votes.append(ABSTAIN(motion.NAME, "single frame"))
        return votes
