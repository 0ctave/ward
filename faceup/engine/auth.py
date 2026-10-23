"""The authentication pipeline (DESIGN.md §6): capture → quality → detect → embed → fuse →
recognise → liveness votes → decide. Returns an Outcome; never raises for expected failures.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import brightness, calibration, config, emitter as emitmod, policy as polmod, replay as replaymod, store as storemod
from .capture import Capture, CaptureError, Frame
from .detect import Box, Models, fuse
from .liveness import Result, Vote, antispoof_cnn, cross_spectrum, motion, shadow
from .quality import MIN_FACE_PX, Quality, measure


def cv2_gray_to_rgb(gray):
    import cv2
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


@dataclass
class Obs:
    frame: Frame
    quality: Quality
    box: Box | None
    embedding: np.ndarray | None
    dark: bool
    light: str = ""          # face light state: ok | dim | bright | dark | blown ("" = no face)
    dist: float | None = None   # this frame's own raw distance to the nearest model (None = no embedding)


@dataclass
class Outcome:
    accepted: bool
    reason: str                      # accept | no-face | dark | no-match | liveness | spoof:<voter> | camera | no-models | ssh | timeout | store | login
    user: str = ""
    ir_dist: float | None = None
    rgb_dist: float | None = None
    wdist: dict = field(default_factory=dict)      # spectrum -> (whitened distance, threshold)
    learned: str = ""                             # label of a model added by online adaptation
    identity: str = ""                            # named identity that matched ("" = the account owner)
    condition: str = ""                           # condition of the matched model (glasses, headset…)
    policy: str = ""                              # policy level this attempt ran under
    votes: list[Vote] = field(default_factory=list)
    elapsed_ms: int = 0
    frames: int = 0
    detail: str = ""
    phases: dict = field(default_factory=dict)    # ms from the start: models, cams, face (roadmap F3)

    def logline(self) -> str:
        parts = [time.strftime("%Y-%m-%dT%H:%M:%S"), f"user={self.user}", f"result={'accept' if self.accepted else self.reason}"]
        if self.ir_dist is not None:
            parts.append(f"ir={self.ir_dist:.2f}")
        if self.rgb_dist is not None:
            parts.append(f"rgb={self.rgb_dist:.2f}")
        for sp, (d, thr) in self.wdist.items():
            if d is not None:
                parts.append(f"{sp}w={d:.2f}/{thr:.2f}")
        if self.votes:
            parts.append("votes=" + ",".join(str(v) for v in self.votes))
        if self.learned:
            parts.append(f"learned={self.learned}")
        if self.identity:
            parts.append(f"who={self.identity}")
        if self.condition:
            parts.append(f"cond={self.condition}")
        if self.policy and self.policy != "balanced":
            parts.append(f"policy={self.policy}")
        parts.append(f"frames={self.frames}")
        parts.append(f"t={self.elapsed_ms}ms")
        if self.phases:
            parts.append("ph=" + ",".join(f"{k}:{v}" for k, v in self.phases.items()))
        if self.detail:
            parts.append(f"detail={self.detail!r}")
        return " ".join(parts)


def log(outcome: Outcome) -> None:
    svc = os.environ.get("PAM_SERVICE", "")
    if svc and not outcome.detail.startswith("via "):        # which caller: sudo, gdm-password, polkit-1…
        outcome.detail = f"via {svc}" + (f"; {outcome.detail}" if outcome.detail else "")
    try:
        with open(config.LOG_PATH, "a") as f:
            f.write(outcome.logline() + "\n")
    except OSError:
        pass


class Engine:
    def __init__(self, cfg: dict | None = None, models: Models | None = None, verbose: bool = False):
        self.cfg = cfg or config.load()
        rec = self.cfg["recognition"]
        from .. import hw
        hw.apply()
        self.models = models or Models(rec["detector"], rec["max_height"], rec.get("accel", "cpu"), rec.get("recogniser", "dlib"))
        self.verbose = verbose
        self.events = None          # optional callable(kind, **kw): 'face' once, 'hold' (recognised, liveness pending)
        # per-attempt state, also read by the lab which calls _liveness outside authenticate()
        self._cap = None
        self._emitter = None
        self._shadow_done = None

    def say(self, msg: str) -> None:
        if self.verbose:
            print("  " + msg, flush=True)

    def emit(self, kind: str, **kw) -> None:
        if self.events is not None:
            try:
                self.events(kind, **kw)
            except Exception:
                pass                 # feedback must never break authentication

    # ------------------------------------------------------------------ run ----
    def authenticate(self, user: str) -> Outcome:
        t0 = time.monotonic()
        cams, rec, live, timing, beh = (self.cfg[k] for k in ("cameras", "recognition", "liveness", "timing", "behaviour"))

        def fail(reason: str, detail: str = "") -> Outcome:
            out = Outcome(False, reason, user, detail=detail, elapsed_ms=int((time.monotonic() - t0) * 1000))
            log(out)
            return out

        if beh["abort_if_ssh"] and os.environ.get("SSH_CONNECTION"):
            return fail("ssh")
        service = os.environ.get("PAM_SERVICE", "")
        pol = polmod.effective(self.cfg, service)
        self._policy = pol
        if pol.get("off"):
            return fail("policy", f"policy off for {service or 'this service'}")
        if service == "login" and str(os.environ.get("PAM_TTY", "")).startswith("/dev/tty") and not beh.get("tty_login", False):
            return fail("policy", "text console (behaviour.tty_login = false)")
        locked, remaining = polmod.locked_out(self.cfg, config.LOG_PATH, user)
        if locked:
            return fail("lockout", f"{remaining}s left after repeated liveness failures")
        # this attempt's knobs: the preset tightens the measured gates, never loosens them
        rec = dict(rec); live = dict(live); beh = dict(beh)
        rec["certainty"], rec["rgb_certainty"] = pol["certainty"], pol["rgb_certainty"]
        rec["whitened"] = pol["whitened"]; rec["require_rgb_match"] = pol["require_rgb_match"]
        live["required_votes"] = pol["required_votes"]; beh["learn"] = pol["learn"]
        try:
            st = storemod.load(user, self.models.recogniser_name, self.models.distance_scale)
        except (storemod.StoreError, OSError, ValueError, FileNotFoundError) as e:
            return fail("store", str(e))
        if not st.models:
            return fail("no-models")

        replay_dir = replaymod.replay_dir()
        if replay_dir:                                   # roadmap L7: recorded frame sets instead of cameras
            try:
                rc = replaymod.ReplayCapture(replay_dir)
            except CaptureError as e:
                return fail("camera", str(e))
            devices = {s: f"replay:{s}" for s in rc.spectra if s in st.spectra()}
            notes = []
        else:
            paths, notes = config.camera_paths(self.cfg)
            for n in notes:
                self.say(n)
            devices = {s: paths[s] for s in ("ir", "rgb") if paths.get(s) and s in st.spectra()}
        if not devices:
            return fail("camera", "; ".join(notes) or "no camera matches the enrolled spectra")
        primary = "ir" if "ir" in devices else "rgb"
        self._shadow_done = None
        self._emitter = None
        # Turn the IR LEDs on FIRST, before the cameras open: they take ~400 ms to reach full
        # brightness, and asserting them here lets that ramp overlap the camera open + model load
        # instead of the first ~800 ms of recognition seeing a dark IR frame (the "LEDs light up
        # slowly, then it starts recognising" delay). Done whenever an emitter is known, not only
        # for the shadow test; a probe/previous run may have left the LEDs off.
        if "ir" in devices and not replay_dir:
            try:
                em = emitmod.Emitter(devices["ir"])
                if em.available:
                    em.on(); self._emitter = em                # fails soft: .available False → shadow ABSTAINs
            except Exception:
                self._emitter = None
        # open the cameras in the background while the dlib models load: the two costs overlap
        # (roadmap R1; each V4L2 open blocks 150–500 ms in the driver and releases the GIL)
        pending = replaymod.PendingReplay(replay_dir) if replay_dir else Capture.open_async(devices)
        ph = self._phases = {}
        # Let the IR camera reach streaming BEFORE loading the dlib models: their load holds the GIL
        # for 1–2.5 s and stalled the open thread behind it, so the LEDs came on ~3 s into the
        # attempt. IR open alone is ~0.4 s; the LED ramp and auto-exposure then settle during the load.
        try:
            if hasattr(pending, "wait_primary"):
                pending.wait_primary(min(1.5, timing["hard_timeout_ms"] / 1000))
                ph["ir"] = int((time.monotonic() - t0) * 1000)
            self.models.preload()
            ph["models"] = int((time.monotonic() - t0) * 1000)
            # start on the primary (IR) camera; RGB joins on a later read once open (light read ~2 s)
            get = getattr(pending, "result_primary", pending.result)
            cap = get(timeout=timing["hard_timeout_ms"] / 1000)
        except BaseException as e:
            # nobody will read what is still opening: release it (non-blocking) and the emitter
            try:
                pending.cancel()
            except Exception:
                pass
            if self._emitter is not None:
                self._emitter.close(); self._emitter = None
            if isinstance(e, CaptureError):
                return fail("camera", str(e))
            raise
        self._cap = cap
        ph["cams"] = int((time.monotonic() - t0) * 1000)
        self._notes_seen = 0
        wanted = dict(devices)
        devices = self._follow_cameras(cap, devices)

        camera_lost = ""
        obs: dict[str, list[Obs]] = {s: [] for s in devices}
        last_votes: list[Vote] = []
        saw_face = False
        all_dark = True
        hard = t0 + timing["hard_timeout_ms"] / 1000
        n_sets = 0
        ir_means: list[float] = []
        try:
            if timing["settle_ms"] > 0:
                self.say(f"exposure settled in {self._settle(cap, primary, timing['settle_ms'])} ms")
            deadline = min(hard, time.monotonic() + timing["window_ms"] / 1000)   # window starts now
            while time.monotonic() < deadline:
                try:
                    fs = cap.read_set()
                except CaptureError as e:
                    self.say(str(e)); camera_lost = str(e); break      # fail open: the password follows
                if set(getattr(cap, "spectra", devices)) != set(devices):
                    devices = self._follow_cameras(cap, wanted)      # RGB joined late, or dropped (dark room)
                    primary = "ir" if "ir" in devices else primary
                    for s in devices:
                        obs.setdefault(s, [])
                if not fs:
                    continue
                n_sets += 1
                for s, fr in fs.items():
                    prev = next((o.box for o in reversed(obs[s]) if o.box is not None), None)
                    o = self._observe(fr, s, cams, rec, prev, cap)
                    if o.embedding is not None and st.by_spectrum(s):
                        o.dist = st.min_distance(o.embedding, s)[0]     # per-frame raw distance (early reject, log)
                    obs[s].append(o)
                for s in fs:
                    if obs[s][-1].box is not None:
                        if not saw_face:
                            ph["face"] = int((time.monotonic() - t0) * 1000)
                            self.emit("face")
                        saw_face = True
                    if not obs[s][-1].dark:
                        all_dark = False
                if "ir" in fs:
                    ir_means.append(obs["ir"][-1].quality.mean)
                # adaptive primary (DESIGN §10): IR is preferred, but if its face is dark/unusable
                # (emitter off, too far for the LEDs) while RGB sees the face well, RGB carries this attempt
                if primary == "ir" and "rgb" in devices and n_sets >= 3:
                    ir_bad = all(o.dark or o.embedding is None for o in obs["ir"][-3:])
                    rgb_ok = any(o.embedding is not None and o.light in ("ok", "dim", "bright") for o in obs["rgb"][-3:])
                    if ir_bad and rgb_ok:
                        primary = "rgb"
                        self.say("IR face unusable, RGB carries this attempt")
                self.say(f"set {n_sets}: " + ", ".join(
                    f"{s}: {'dark' if obs[s][-1].dark else ('face:' + obs[s][-1].light if obs[s][-1].box else 'no-face')}" for s in fs))

                decision = self._decide(st, obs, primary, devices, rec, live)
                if decision is None:
                    if self._confident_reject(obs[primary], rec):
                        self.say("confident reject: ending the window early")
                        break                                     # roadmap R7: the password prompt comes sooner
                    continue
                accepted, votes, dists = decision
                last_votes = votes
                spoof = next((v for v in votes if v.result == Result.FAIL and v.voter in polmod.SPOOF_VOTERS), None)
                if spoof is not None:
                    # sticky: a spoof signal ends the attempt — probing angles until one frame slips
                    # through must not accept, and the FAIL is logged (spoof:*) for the lockout
                    self.say(f"{spoof.voter} FAIL ({spoof.reason}): ending the attempt")
                    break
                if accepted:
                    who = getattr(self, "_last_who", None)
                    out = Outcome(True, "accept", user, dists.get("ir"), dists.get("rgb"), getattr(self, "_last_w", {}), "",
                                  who.identity if who else "", who.condition if who else "", pol["level"], votes,
                                  int((time.monotonic() - t0) * 1000), n_sets)
                    out.detail = self._light_note(cap); out.phases = dict(ph)
                    out.learned = self._learn(st, obs, dists, votes, beh, rec)
                    self._remember(st, obs, dists, votes)
                    log(out)
                    return out
                # recognised but liveness is holding: tell the user what would help
                still = any(v.result == Result.ABSTAIN and v.voter == motion.NAME and "still" in v.reason for v in votes)
                self.emit("hold", still=still, votes=votes)
                if time.monotonic() > hard:
                    break
        finally:
            if self._emitter is not None:
                self._emitter.on(); self._emitter.close(); self._emitter = None
            cap.close()
            self._cap = None

        elapsed = int((time.monotonic() - t0) * 1000)
        dists = self._best_dists(st, obs, rec)
        fails = [v for v in last_votes if v.result == Result.FAIL]
        # what each camera saw, for the log: "ir:face:dim rgb:dark" — the difference between
        # "not recognised" and "recognised but the other camera was covered" must be visible
        seen = " ".join(f"{s}:" + ("dark" if all(o.dark for o in lst) else f"face:{next((o.light for o in reversed(lst) if o.box), '')}" if any(o.box for o in lst) else "no-face")
                        for s, lst in obs.items() if lst)
        if fails:
            reason, detail = f"spoof:{fails[0].voter}", fails[0].reason
        elif last_votes and dists.get(primary) is not None:
            # recognised at least once but liveness never reached the required PASSes
            reason = "liveness"
            detail = ", ".join(f"{v.voter} {v.result.value.lower()}: {v.reason}" for v in last_votes) + f" | {seen}"
        elif not saw_face:
            reason, detail = ("dark", "") if all_dark and n_sets else ("no-face", "")
        elif dists.get(primary) is None and any(d is not None for d in dists.values()):
            n_ir = sum(o.box is not None for o in obs.get("ir", []))
            rng = f"{min(ir_means):.0f}..{max(ir_means):.0f}" if ir_means else "-"
            reason, detail = f"no-face:{primary}", f"{primary} faces {n_ir}/{len(obs.get(primary, []))}, ir brightness {rng}"
        elif any(d is not None for d in dists.values()):
            reason, detail = "no-match", seen
        else:
            reason, detail = "no-face", ""
        if n_sets == 0:
            reason, detail = "camera", "no frames"
        if camera_lost:
            reason, detail = "camera", camera_lost
        note = self._light_note(cap)
        detail = f"{note}; {detail}" if note and detail else (note or detail)
        out = Outcome(False, reason, user, dists.get("ir"), dists.get("rgb"), getattr(self, "_last_w", {}), "", "", "",
                      getattr(self, "_policy", {}).get("level", ""), last_votes, elapsed, n_sets, detail, dict(ph))
        log(out)
        return out

    # ------------------------------------------------------- adaptation ----
    def _remember(self, st, obs, dists, votes) -> None:
        """Rolling calibration (calibration.py): a confident accept's distances go into the store's
        ring; every REFRESH_EVERY accepts the whitening threshold follows the recent window."""
        try:
            if any(v.result == Result.FAIL for v in votes):
                return
            light = {s: next((o.light for o in reversed(lst) if o.box is not None), "") for s, lst in obs.items()}
            calibration.record_accept(st, dists, getattr(self, "_last_w", {}), light)
            st.save()
        except Exception:
            pass                                       # never on the critical path

    # ------------------------------------------------------------- learning ----
    def _learn(self, st, obs, dists, votes, beh, rec) -> str:
        """Online adaptation (§6.5b): after a CONFIDENT accept — raw distance well inside the gate
        and liveness passed with no FAIL — store the live fused embedding as a learned model if it
        is novel (far enough from every existing model of that bin), so real night-time / new-glasses
        conditions enter the store from use. Strict on purpose: a stranger never gets near
        learn_max_dist, and originals are never replaced."""
        if not beh.get("learn", True):
            return ""
        if any(v.result == Result.FAIL for v in votes):
            return ""
        if not any(v.result == Result.PASS and v.voter in polmod.SPOOF_VOTERS for v in votes):
            return ""                                         # motion alone is weak liveness: never learn from it
        # never past the gate, even when hand-edited: each learned model widens the accept region around it
        max_dist = min(beh.get("learn_max_dist", 0.30), rec.get("certainty", 0.40) - 0.10)
        learned = []
        for sp, fused in getattr(self, "_last_fused", {}).items():
            originals = [m for m in st.by_spectrum(sp) if not m.learned]
            if fused is None or not originals:
                continue
            # measured against ENROLLED models only: learned models must never anchor further learning,
            # or each one sits at the edge of the accept region and the next chains off it (review D1)
            d = min(st.distance(m.embedding, fused) for m in originals)
            if d > max_dist:
                continue
            if len(getattr(self, "_last_samples", {}).get(sp, [])) < 2:
                continue                                      # one frame is not evidence of a new condition
            # pose bin of the best-quality observation with a face
            best = max((o for o in obs.get(sp, []) if o.box is not None), key=lambda o: o.quality.score(), default=None)
            label = None
            if best is not None:
                try:
                    from . import pose as posemod
                    lm = self.models.landmarks68(best.frame.gray, best.box)
                    shape = np.array(st.calib["shape3d"]) if st.calib.get("shape3d") else None
                    p = posemod.estimate(lm, best.frame.gray.shape[1], best.frame.gray.shape[0], shape3d=shape)
                    if p is not None:
                        label = posemod.cell_name(p.cell() or posemod.nearest_cell(p.yaw, p.pitch))
                except Exception:
                    label = None
            label = label or "unknown"
            # novelty against EVERY model of the spectrum: a condition already covered by any bin
            # (or a bin the pose estimator failed to name) is not new
            if min(st.distance(m.embedding, fused) for m in st.by_spectrum(sp)) < beh.get("learn_min_novelty", 0.25):
                continue                                      # nothing new about this condition
            mine = [m for m in st.by_spectrum(sp) if m.label == label and m.learned]
            if len(mine) >= beh.get("learn_per_bin", 2):
                st.models.remove(min(mine, key=lambda m: m.time))   # rotate the oldest learned one
            allmine = [m for m in st.by_spectrum(sp) if m.learned]
            if len(allmine) >= beh.get("learn_max_total", 12):
                st.models.remove(min(allmine, key=lambda m: m.time))   # and a total cap per camera
            # the per-frame embeddings (not just their mean) so the whitening scatter learns this condition
            frames = [np.asarray(v) for v in getattr(self, "_last_samples", {}).get(sp, [])] or [fused]
            who = st.nearest(fused, sp)[1]
            st.add(label, sp, fused, 0.5, None, samples=frames, learned=True,
                   identity=who.identity if who else "", condition=who.condition if who else "")
            learned.append(f"{sp}:{label}")
        if learned:
            try:
                for sp in st.spectra():
                    st.fit_whitening(sp, clamp=0.2)          # bounded drift; `calibrate refresh` re-derives it
                st.save()
            except OSError:
                pass
        return ",".join(learned)

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

    def _observe(self, fr: Frame, s: str, cams: dict, rec: dict, near: Box | None = None, cap: Capture | None = None) -> Obs:
        q = measure(fr.gray, rec["max_height"])
        frame_dark = q.mean < cams[f"{s}_dark_threshold"]
        if q.sharpness < 5.0 and frame_dark:        # nothing at all in the frame: skip the detector
            return Obs(fr, q, None, None, True)
        gray = brightness.normalise(fr.gray, q.mean)      # identity when the frame is in range
        box = self.models.detect(gray, near)
        if box is None or box.w < MIN_FACE_PX:
            return Obs(fr, q, None, None, frame_dark)
        light = (cap.meter(s, fr.gray, box) if cap is not None else brightness.meter(fr.gray, box))
        state = light.state(s)
        if state == "dark":
            return Obs(fr, q, box, None, True, state)     # a face, but unusably dark (emitter off?)
        rgb = fr.rgb if s == "ir" and gray is fr.gray else (cv2_gray_to_rgb(gray) if s == "ir" else brightness.normalise_rgb(fr.rgb, fr.gray))
        emb = self.models.embed(rgb, box)
        return Obs(fr, q, box, emb, False, state)

    def _fused(self, obs: list[Obs], n: int, samples: list | None = None) -> np.ndarray | None:
        """Fusion of the n best-quality embeddings; `samples` (if given) receives them (roadmap L3)."""
        good = [o for o in obs if o.embedding is not None]
        if not good:
            return None
        good.sort(key=lambda o: o.quality.score(), reverse=True)
        chosen = [o.embedding for o in good[:n]]
        if samples is not None:
            samples[:] = chosen
        return fuse(chosen)

    @staticmethod
    def _confident_reject(obs: list[Obs], rec: dict) -> bool:
        """True when enough frames of the primary camera agree that this is not the user (§6.5):
        `early_reject` (0 = off) exceeded by the median raw distance of the last max(6, fusion_frames + 2)."""
        thr = float(rec.get("early_reject", 0.0) or 0.0)
        if thr <= 0:
            return False
        # the MEDIAN of the last n, not all of them: one lucky frame under the threshold kept the
        # old rule from ever firing, so every stranger/no-match waited out the whole window (3.5 s)
        good = [o.dist for o in obs if o.embedding is not None and getattr(o, "dist", None) is not None]
        n = max(6, int(rec.get("fusion_frames", 4)) + 2)
        return len(good) >= n and float(np.median(good[-n:])) > thr

    def _best_dists(self, st, obs, rec) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for s, lst in obs.items():
            f = self._fused(lst, rec["fusion_frames"])
            out[s] = st.min_distance(f, s)[0] if f is not None and st.by_spectrum(s) else None
        return out

    def _recognised(self, st, fused: np.ndarray | None, spectrum: str, raw: float | None, gate: float, rec: dict) -> tuple[bool, tuple]:
        """Raw gate, or — if the store has whitening for this spectrum — the whitened gate under a hard
        raw cap. The whitened path extends acceptance only along directions this user's own face is
        known to vary in (pose, expression, lighting), never along identity directions (§6.5a)."""
        if raw is None or fused is None:
            return False, (None, None)
        if raw <= gate:
            return True, st.wdist(fused, spectrum) if rec.get("whitened", True) else (None, None)
        if rec.get("whitened", True) and raw <= rec.get("hard_cap", 0.55):
            wd, thr = st.wdist(fused, spectrum)
            if wd is not None and wd <= thr:
                return True, (wd, thr)
            return False, (wd, thr)
        return False, (None, None)

    @staticmethod
    def _light_note(cap) -> str:
        """'ir-only (dark)' when the attempt ran on IR alone because of the room light — the
        attempts view and `log --summary` can then tell night unlocks from day ones."""
        return "ir-only (dark)" if any("IR only" in n for n in getattr(cap, "notes", [])) else ""

    def _follow_cameras(self, cap, devices: dict) -> dict:
        """The cameras actually streaming: Capture opens IR alone in a dark room and drops RGB
        when it starves the IR stream (ambient.py). Liveness availability follows from this."""
        for n in getattr(cap, "notes", [])[getattr(self, "_notes_seen", 0):]:
            self.say(n)
        self._notes_seen = len(getattr(cap, "notes", []))
        spectra = getattr(cap, "spectra", None)
        return devices if spectra is None else {s: p for s, p in devices.items() if s in spectra}

    def _decide(self, st, obs, primary, devices, rec, live):
        """None = not enough evidence yet; else (accepted, votes, dists)."""
        dists = self._best_dists(st, obs, rec)
        if dists.get(primary) is None:
            return None
        samples: dict[str, list] = {s: [] for s in obs}
        fused = {s: self._fused(obs[s], rec["fusion_frames"], samples[s]) for s in obs}
        self._last_samples = samples
        self._last_who = st.nearest(fused[primary], primary)[1] if fused.get(primary) is not None else None
        rec_ir, w_ir = self._recognised(st, fused.get("ir"), "ir", dists.get("ir"), rec["certainty"], rec)
        rec_rgb, w_rgb = self._recognised(st, fused.get("rgb"), "rgb", dists.get("rgb"), rec["rgb_certainty"], rec)
        self._last_w = {sp: w for sp, w in (("ir", w_ir), ("rgb", w_rgb)) if w[0] is not None}
        self._last_fused = fused
        recognised = rec_ir if primary == "ir" else rec_rgb
        pol = getattr(self, "_policy", {}) or {}
        rgb_sees_well = any(o.light in ("ok", "bright") and o.embedding is not None for o in obs.get("rgb", [])[-3:])
        if primary == "ir" and (rec["require_rgb_match"] or (pol.get("rgb_when_bright") and rgb_sees_well and "rgb" in devices)):
            recognised = recognised and rec_rgb            # balanced/strict: the colour camera must agree when it can see you
        if not recognised:
            return None   # keep collecting; a later frame may match

        votes = self._liveness(st, obs, devices, live)
        n_pass = sum(v.result == Result.PASS for v in votes)
        n_fail = sum(v.result == Result.FAIL for v in votes)
        enabled = len(votes)
        required = min(live["required_votes"], enabled)
        strong = any(v.result == Result.PASS and v.voter in polmod.STRONG_VOTERS for v in votes)
        if n_fail:
            ok = False
        elif pol.get("need_strong_voter") and not strong:
            ok = False           # strict: a shadow-test or CNN PASS is mandatory
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

    @staticmethod
    def _shadow_enabled(live: dict) -> bool:
        """True when config says so, or "auto" and `emitter toggle-bench` proved the LEDs switch fast enough."""
        v = live.get("shadow_test", "auto")
        if v is True:
            return True
        if v == "auto":
            ms = int(live.get("shadow_toggle_ms", -1))
            return 0 <= ms <= emitmod.SHADOW_MAX_LATENCY_MS
        return False

    def _shadow_vote(self, li) -> Vote:
        """Emitter-modulation shadow test, once per attempt (it costs ~4 IR frames)."""
        if self._shadow_done is not None:
            return self._shadow_done
        from .liveness import ABSTAIN
        em, cap = getattr(self, "_emitter", None), getattr(self, "_cap", None)
        if em is None or not em.available or cap is None or li is None or li.box is None:
            self._shadow_done = ABSTAIN(shadow.NAME, (em.error if em is not None and em.error else "toggle unavailable"))
            return self._shadow_done
        def read_gray():
            fr = cap.read_one("ir", flush=1)
            return None if fr is None else fr.gray
        # settle frames follow the measured switch latency (reference module: ~3 frames to full brightness)
        ms = int(self.cfg["liveness"].get("shadow_toggle_ms", 100) or 100)
        settle = max(2, -(-ms // 33) + 1)
        try:
            off, on = emitmod.shadow_pair(em, read_gray, settle_frames=settle)
        except Exception as e:
            off = on = None
            em.error = str(e)
        self._shadow_done = shadow.vote(off, on, li.box)
        self.say(f"shadow test: {self._shadow_done}({self._shadow_done.reason})")
        return self._shadow_done

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
        if self._shadow_enabled(live) and "ir" in devices:
            votes.append(self._shadow_vote(li))
        if live["motion"]:
            seq = [o for o in (ir if ir else rgb) if o.box is not None]
            if len(seq) >= 2:
                votes.append(motion.vote(seq[0].frame.gray, seq[-1].frame.gray, seq[-1].box))
            else:
                from .liveness import ABSTAIN
                votes.append(ABSTAIN(motion.NAME, "single frame"))
        return votes
