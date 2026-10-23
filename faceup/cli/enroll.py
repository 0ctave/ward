"""faceup enroll · calibrate — registration, calibration sessions, self-check."""
from __future__ import annotations

import os
import sys

from .. import ui
from ..engine import config as cfgmod
from ..engine import store as storemod
from .common import OPTIONAL_POSES, POSES, need_root, real_user
from .manage import _recogniser_name

# ----------------------------------------------------------------- enroll ----
def _display_available() -> bool:
    """A graphical session the registration window can open on (directly or via the sudo/pkexec user's socket)."""
    from .. import display
    return bool(os.environ.get(display.STREAM_ENV) or os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY") or display.user_wayland_env())


def _stream_ui(args) -> None:
    """--stream-ui: the manager app draws the window; snapshots go to stdout (display.StreamDisplay)."""
    if getattr(args, "stream_ui", False):
        from .. import display
        os.environ[display.STREAM_ENV] = "1"
        display.stream_out()                     # take stdout now: every message goes to the app's console


def cmd_enroll(args) -> int:
    need_root("enroll")
    _stream_ui(args)
    for k in ("fresh", "all", "selfcheck", "guided", "profile"):
        if not hasattr(args, k):
            setattr(args, k, False)
    from ..engine.capture import Capture, CaptureError
    from ..engine.detect import Models
    from ..engine import enroll as en
    cfg = cfgmod.load()
    cams = cfg["cameras"]
    user = real_user(args)
    devices, notes = cfgmod.camera_paths(cfg)
    for n in notes:
        ui.warn(n)
    if not devices:
        ui.fail("no configured camera is connected — run `sudo faceup setup`"); return 1
    ui.banner("pose-guided enrollment", f"for {user} on {' + '.join(devices)} — one model per pose and spectrum")
    models = Models(cfg["recognition"]["detector"], cfg["recognition"]["max_height"], recogniser=cfg["recognition"].get("recogniser", "dlib"))
    try:
        st = storemod.load(user, models.recogniser_name, models.distance_scale)
    except storemod.StoreError as e:
        ui.warn(str(e))
        if not args.fresh and not args.guided and not getattr(args, "classic", False):
            ui.tip("re-run with --fresh to start a store for this recogniser (the window path asks nothing in the terminal)")
            return 1
        if not (args.fresh or ui.confirm("Start a fresh store for this recogniser?", default=True)):
            return 1
        st = storemod.Store(user=user, path=storemod.path_for(user), recogniser=models.recogniser_name, scale=models.distance_scale)
    except FileNotFoundError as e:
        ui.fail(str(e)); return 1
    ui.info(f"recogniser: {models.recogniser_name}")
    identity = (getattr(args, "identity", "") or "").strip()
    condition = (getattr(args, "add", "") or "").strip().lower()
    window = not args.guided and not getattr(args, "classic", False)     # the default: one window, no terminal prompts
    if identity and identity == user:
        identity = ""                                        # the owner is the unnamed identity
    if condition:
        if not [m for m in st.models if m.identity == identity]:
            ui.fail(f"enroll {identity or user} first, then add the condition"); return 1
        ui.info(f"adding condition {condition!r} for {identity or user}: a short sweep, existing models stay")
    elif st.models:
        who = sorted({m.identity or user for m in st.models})
        ui.info(f"already registered: {', '.join(who)}" + (f" · conditions: {', '.join(st.conditions())}" if st.conditions() else ""))
        if identity and identity not in {m.identity for m in st.models}:
            ui.info(f"registering a new identity: {identity}")
        elif args.fresh or (not window and ui.confirm(f"Delete {identity or user}'s models and start clean? (required after changing cameras)", default=False)):
            if identity:
                st.remove_identity(identity)
            else:
                st.models = [m for m in st.models if m.identity]   # keep other people registered on this login
        elif window:
            # the window asks nothing in the terminal: each pose it captures replaces that pose's
            # models; learned models, conditions and other people stay (`--fresh` wipes instead)
            ui.info("registering again: captured poses replace their old models (--fresh starts clean)")
    if window and not sys.stdin.isatty() and not _display_available():
        ui.fail("no display for the registration window and no terminal to fall back to")
        ui.tip("run `sudo faceup enroll --guided` in a terminal, or start it from the faceup app on your desktop")
        return 1
    poses = list(POSES)
    if not window and (args.all or ui.confirm("Also add the optional poses (glasses, headset, lights off)?", default=False)):
        poses += OPTIONAL_POSES
    try:
        cap = Capture(devices)
    except CaptureError as e:
        ui.fail(str(e)); return 1
    done = 0
    with cap:
        cap.warm_up(5)
        if not window and len(cap.spectra) == 2 and not condition and not st.calib.get("s"):   # the window aligns during its intro
            ui.info("Calibrating the two cameras — look straight at the camera…")
            calib = en.calibrate(cap, models, cams)
            if calib:
                st.calib = calib; ui.ok(f"calibration dx={calib['dx']} dy={calib['dy']} s={calib['s']}")
            else:
                ui.warn("calibration failed (no face in both cameras); keeping previous values")
        if not args.guided and not getattr(args, "classic", False):
            # ONE window: enroll + calibrate in a single guided sweep (the fused setup)
            from .. import gui
            if not getattr(args, "stream_ui", False):
                ui.info("A window opens. Follow the arrow to bring your face into each target ring; it registers")
                ui.info("your face and tunes the gate in one pass. Enter skips a pose, Esc cancels.")
            try:
                ui_models = models if cfg["recognition"]["detector"] != "cnn" else Models("yunet", 320, recogniser=models.recogniser_kind)
                from ..engine import calibration as cal
                res = gui.guided_setup(st, cap, ui_models, cams, cfg["recognition"], identity=identity, condition=condition,
                                       poses=cal.CONDITION_POSES if condition else None)
            except gui.GuiUnavailable as e:
                if not sys.stdin.isatty():                   # e.g. the app's console: a text flow would answer itself
                    ui.fail(f"no camera window available ({e})"); return 1
                ui.warn(f"no camera window available ({e}); falling back to the guided text flow")
                args.guided = True; window = False
                if len(cap.spectra) == 2 and not condition and not st.calib.get("s"):
                    calib = en.calibrate(cap, models, cams)        # the window would have aligned in its intro
                    if calib:
                        st.calib = calib
            else:
                cap.close()
                if res is None:
                    ui.warn("cancelled — nothing changed"); return 1
                if res.get("failed"):
                    ui.fail(f"{res['failed']} — nothing changed"); return 1
                if not res.get("added"):
                    # never save a sweep that produced nothing: with --fresh that would erase the old models
                    ui.fail("no pose produced a model (face not found, too dark, or every pass skipped) — nothing changed")
                    return 1
                st.save()
                r = res.get("result")
                if r is not None and r.ok() and not condition and not identity:
                    cal.apply(st, cfg, r, [], save=True)      # gate + config from the owner's held-out distances
                    cfgmod.save(cfg)
                elif r is not None and r.ok():
                    ui.info("gate unchanged: conditions and extra identities never loosen the owner's calibration")
                print()
                st = storemod.load(user, models.recogniser_name, models.distance_scale)
                ui.ok(f"{res.get('added', 0)} models added. Total: {len(st.models)} ({', '.join(f'{s}:{len(st.by_spectrum(s))}' for s in sorted(st.spectra()))})")
                report_pose_calibration(st)
                if r is not None and r.ok():
                    for sp in sorted(r.gate):
                        ui.ok(f"{sp}: gate {r.gate[sp]} (median {r.p50[sp]:.2f}, p95 {r.p95[sp]:.2f})" + (f", whitening {r.w_threshold[sp]}" if sp in r.w_threshold else ""))
                    for wn in r.warnings:
                        ui.warn(wn)
                selfcheck(user, cfgmod.load(), n=2)
                return 0
        ui.info("Sit as you normally do. Each pose captures for about a second — hold still until it says done.")
        existing = {m.label for m in st.models}
        for label, instruction in poses:
            if label in existing:
                ui.info(f"{label}: already registered, skipping"); continue
            print(); ui.info(f"{ui.bold(label)} — {instruction}")
            if ui.ask("Press Enter to capture, or s to skip", "").lower().startswith("s"):
                continue
            for attempt in (1, 2):
                res = en.capture_pose(cap, models, cams)
                wp = en.wrong_person(st, res)
                if res.embeddings and not wp:
                    for s, emb in res.embeddings.items():
                        st.add(label, s, emb, res.quality[s])
                    st.save(); done += 1
                    ui.ok(f"{label} saved ({', '.join(res.embeddings)}; {res.frames} frames)")
                    for p in res.problems: ui.warn(p)
                    break
                ui.warn(wp or "; ".join(res.problems) or "no usable frames")
                if attempt == 1 and not ui.confirm("Retry this pose?", default=True):
                    break
    print()
    ui.ok(f"{done} pose(s) added. Total models: {len(st.models)} ({', '.join(f'{s}:{len(st.by_spectrum(s))}' for s in sorted(st.spectra()))})")
    if args.selfcheck or ui.confirm("Run the calibration session now (8 short passes set the gate from your real distances)?", default=True):
        if run_calibration_session(user, cfg, st) == 0:
            selfcheck(user, cfgmod.load(), n=2)
    return 0


def report_pose_calibration(st) -> None:
    r = st.calib.get("pose_report")
    if not r:
        return
    if "error" in r:
        ui.warn(f"personal 3D face model not built: {r['error']}"); return
    ui.ok(f"personal 3D face model built from {r['frames']} frames (fit residual {r['rms_px']} px)")
    if "pitch_rms" in r:
        ui.info(f"fast estimator vs personal model — yaw: gain {r['yaw_gain']} (1.0 = same scale), residual {r['yaw_rms']}° over a {r['yaw_range']}° range  ·  "
                f"pitch: gain {r['pitch_gain']}, residual {r['pitch_rms']}° over {r['pitch_range']}°")
        ui.tip("the personal model is used for live pose from the next enrollment on")


def cmd_calibrate(args) -> int:
    """session (default): measure the user's real distances and derive the gate + whitening
    threshold · refresh: re-derive from recent accepts · pose: rebuild the personal 3D model."""
    need_root("calibrate")
    _stream_ui(args)
    action = getattr(args, "action", None) or "session"
    if action == "pose":
        return calibrate_pose(args)
    from ..engine import calibration as cal
    cfg = cfgmod.load()
    user = real_user(args)
    try:
        st = storemod.load(user, _recogniser_name(cfg))
    except storemod.StoreError as e:
        ui.fail(str(e)); return 1
    if not st.models:
        ui.fail(f"no models for {user} — enroll first"); return 1
    if action == "refresh":
        recent = st.calib.get("recent") or {}
        raws = [e[1] for e in recent.get("ir", recent.get("rgb", [])) if len(e) > 1 and e[1] is not None]
        if len(raws) < cal.MIN_SAMPLES:
            ui.warn(f"only {len(raws)} recent accepts recorded — use the session: sudo faceup calibrate"); return 1
        res = cal.Result()
        for sp, lst in recent.items():
            rr = [e[1] for e in lst if e[1] is not None]
            if len(rr) >= cal.MIN_SAMPLES:
                res.n[sp] = len(rr); res.p50[sp], res.p95[sp], res.max[sp] = cal.percentile(rr, 50), cal.percentile(rr, 95), max(rr)
                res.gate[sp], note = cal.derive_gate(rr)
                if note: res.warnings.append(f"{sp}: {note}")
                thr = cal.derive_whitening_threshold([e[2] for e in lst if len(e) > 2])
                if thr is not None: res.w_threshold[sp] = thr
        changed = cal.apply(st, cfg, res, [], save=True)
        cfgmod.save(cfg)
        _report_calibration(res, changed, st)
        return 0
    return run_calibration_session(user, cfg, st, seconds=getattr(args, "seconds", None), text=getattr(args, "text", False))


def run_calibration_session(user: str, cfg: dict, st, seconds: float | None = None, confirm_each: bool = True, text: bool = False) -> int:
    """Window with head-movement cues when a display is available (each pass completes once the
    pose estimate has seen the movement), text prompts otherwise or with --text."""
    from ..engine import calibration as cal
    from ..engine import lab as labmod
    from ..engine.capture import CaptureError
    seconds = seconds or cal.PASS_TIMEOUT_S
    ui.banner("calibration session", f"for {user}: {len(cal.PROMPTS)} guided passes measure your real distances; the gate and the whitening threshold come from them")
    if not text:
        # the window: cameras + models here, embeddings in a worker process (gui.calibration_session)
        from .. import gui
        from ..engine.capture import Capture
        from ..engine.detect import Models
        devices, notes = cfgmod.camera_paths(cfg)
        for n in notes:
            ui.warn(n)
        if not devices:
            ui.fail("no configured camera is connected"); return 1
        models = Models("yunet", cfg["recognition"]["max_height"], recogniser=cfg["recognition"].get("recogniser", "dlib"))
        try:
            cap = Capture(devices)
        except CaptureError as e:
            ui.fail(str(e)); return 1
        measures = "text"
        try:
            cap.warm_up(5)
            measures = gui.calibration_session(st, cap, models, cfg["cameras"], cfg["recognition"], timeout=seconds)
        except gui.GuiUnavailable as e:
            ui.warn(f"no window available ({e}); text mode")
        except Exception as e:
            cap.close(); ui.fail(f"session failed: {e!r} — nothing changed"); return 1
        cap.close()
        if measures is None:
            ui.warn("cancelled — nothing changed"); return 1
        if measures != "text":
            fps = getattr(gui.calibration_session, "last_fps", 0.0)
            if fps:
                (ui.ok if fps >= 20 else ui.warn)(f"window ran at {fps:.0f} fps")
            return _finish_calibration(user, cfg, st, measures)
    ev = labmod.LiveEvaluator(user, cfg)
    try:
        ev.open()
    except CaptureError as e:
        ui.fail(str(e)); return 1
    state = {"n": 0}
    def on_prompt(key, text):
        print(); ui.info(f"{ui.bold(key)} — {text}")
    def pause():
        if confirm_each:
            ui.pause("Press Enter, then hold that for a moment")
    def on_tick(smp):
        state["n"] += 1
        if state["n"] % 5 == 0:
            parts = [f"{sp} {smp.raw[sp]:.2f}" + (f" (w {smp.wdist[sp]:.1f})" if smp.wdist.get(sp) is not None else "") for sp in ("ir", "rgb") if smp.raw.get(sp) is not None]
            print(f"\r\033[K    {' · '.join(parts) or 'no face'}", end="", flush=True)
    try:
        measures = cal.collect(ev, seconds=seconds, on_prompt=on_prompt, on_tick=on_tick, pause=pause)
    except KeyboardInterrupt:
        print(); ui.warn("interrupted — nothing changed"); ev.close(); return 1
    except Exception as e:                      # close the cameras before the interpreter tears threads down
        print(); ev.close(); ui.fail(f"session failed: {e!r} — nothing changed"); return 1
    print()
    ev.close()
    return _finish_calibration(user, cfg, st, measures)


def _finish_calibration(user: str, cfg: dict, st, measures) -> int:
    from ..engine import calibration as cal
    res = cal.summarise(measures)
    if not res.ok():
        for w in res.warnings: ui.warn(w)
        ui.fail("not enough measurements to calibrate — nothing changed"); return 1
    changed = cal.apply(st, cfg, res, measures, save=True)
    cfgmod.save(cfg)
    _report_calibration(res, changed, st)
    return 0


def _report_calibration(res, changed: dict, st) -> None:
    from ..engine import calibration as cal
    print()
    for sp in sorted(res.gate):
        cond = ", ".join(f"{k} {v}" for k, v in sorted(res.conditions.get(sp, {}).items()))
        ui.ok(f"{sp}: {res.n[sp]} measurements · raw median {res.p50[sp]:.2f}, p95 {res.p95[sp]:.2f}, max {res.max[sp]:.2f} → gate {res.gate[sp]}"
              + (f" · whitened p95 {res.w_p95[sp]:.1f} → threshold {res.w_threshold[sp]}" if sp in res.w_threshold else " · whitening: not enough samples")
              + (f" · light: {cond}" if cond else ""))
    for k, (old, new) in sorted(changed.items()):
        ui.info(f"{k}: {old} → {new}")
    if not changed:
        ui.info("config already matched the measurements")
    for w in res.warnings:
        ui.warn(w)
    ui.info(f"evidence stored in the model store (calib.calibration); the whitening threshold now follows your accepts every {cal.REFRESH_EVERY} unlocks, within ±{int(cal.REFRESH_STEP * 100)} % and never below this session's value")


def calibrate_pose(args) -> int:
    """Rebuild the personal 3D model from the landmark tracks saved by the last enrollment."""
    from ..engine import sfm
    user = real_user(args)
    try:
        st = storemod.load(user, _recogniser_name(cfgmod.load()))
    except storemod.StoreError as e:
        ui.fail(str(e)); return 1
    f = st.path.with_suffix(".tracks.npz")
    if not f.exists():
        ui.fail(f"no landmark tracks for {user} ({f}) — run `sudo faceup enroll` first"); return 1
    import numpy as np
    d = np.load(str(f))
    tracks, fast, (w, h) = d["tracks"].astype(np.float64), [tuple(x) for x in d["fast"]], d["size"]
    try:
        cal = sfm.calibrate(tracks, int(w), int(h), fast)
    except sfm.CalibrationError as e:
        ui.fail(str(e)); return 1
    st.calib["shape3d"] = [[round(float(v), 2) for v in row] for row in cal.shape3d]
    st.calib["pose_report"] = cal.report
    st.save()
    report_pose_calibration(st)
    return 0


def selfcheck(user: str, cfg: dict, n: int = 3) -> None:
    from ..engine.auth import Engine
    eng = Engine(cfg, verbose=False)
    ok = 0
    for i in range(n):
        out = eng.authenticate(user)
        (ui.ok if out.accepted else ui.warn)(f"attempt {i + 1}: {'accepted' if out.accepted else out.reason} in {out.elapsed_ms} ms"
                                            + (f"  ir={out.ir_dist:.2f}" if out.ir_dist is not None else "")
                                            + (f"  rgb={out.rgb_dist:.2f}" if out.rgb_dist is not None else "")
                                            + (f"  votes={','.join(str(v) for v in out.votes)}" if out.votes else ""))
        ok += out.accepted
    if ok == n:
        ui.ok(ui.bold("Self-check passed."))
    else:
        ui.warn(ui.bold(f"{ok}/{n} accepted — see `faceup log` for reasons; `faceup auth -v` shows every step."))
