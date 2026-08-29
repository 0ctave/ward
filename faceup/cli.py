"""faceup command line — see docs/DESIGN.md §11."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.request

from . import __version__, cameras, howdy, probe, ui
from .engine import config as cfgmod
from .engine import pam as pammod
from .engine import store as storemod

EMITTER_REPO = "EmixamPP/linux-enable-ir-emitter"

POSES = [
    ("straight",   "Look straight at the camera, neutral face."),
    ("keyboard",   "Look DOWN at the keyboard — where your face is when a sudo prompt appears."),
    ("tilt-left",  "Tilt your head ~15° to the left (ear toward shoulder)."),
    ("tilt-right", "Tilt your head ~15° to the right."),
    ("turn-left",  "Turn your head ~20° to the left, eyes on the screen."),
    ("turn-right", "Turn your head ~20° to the right."),
    ("lean-back",  "Sit back further from the screen, relaxed posture."),
    ("close",      "Lean in close, like reading small text."),
]
OPTIONAL_POSES = [
    ("glasses", "Wearing your glasses (or without them, if you enrolled with them on)."),
    ("headset", "Wearing headphones / a headset."),
    ("dark",    "Lights OFF — verifies the IR emitter is doing the work."),
]


def need_root(what: str) -> None:
    if not probe.is_root() and "FACEUP_ETC" not in os.environ:
        ui.fail(f"{what} needs root (config and models live under /etc/faceup).")
        ui.tip(f"run:  sudo faceup {' '.join(sys.argv[1:])}")
        sys.exit(2)


def real_user(args) -> str:
    return getattr(args, "user", None) or os.environ.get("PAM_USER") or os.environ.get("SUDO_USER") or os.environ.get("USER") or "root"


# ----------------------------------------------------------------- doctor ----
def cmd_doctor(args) -> int:
    ui.banner(f"faceup {__version__} · doctor", "Everything that decides whether face unlock works on this machine")
    problems = 0
    cfg = cfgmod.load()
    user = real_user(args)

    ui.section("Cameras")
    cams = cameras.list_cameras()
    if not cams:
        ui.fail("no capture-capable /dev/video* device"); problems += 1
    for c in cams:
        (ui.ok if c.is_ir else ui.info)(c.describe())
    ir = [c for c in cams if c.is_ir]
    if cfgmod.CONFIG_PATH.exists():
        ui.ok(f"config: ir = {cfg['cameras']['ir'] or '(none)'}, rgb = {cfg['cameras']['rgb'] or '(none)'}   ({cfgmod.CONFIG_PATH})")
        if ir and not cfg["cameras"]["ir"]:
            ui.warn(f"an IR camera exists ({ir[0].path}) but is not configured"); problems += 1
    else:
        ui.warn(f"no config yet ({cfgmod.CONFIG_PATH}) — run `sudo faceup setup`"); problems += 1
    for s in ("ir", "rgb"):
        dev = cfg["cameras"][s]
        if dev and os.path.exists(dev):
            L = probe.lighting(dev)
            line = f"{s} lighting now: mean {L.mean:.0f}/255 → {L.verdict}"
            if L.frames == 0:
                ui.fail(f"{s}: camera gave no frames (busy or permission denied)"); problems += 1
            elif L.verdict in ("dark", "dim") and s == "ir":
                ui.warn(line); ui.tip("dark IR = the emitter is probably not firing"); problems += 1
            else:
                ui.ok(line)

    if ir:
        ui.section("IR emitter (linux-enable-ir-emitter)")
        e = probe.emitter_status()
        if not e["installed"]:
            ui.warn("not installed — if the IR preview is black in a dark room, the emitter isn't firing")
            ui.tip("sudo faceup emitter install")
        else:
            (ui.ok if e["enabled"] else ui.warn)("service " + ("enabled (re-applies at boot and resume)" if e["enabled"] else "NOT enabled"))
            if e["configured"] is False:
                ui.fail("installed but NO device configured — the tool has nothing to switch on")
                ui.tip("sudo faceup emitter configure"); problems += 1
            elif e["configured"] is None:
                ui.info("configuration state unknown (run as root to check)")
            else:
                ui.ok("a device is configured")

    ui.section("Engine")
    py, dlib_ok = probe.python_shadowed()
    if dlib_ok:
        ui.ok(f"`python3` on PATH ({py}) can import dlib + cv2")
    else:
        ui.fail(f"`python3` on PATH is {py} and cannot import dlib/cv2 (a pyenv/mise/conda Python shadows the system one)"); problems += 1
    from .engine.liveness import antispoof_cnn
    if antispoof_cnn.available():
        ui.ok(f"anti-spoof model present ({cfgmod.ANTISPOOF_MODEL})")
    else:
        ui.info("anti-spoof CNN model not installed — that voter is disabled (`sudo faceup setup` can fetch it)")
    rec, live = cfg["recognition"], cfg["liveness"]
    ui.info(f"recognition: certainty={rec['certainty']} rgb_certainty={rec['rgb_certainty']} fusion_frames={rec['fusion_frames']} detector={rec['detector']}@{rec['max_height']}")
    if rec["detector"] == "cnn":
        ui.warn("detector = cnn costs ~0.5–0.9 s per frame on a laptop CPU: only 2–3 frame sets fit in the auth window")
        ui.tip("sudo faceup tune recognition.detector yunet   (4 ms per frame and pose-robust)")
        problems += 1
    elif rec["detector"] == "hog":
        ui.warn("detector = hog only finds frontal faces; YuNet is faster and handles turned/tilted heads")
        ui.tip("sudo faceup tune recognition.detector yunet")
        problems += 1
    prof = subprocess.run(["powerprofilesctl", "get"], capture_output=True, text=True).stdout.strip()
    if prof == "power-saver":
        ui.warn("power profile is power-saver: the CPU is throttled, every stage runs 1.5–2× slower")
    ui.info(f"liveness: required_votes={live['required_votes']} cross={live['cross_spectrum']} cnn={live['antispoof_cnn']} shadow={live['shadow_test']} motion={live['motion']}")
    try:
        st = storemod.load(user)
        if not st.models:
            ui.fail(f"no models enrolled for {user}"); ui.tip("sudo faceup enroll"); problems += 1
        else:
            by = {s: [m.label for m in st.by_spectrum(s)] for s in sorted(st.spectra())}
            for s, labels in by.items():
                (ui.ok if len(labels) >= 4 else ui.warn)(f"{s}: {len(labels)} model(s) for {user}: {', '.join(labels)}")
                if len(labels) < 4:
                    ui.tip("one pose ≈ one working angle — `sudo faceup enroll` adds more"); problems += 1
            missing = [s for s in ("ir", "rgb") if cfg["cameras"][s] and s not in st.spectra()]
            if missing:
                ui.warn(f"camera(s) configured but not enrolled: {', '.join(missing)} — re-enroll"); problems += 1
    except PermissionError:
        ui.info("model store needs root to read (sudo faceup doctor)")
    except storemod.StoreError as e:
        ui.fail(str(e)); problems += 1

    ui.section("PAM")
    ps = pammod.status()
    if ps["enabled"]:
        ui.ok(f"faceup-auth is enabled in {ps['file']}")
    else:
        ui.warn("faceup-auth not enabled in PAM — authentication never calls the engine"); ui.tip("sudo faceup pam enable")
        problems += 1
    if ps["howdy"]:
        ui.info("pam_howdy is also active — two face steps in a row; `sudo faceup pam enable` offers to disable it")

    print()
    if problems == 0:
        ui.ok(ui.bold("No problems found."))
    else:
        ui.warn(ui.bold(f"{problems} thing(s) to fix — `sudo faceup setup` walks through all of them."))
    return 1 if problems else 0


# ------------------------------------------------------------------ bench ----
def cmd_bench(args) -> int:
    cfg = cfgmod.load()
    dev = args.device or cfg["cameras"]["ir"] or cfg["cameras"]["rgb"]
    if not dev:
        d = cameras.pick_default(cameras.list_cameras()); dev = d.path if d else ""
    if not dev or not os.path.exists(dev):
        ui.fail("no camera to benchmark"); return 1
    ui.banner("detector benchmark", f"on a live frame from {dev}")
    b = probe.benchmark(dev)
    if b is None:
        ui.fail("needs python3-opencv and python3-dlib"); return 1
    ui.info(f"HOG (default, frontal-only):   {b.hog_ms:6.0f} ms   faces found: {b.faces_hog}")
    for h, ms in b.cnn_ms.items():
        ui.info(f"CNN at max_height={h}:         {ms:6.0f} ms   faces found: {b.faces_cnn}")
    rec = recommend_cnn(b)
    print()
    if rec:
        ui.ok(f"CNN affordable at max_height={rec}: `sudo faceup tune recognition.detector cnn` + `recognition.max_height {rec}`")
    else:
        ui.warn("CNN too slow here for a snappy unlock; stay on HOG and enroll more poses")
    return 0


def recommend_cnn(b: probe.Bench) -> int | None:
    for h in sorted(b.cnn_ms, reverse=True):
        if b.cnn_ms[h] < 900 and h >= 180:
            return h
    return None


# ------------------------------------------------------------------- tune ----
def cmd_tune(args) -> int:
    need_root("tune")
    cfg = cfgmod.load()
    if not args.key:
        print(cfgmod.dump(cfg)); return 0
    try:
        val = cfgmod.set_key(cfg, args.key, args.value)
    except ValueError as e:
        ui.fail(str(e)); return 1
    cfgmod.save(cfg)
    ui.ok(f"{args.key} = {val}")
    return 0


# ----------------------------------------------------------------- enroll ----
def cmd_enroll(args) -> int:
    need_root("enroll")
    from .engine.capture import Capture, CaptureError
    from .engine.detect import Models
    from .engine import enroll as en
    cfg = cfgmod.load()
    cams = cfg["cameras"]
    user = real_user(args)
    devices = {s: cams[s] for s in ("ir", "rgb") if cams[s]}
    if not devices:
        ui.fail("no cameras configured — run `sudo faceup setup` first"); return 1
    ui.banner("pose-guided enrollment", f"for {user} on {' + '.join(devices)} — one model per pose and spectrum")
    st = storemod.load(user)
    if st.models:
        ui.info(f"already enrolled: {', '.join(sorted({m.label for m in st.models}))}")
        if args.fresh or ui.confirm("Delete them and start clean? (required after changing cameras)", default=False):
            st.models = []
    poses = list(POSES)
    if args.all or ui.confirm("Also add the optional poses (glasses, headset, lights off)?", default=False):
        poses += OPTIONAL_POSES
    models = Models(cfg["recognition"]["detector"], cfg["recognition"]["max_height"])
    try:
        cap = Capture(devices)
    except CaptureError as e:
        ui.fail(str(e)); return 1
    done = 0
    with cap:
        cap.warm_up(5)
        if len(devices) == 2:
            ui.info("Calibrating the two cameras — look straight at the camera…")
            calib = en.calibrate(cap, models, cams)
            if calib:
                st.calib = calib; ui.ok(f"calibration dx={calib['dx']} dy={calib['dy']} s={calib['s']}")
            else:
                ui.warn("calibration failed (no face in both cameras); keeping previous values")
        if not args.guided:
            from . import gui
            ui.info("A camera window opens. Move your head slowly so the dot visits every highlighted cell;")
            ui.info("green cells are captured. Enter finishes, Esc cancels.")
            try:
                # The live preview always tracks with HOG (~5 ms); the CNN (~700 ms/frame) is an
                # auth-engine option and would make the window unusable.
                ui_models = models if cfg["recognition"]["detector"] != "cnn" else Models("yunet", 320)
                done = gui.free_move_enroll(st, cap, ui_models, cams, profile=args.profile)
            except gui.GuiUnavailable as e:
                ui.warn(f"no camera window available ({e}); falling back to the guided text flow")
                args.guided = True
            else:
                st = storemod.load(user)
                print()
                ui.ok(f"{done} model(s) added. Total: {len(st.models)} ({', '.join(f'{s}:{len(st.by_spectrum(s))}' for s in sorted(st.spectra()))})")
                if st.calib.get("pose_offset"):
                    ui.info(f"neutral pose offset: yaw {st.calib['pose_offset'][0]:+.0f}°, pitch {st.calib['pose_offset'][1]:+.0f}°")
                report_pose_calibration(st)
                want_check = args.selfcheck or ui.confirm("Run a self-check now (3 authentication attempts)?", default=True)
                cap.close()                     # the engine opens the cameras itself
                if want_check:
                    selfcheck(user, cfg)
                return 0
        ui.info("Sit as you normally do. Each pose captures for about a second — hold still until it says done.")
        existing = {m.label for m in st.models}
        for label, instruction in poses:
            if label in existing:
                ui.info(f"{label}: already enrolled, skipping"); continue
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
    if args.selfcheck or ui.confirm("Run a self-check now (3 authentication attempts)?", default=True):
        selfcheck(user, cfg)
    return 0


def report_pose_calibration(st) -> None:
    r = st.calib.get("pose_report")
    if not r:
        return
    if "error" in r:
        ui.warn(f"personal 3D face model not built: {r['error']}"); return
    ui.ok(f"personal 3D face model built from {r['frames']} frames (fit residual {r['rms_px']} px)")
    if "pitch_rms" in r:
        ui.info(f"fast estimator vs personal model — yaw: bias {r['yaw_bias']:+.1f}°, rms {r['yaw_rms']}°  ·  "
                f"pitch: bias {r['pitch_bias']:+.1f}°, rms {r['pitch_rms']}°")
        ui.tip("the personal model is used for live pose from the next enrollment on")


def cmd_calibrate(args) -> int:
    """Rebuild the personal 3D model from the landmark tracks saved by the last enrollment."""
    need_root("calibrate")
    from .engine import sfm
    user = real_user(args)
    st = storemod.load(user)
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


def selfcheck(user: str, cfg: dict) -> None:
    from .engine.auth import Engine
    eng = Engine(cfg, verbose=False)
    ok = 0
    for i in range(3):
        out = eng.authenticate(user)
        (ui.ok if out.accepted else ui.warn)(f"attempt {i + 1}: {'accepted' if out.accepted else out.reason} in {out.elapsed_ms} ms"
                                            + (f"  ir={out.ir_dist:.2f}" if out.ir_dist is not None else "")
                                            + (f"  rgb={out.rgb_dist:.2f}" if out.rgb_dist is not None else "")
                                            + (f"  votes={','.join(str(v) for v in out.votes)}" if out.votes else ""))
        ok += out.accepted
    if ok == 3:
        ui.ok(ui.bold("Self-check passed."))
    else:
        ui.warn(ui.bold(f"{ok}/3 accepted — see `faceup log` for reasons; `faceup auth -v` shows every step."))


# ------------------------------------------------------------------- auth ----
def cmd_auth(args) -> int:
    """The engine entry point. PAM calls it via /usr/local/bin/faceup-auth; exit 0 = authenticated."""
    from .engine.auth import Engine
    user = real_user(args)
    out = Engine(verbose=args.verbose).authenticate(user)
    if args.verbose or sys.stdout.isatty():
        (ui.ok if out.accepted else ui.fail)(out.logline())
    return 0 if out.accepted else 1


# -------------------------------------------------------------------- pam ----
def cmd_pam(args) -> int:
    if args.action == "status":
        for k, v in pammod.status().items():
            ui.info(f"{k}: {v}")
        return 0
    need_root("pam")
    if args.action == "enable":
        new = pammod.enable(dry_run=args.dry_run)
        ui.ok(("would write" if args.dry_run else "enabled") + f" faceup-auth in {pammod.COMMON_AUTH} (backup: {pammod.BACKUP})")
        if pammod.status()["howdy"] and (args.dry_run or ui.confirm("pam_howdy is also active. Disable it (two face steps are pointless)?", default=True)):
            new = pammod.disable_howdy(dry_run=args.dry_run, text=new if args.dry_run else None)
        if args.dry_run:
            print(new)
    elif args.action == "disable":
        pammod.disable(dry_run=args.dry_run)
        ui.ok("faceup-auth removed from PAM" + (" (dry run)" if args.dry_run else ""))
    return 0


# -------------------------------------------------------------------- log ----
def cmd_log(args) -> int:
    try:
        lines = cfgmod.LOG_PATH.read_text().splitlines()
    except OSError as e:
        ui.fail(f"cannot read {cfgmod.LOG_PATH}: {e}"); return 1
    for ln in lines[-args.n:]:
        print(ln)
    return 0


# ----------------------------------------------------------------- models ----
def cmd_models(args) -> int:
    need_root("models")
    user = real_user(args)
    st = storemod.load(user)
    if args.remove:
        n = st.remove_label(args.remove); st.save()
        ui.ok(f"removed {n} model(s) labelled {args.remove}"); return 0
    if not st.models:
        ui.warn(f"no models for {user}"); return 1
    ui.info(f"{user}: calib={st.calib}")
    for m in st.models:
        ui.info(f"#{m.id:<3} {m.label:<12} {m.spectrum:<4} quality {m.quality:.2f}  {m.time}")
    return 0


# ------------------------------------------------------------------ setup ----
def cmd_setup(args) -> int:
    need_root("setup")
    ui.banner(f"faceup {__version__} · setup", "guided and reversible — config is rewritten, models are yours, PAM is backed up")
    cfg = cfgmod.load()
    cams = cameras.list_cameras()
    if not cams:
        ui.fail("no camera found"); return 1

    ui.section("1/6 Cameras")
    for c in cams: ui.info(c.describe())
    ir = [c for c in cams if c.is_ir]
    rgb = [c for c in cams if not c.is_ir]
    cfg["cameras"]["ir"] = ir[0].path if ir else ""
    if rgb:
        best = max(rgb, key=lambda c: c.max_size[0] * c.max_size[1])
        cfg["cameras"]["rgb"] = best.path
    profile = "A (IR + RGB)" if ir and rgb else "B (IR only)" if ir else "C (RGB only)"
    ui.ok(f"profile {profile}: ir={cfg['cameras']['ir'] or '-'} rgb={cfg['cameras']['rgb'] or '-'}")
    if ir and rgb and not ui.confirm("Use both cameras? (recommended: IR for recognition, RGB for liveness)", default=True):
        cfg["cameras"]["rgb"] = ""

    ui.section("2/6 IR emitter")
    if ir:
        e = probe.emitter_status()
        L = probe.lighting(ir[0].path)
        ui.info(f"IR frame brightness now: {L.mean:.0f}/255 ({L.verdict})")
        if e["configured"]:
            ui.ok("emitter tool installed and configured")
        else:
            ui.warn("Without a working emitter, IR only works in bright rooms.")
            if ui.confirm("Install/configure linux-enable-ir-emitter now? (interactive, ~2–5 min, well-lit room)", default=not e["installed"]):
                if not e["installed"]:
                    emitter_install()
                emitter_configure(ir[0])
    else:
        ui.info("no IR camera — skipped")

    ui.section("3/6 Detector")
    from .engine.detect import yunet_available
    if yunet_available():
        cfg["recognition"]["detector"], cfg["recognition"]["max_height"] = "yunet", 320
        ui.ok("YuNet (OpenCV, ~4 ms per frame, robust to turned/tilted faces)")
    else:
        cfg["recognition"]["detector"], cfg["recognition"]["max_height"] = "hog", 320
        ui.warn("YuNet model missing — falling back to dlib HOG (frontal faces only)")

    ui.section("4/6 Anti-spoof model")
    from .engine.liveness import antispoof_cnn
    if antispoof_cnn.available():
        ui.ok("present")
    elif cfg["cameras"]["rgb"]:
        ui.info("The anti-spoof CNN needs a model file (MiniFASNet, ONNX). Not bundled yet — see README §voters.")
        ui.info("Without it, liveness uses cross-spectrum + motion (still 2 voters on profile A).")
    cfgmod.save(cfg)
    ui.ok(f"config written to {cfgmod.CONFIG_PATH}")

    ui.section("5/6 Enroll")
    if ui.confirm("Run pose-guided enrollment now?", default=True):
        args.fresh, args.all, args.selfcheck = False, False, False
        cmd_enroll(args)

    ui.section("6/6 PAM")
    ps = pammod.status()
    if ps["enabled"]:
        ui.ok("already enabled")
    elif ui.confirm(f"Enable faceup for login/sudo/polkit now? (adds one line to {pammod.COMMON_AUTH}, backup kept)", default=True):
        args.action, args.dry_run = "enable", False
        cmd_pam(args)
    print()
    ui.ok(ui.bold("Setup complete."))
    ui.tip("verify:  sudo -k && sudo true    ·  every attempt:  faceup log    ·  step by step:  sudo faceup auth -v")
    return 0


# ---------------------------------------------------------------- emitter ----
def emitter_install() -> None:
    ui.info("Fetching the latest linux-enable-ir-emitter release…")
    with urllib.request.urlopen(f"https://api.github.com/repos/{EMITTER_REPO}/releases/latest", timeout=30) as r:
        rel = json.load(r)
    asset = next((a for a in rel["assets"] if a["name"].endswith(".systemd.x86-64.tar.gz")), None)
    if not asset:
        ui.fail("no systemd x86-64 asset in the latest release"); return
    ui.info(f"{rel['tag_name']}: {asset['name']}")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        urllib.request.urlretrieve(asset["browser_download_url"], tmp.name)
        subprocess.check_call(["tar", "-C", "/", "--no-same-owner", "-m", "-h", "-xzf", tmp.name])
    os.unlink(tmp.name)
    subprocess.call(["systemctl", "enable", "--now", "linux-enable-ir-emitter"])
    ui.ok("installed; service enabled (runs at boot and after suspend)")


def emitter_configure(cam) -> None:
    w, h = cam.max_size
    ui.info("Sit close, room well lit, no other app using the camera. Answer Y in the video window when it flashes. Do not kill it.")
    rc = subprocess.call(["linux-enable-ir-emitter", "--device", cam.path, "--width", str(w), "--height", str(h), "configure"])
    (ui.ok if rc == 0 else ui.warn)("configure finished" if rc == 0 else "configure did not succeed — see the tool's README for --manual / --limit -1")


def cmd_emitter(args) -> int:
    ir = [c for c in cameras.list_cameras() if c.is_ir]
    if not ir:
        ui.fail("no IR camera on this machine"); return 1
    cam = ir[0]
    if args.action in ("test", "check"):
        from . import gui
        from .engine.capture import Capture, CaptureError
        from .engine.detect import Models
        cfg = cfgmod.load()
        devices = {"ir": cam.path}
        rgb = cfg["cameras"]["rgb"] or next((c.path for c in cameras.list_cameras() if not c.is_ir), "")
        if rgb:
            devices["rgb"] = rgb
        ui.info("Turn the room lights OFF. If the colour camera sees darkness while the IR camera still sees your face, the emitter works.")
        try:
            with Capture(devices) as cap:
                cap.warm_up(8)
                state, msg = gui.emitter_check(cap, Models(), gui=(args.action == "check"), seconds=2.0)
        except CaptureError as e:
            ui.fail(str(e)); return 1
        {"ok": ui.ok, "dead": ui.fail, "inconclusive": ui.warn}.get(state, ui.info)(msg)
        if state == "dead":
            ui.tip("sudo faceup emitter configure")
        return 0 if state == "ok" else 1
    need_root("emitter")
    if args.action == "install":
        if probe.emitter_status()["installed"]:
            ui.info("already installed")
        else:
            emitter_install()
        emitter_configure(cam)
    elif args.action == "configure":
        emitter_configure(cam)
    return 0


# ------------------------------------------------------------------- main ----
def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="faceup", description="Face unlock for Linux that actually works: dual-spectrum recognition with liveness.")
    p.add_argument("--version", action="version", version=f"faceup {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="diagnose cameras, emitter, engine, models, PAM"); d.add_argument("-U", "--user"); d.set_defaults(fn=cmd_doctor)
    s = sub.add_parser("setup", help="guided setup: cameras → emitter → detector → model → enrollment → PAM (root)"); s.add_argument("-U", "--user"); s.set_defaults(fn=cmd_setup)

    e = sub.add_parser("enroll", help="pose-guided, dual-spectrum enrollment (root)")
    e.add_argument("-U", "--user"); e.add_argument("--fresh", action="store_true", help="delete existing models first")
    e.add_argument("--all", action="store_true", help="include optional poses without asking")
    e.add_argument("--selfcheck", action="store_true", help="run the self-check without asking")
    e.add_argument("--guided", action="store_true", help="scripted 8-pose text flow instead of the free-move camera window")
    e.add_argument("--profile", action="store_true", help="print per-stage timings once a second (diagnose a laggy preview)")
    e.set_defaults(fn=cmd_enroll)

    t = sub.add_parser("tune", help="show config, or set section.key VALUE (root)")
    t.add_argument("key", nargs="?"); t.add_argument("value", nargs="?"); t.set_defaults(fn=cmd_tune)

    b = sub.add_parser("bench", help="time HOG vs CNN detectors on a live frame"); b.add_argument("--device"); b.set_defaults(fn=cmd_bench)

    a = sub.add_parser("auth", help="run the authentication engine once (what PAM calls); exit 0 = accepted")
    a.add_argument("-U", "--user"); a.add_argument("-v", "--verbose", action="store_true"); a.set_defaults(fn=cmd_auth)

    pm = sub.add_parser("pam", help="enable | disable | status the PAM integration (root)")
    pm.add_argument("action", choices=["enable", "disable", "status"]); pm.add_argument("--dry-run", action="store_true"); pm.set_defaults(fn=cmd_pam)

    lg = sub.add_parser("log", help="recent authentication attempts"); lg.add_argument("-n", type=int, default=20); lg.set_defaults(fn=cmd_log)

    cb = sub.add_parser("calibrate", help="rebuild the personal 3D face model from the last enrollment's landmark tracks (root)")
    cb.add_argument("-U", "--user"); cb.set_defaults(fn=cmd_calibrate)

    m = sub.add_parser("models", help="list or remove enrolled models (root)")
    m.add_argument("-U", "--user"); m.add_argument("--remove", metavar="LABEL"); m.set_defaults(fn=cmd_models)

    em = sub.add_parser("emitter", help="IR emitter: check (camera window) | test (text) | install | configure")
    em.add_argument("action", choices=["check", "test", "install", "configure"]); em.set_defaults(fn=cmd_emitter)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        print(); return 130


if __name__ == "__main__":
    sys.exit(main())
