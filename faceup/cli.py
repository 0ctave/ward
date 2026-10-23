"""faceup command line — see docs/DESIGN.md §11."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

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
    ("glasses", "Wearing your glasses (or without them, if you registered with them on)."),
    ("headset", "Wearing headphones / a headset."),
    ("dark",    "Lights OFF — verifies the IR emitter is doing the work."),
]


def need_root(what: str) -> None:
    if not probe.is_root() and "FACEUP_ETC" not in os.environ:
        ui.fail(f"{what} needs root (config and models live under /etc/faceup).")
        ui.tip(f"run:  sudo faceup {' '.join(sys.argv[1:])}")
        sys.exit(2)


def real_user(args) -> str:
    from .manage import invoking_user
    return getattr(args, "user", None) or os.environ.get("PAM_USER") or invoking_user()


# ----------------------------------------------------------------- doctor ----
def cmd_doctor(args) -> int:
    ui.banner(f"faceup {__version__} · doctor", "Everything that decides whether face unlock works on this machine")
    problems = 0
    cfg = cfgmod.load()
    user = real_user(args)

    ui.section("Cameras")
    cams, unusable = cameras.list_cameras(include_unusable=True)
    if not cams and not unusable:
        ui.fail("no /dev/video* device at all — no camera driver bound (check `lsusb` / `dmesg | grep -i video`)"); problems += 1
    elif not cams:
        ui.fail("no usable camera"); problems += 1
    for c in cams:
        (ui.ok if c.is_ir else ui.info)(f"{c.describe()}   id {c.identity}  [{c.driver or c.kind}]")
    for u in unusable:
        ui.warn(f"{u.path}  {u.name}: {u.reason}"); problems += 1
    ir = [c for c in cams if c.is_ir]
    paths, notes = cfgmod.camera_paths(cfg)
    if cfgmod.CONFIG_PATH.exists():
        ui.ok(f"config: ir = {cfg['cameras']['ir'] or '(none)'} → {paths.get('ir', '-')}, rgb = {cfg['cameras']['rgb'] or '(none)'} → {paths.get('rgb', '-')}")
        for n in notes:
            ui.warn(n); problems += 1
        if ir and not cfg["cameras"]["ir"]:
            ui.warn(f"an IR camera exists ({ir[0].path}) but is not configured"); problems += 1
        if any(str(cfg["cameras"][r]).startswith("/dev/") for r in ("ir", "rgb")):
            ui.warn("config stores /dev paths, which change when cameras re-enumerate — `sudo faceup setup` stores stable USB ids"); problems += 1
    else:
        ui.warn(f"no config yet ({cfgmod.CONFIG_PATH}) — run `sudo faceup setup`"); problems += 1
    light = {}
    for s in ("ir", "rgb"):
        dev = paths.get(s, "")
        if dev and os.path.exists(dev):
            L = probe.lighting(dev); light[s] = L
            face = f", face {L.face_mean:.0f}" if L.face_mean is not None else ", no face in view"
            line = f"{s} lighting now: frame mean {L.mean:.0f}/255{face}"
            if L.frames == 0:
                ui.fail(f"{s}: camera gave no frames (busy or permission denied)"); problems += 1
            elif s == "ir" and L.face_mean is not None and L.face_mean < 60:
                ui.warn(line + " → the face is dark to the IR camera"); problems += 1
            elif s == "ir" and L.face_mean is None:
                ui.info(line + " (sit in front of the camera for a face reading)")
            elif s == "rgb" and L.verdict in ("dark", "dim"):
                ui.warn(line + f" → {L.verdict} (shutter closed? night?)")
            else:
                ui.ok(line)
    if "ir" in light and light["ir"].face_mean is not None:
        if light["ir"].face_mean >= 60 and light["ir"].mean < 60:
            ui.ok("IR: lit face on a dark background — the emitter is doing the work")
        elif light["ir"].face_mean < 60:
            ui.tip("a lit face with a dark IR frame means a working emitter; a DARK face means it is off or too far")

    if ir:
        ui.section("IR emitter")
        from .engine.emitter import config_instruction as emitmod_config_instruction
        e = probe.emitter_status()
        if not e["installed"]:
            ui.warn("not installed — if the IR preview is black in a dark room, the emitter isn't firing")
            ui.tip("sudo faceup emitter install")
        else:
            (ui.ok if e["enabled"] else ui.warn)("service " + ("enabled (re-applies at boot and resume)" if e["enabled"] else "NOT enabled"))
            own = emitmod_config_instruction(cfg)
            if own is not None:
                ui.ok(f"LED control from `faceup emitter probe`: unit {own.unit} selector {own.selector} on {list(own.on)} off {list(own.off)}")
                ms = int(cfg["liveness"].get("shadow_toggle_ms", -1))
                if ms < 0:
                    ui.info("shadow-test liveness not yet benchmarked"); ui.tip("sudo faceup emitter toggle-bench")
                else:
                    (ui.ok if ms <= 150 else ui.warn)(f"emitter toggle latency {ms} ms → shadow test {'active' if ms <= 150 else 'too slow, disabled'}")
            elif e.get("state") == "absent":
                ui.fail("installed but NO device configured — the tool has nothing to switch on")
                ui.tip("sudo faceup emitter configure"); problems += 1
            elif e.get("state") == "none":
                ui.info("the tool has no working instruction (its search found none — on modules whose LEDs are on by default it never needs one)")
                ui.tip("sudo faceup emitter probe   finds the LED control by measurement, in any lighting (needed for the shadow-test liveness voter)")
            elif e["configured"] is None:
                ui.info("configuration state unknown (run as root to check)")
            else:
                ui.ok("a working emitter instruction is configured")
                ms = int(cfg["liveness"].get("shadow_toggle_ms", -1))
                if ms < 0:
                    ui.info("shadow-test liveness not yet benchmarked"); ui.tip("sudo faceup emitter toggle-bench")
                else:
                    (ui.ok if ms <= 150 else ui.warn)(f"emitter toggle latency {ms} ms → shadow test {'active' if ms <= 150 else 'too slow, disabled'}")
                # roadmap L1: measure instead of trusting the config — re-apply the emitter
                # instructions and meter the IR frame again (root only; harmless re-run of the tool)
                if probe.is_root() and "ir" in light and paths.get("ir") and e.get("state") == "working":
                    before = light["ir"].mean
                    subprocess.run(["linux-enable-ir-emitter", "run"], capture_output=True, text=True, timeout=20)
                    after = probe.lighting(paths["ir"]).mean
                    line = f"emitter re-applied: IR mean {before:.0f} → {after:.0f}"
                    if after >= 50 or after > before * 1.5:
                        ui.ok(line)
                    else:
                        ui.warn(line + " — no measurable effect; run `sudo faceup emitter check` with the lights off"); problems += 1

    ui.section("Hardware")
    from . import hw
    hp = hw.profile()
    ui.info(f"{hp.cpu} · {hp.cores} threads · AVX2 {'yes' if hp.avx2 else 'no'} · OpenCV {hp.opencv} · OpenCL {('yes: ' + hp.opencl_device) if hp.opencl else 'no'}")
    for c in cams:
        try:
            import cv2
            capx = cv2.VideoCapture(int(c.path.replace('/dev/video', '')), cv2.CAP_V4L2)
            from .engine.brightness import Exposure
            e = Exposure(capx, c.role); desc = e.describe(); e.restore(); capx.release()
            ui.info(f"{c.path} ({c.role}): {desc}")
        except Exception as ex:
            ui.info(f"{c.path}: exposure controls unknown ({ex})")
    for n in hp.notes:
        ui.warn(n)

    ui.section("Engine")
    py, dlib_ok = probe.python_shadowed()
    if dlib_ok:
        ui.ok(f"`python3` on PATH ({py}) can import dlib + cv2")
    else:
        ui.fail(f"`python3` on PATH is {py} and cannot import dlib/cv2 (a pyenv/mise/conda Python shadows the system one)"); problems += 1
    from .engine.liveness import antispoof_cnn
    _as_ok, _ = antispoof_cnn.verify()
    if not _as_ok:
        ui.fail(f"anti-spoof CNN model missing or corrupt ({cfgmod.ANTISPOOF_MODEL}) — reinstall faceup (sudo ./install.sh)"); problems += 1
    elif not cfg["liveness"]["antispoof_cnn"]:
        ui.warn("anti-spoof CNN (bundled MiniFASNetV2, daylight liveness) is switched off"); ui.tip("sudo faceup antispoof on")
    else:
        ui.ok(f"anti-spoof CNN: MiniFASNetV2 bundled, checksum ok ({cfgmod.ANTISPOOF_MODEL.name})")
    rec, live = cfg["recognition"], cfg["liveness"]
    from .engine import recognise
    rk = rec.get("recogniser", "dlib")
    if rk == "arcface":
        if recognise.available():
            ui.ok(f"recogniser: ArcFace {recognise.installed_variant()} via OpenCV DNN (512-d; distances scaled to dlib units ×{recognise.scale_factor(recognise.ARCFACE_RADIUS):.2f})")
        else:
            ui.fail("recogniser = arcface but the model is not installed"); ui.tip("sudo faceup recogniser install"); problems += 1
    else:
        ui.info("recogniser: dlib ResNet (128-d, trained on colour photos)" + ("" if not recognise.available() else " — ArcFace is installed; `sudo faceup recogniser use arcface` + re-enroll to switch"))
    ui.info(f"recognition: certainty={rec['certainty']} rgb_certainty={rec['rgb_certainty']} fusion_frames={rec['fusion_frames']} detector={rec['detector']}@{rec['max_height']}")
    if int(cfg["timing"].get("settle_ms", 0)) > 0:
        ui.warn(f"timing.settle_ms = {cfg['timing']['settle_ms']}: every attempt waits for auto-exposure first (~400 ms measured); YuNet does not need it")
        ui.tip("sudo faceup tune timing.settle_ms 0"); problems += 1
    if float(rec["certainty"]) > 0.45:
        ui.warn(f"certainty {rec['certainty']} is loose (default 0.40, dlib's same-person radius is 0.6): a look-alike gets closer than that")
        ui.tip("sudo faceup tune recognition.certainty 0.40   — then re-enroll or add conditions if you get rejects; whitening is meant to absorb lighting, not the gate"); problems += 1
    if rec["detector"] == "cnn":
        ui.warn("detector = cnn costs ~0.5–0.9 s per frame on a laptop CPU: only 2–3 frame sets fit in the auth window")
        ui.tip("sudo faceup tune recognition.detector yunet   (4 ms per frame and pose-robust)")
        problems += 1
    elif rec["detector"] == "hog":
        ui.warn("detector = hog only finds frontal faces; YuNet is faster and handles turned/tilted heads")
        ui.tip("sudo faceup tune recognition.detector yunet")
        problems += 1
    ui.info(f"liveness: required_votes={live['required_votes']} cross={live['cross_spectrum']} cnn={live['antispoof_cnn']} shadow={live['shadow_test']} motion={live['motion']}")
    from .engine import policy as polmod
    from .engine.auth import Engine as _Eng
    lvl = polmod.level_for(cfg)
    ui.info(f"policy: {lvl} — {polmod.describe(lvl)}" + "".join(f"; {k[7:]}: {v}" for k, v in cfg.get("security", {}).items() if k.startswith("policy_") and v))
    # which liveness voter can PASS for a STILL face on this hardware profile?
    has_ir, has_rgb = bool(cfg["cameras"]["ir"]), bool(cfg["cameras"]["rgb"])
    still_voters = []
    if has_ir and has_rgb and live["cross_spectrum"]:
        still_voters.append("cross-spectrum")
    if has_ir and _Eng._shadow_enabled(live):
        still_voters.append("shadow test")
    if has_rgb and live["antispoof_cnn"] and antispoof_cnn.available():
        still_voters.append("anti-spoof CNN")
    if still_voters:
        ui.ok("liveness for a still face: " + ", ".join(still_voters))
    else:
        ui.warn("no liveness voter can PASS for a STILL face on this profile: unlock asks you to move your head a little (motion voter)")
        if has_ir:
            ui.tip("sudo faceup emitter probe && sudo faceup emitter toggle-bench   (enables the shadow test)")
        elif not live.get("require_liveness_when_available", True):
            ui.info("degraded rule active: recognition alone unlocks (convenience-grade)")
        else:
            ui.tip("RGB-only: without the anti-spoof model, either nod at the camera, or accept recognition alone: sudo faceup tune liveness.require_liveness_when_available false")
    try:
        st = storemod.load(user, recognise.DLIB_NAME if rk != "arcface" else f"arcface:{recognise.installed_variant() or 'unknown'}")
        if not st.models:
            ui.fail(f"no face registered for {user}"); ui.tip("sudo faceup enroll"); problems += 1
        else:
            by = {s: [m.label for m in st.by_spectrum(s)] for s in sorted(st.spectra())}
            for s, labels in by.items():
                (ui.ok if len(labels) >= 4 else ui.warn)(f"{s}: {len(labels)} model(s) for {user}: {', '.join(labels)}")
                if len(labels) < 4:
                    ui.tip("one pose ≈ one working angle — `sudo faceup enroll` adds more"); problems += 1
            missing = [s for s in ("ir", "rgb") if cfg["cameras"][s] and s not in st.spectra()]
            if missing:
                ui.warn(f"camera(s) configured but not registered: {', '.join(missing)} — register again"); problems += 1
            from .engine import calibration as cal
            c = st.calib.get("calibration")
            if c:
                ui.ok(f"calibrated {c['date'][:10]}: gate " + ", ".join(f"{k} {v}" for k, v in c.get("gate", {}).items())
                      + " from " + ", ".join(f"{k} {v}" for k, v in c.get("n", {}).items()) + " measurements"
                      + (f"; whitening threshold refreshed {st.calib['last_refresh'][:10]}" if st.calib.get("last_refresh") else ""))
                if float(cfg["recognition"]["certainty"]) != float(c.get("gate", {}).get("ir", cfg["recognition"]["certainty"])):
                    ui.warn(f"certainty {cfg['recognition']['certainty']} was changed by hand after calibration set {c['gate'].get('ir')}")
            else:
                ui.warn("thresholds never calibrated on this machine — the gate is a constant, not a measurement"); ui.tip("sudo faceup calibrate"); problems += 1
            adv = cal.gate_advice(st, cfg)
            if adv:
                ui.info(adv)
            learned = [m for m in st.models if m.learned]; enrolled = [m for m in st.models if not m.learned]
            if learned and len(learned) >= len(enrolled):
                ui.warn(f"{len(learned)} learned models vs {len(enrolled)} registered — online learning is churning")
                ui.tip("sudo faceup models --forget-learned   (novelty bar is now 0.25; old stores learned on every accept)"); problems += 1
            elif learned:
                ui.info(f"{len(learned)} learned model(s) from use (night, glasses…)")
    except PermissionError:
        ui.info("model store needs root to read (sudo faceup doctor)")
    except storemod.StoreError as e:
        ui.fail(str(e)); ui.tip("sudo faceup enroll --fresh"); problems += 1

    ui.section("Compatibility")
    import sys as _sys, shutil as _sh
    from .engine import keyring as _kr, detect as _det
    ui.info(f"python {_sys.version.split()[0]} · opencv {__import__('cv2').__version__}"
            f" (YuNet {'yes' if _det.yunet_available() else 'no → HOG'}) · dlib {'yes' if _det.dlib is not None else 'no (ArcFace + YuNet path only)'}")
    try:
        import gi; gi.require_version("Adw", "1"); from gi.repository import Adw  # noqa
        ui.info("GTK 4 + libadwaita: yes (windows) ")
    except Exception:
        ui.warn("GTK 4 + libadwaita missing: enroll/calibrate fall back to text (--guided / --text)"); ui.tip(__import__("faceup.distro", fromlist=["hint"]).hint("gtk"))
    ui.info(f"profile: {'A (IR + RGB)' if cfg['cameras']['ir'] and cfg['cameras']['rgb'] else 'B (IR only)' if cfg['cameras']['ir'] else 'C (RGB only)'}"
            f" · gcc {'yes' if _sh.which('gcc') else 'no (pam_exec fallback, no keyring token)'} · TPM {'yes' if _kr.has_tpm() else 'no'} · encrypted root {'yes' if _kr.root_is_encrypted() else 'no'}")

    ui.section("Attempts")
    from . import attempts
    rows = attempts.read(cfgmod.LOG_PATH, 20)
    if not rows:
        ui.info("no attempts logged yet (or the log is not readable: group adm)")
    else:
        sm = attempts.summary(rows)
        (ui.ok if sm["accept_rate"] >= 0.9 else ui.warn)(f"last {sm['attempts']}: {sm['accepted']} accepted" + (f", failures: {', '.join(f'{k} {v}' for k, v in sm['reasons'].items())}" if sm["reasons"] else ""))
        if sm["median_accept_ms"] is not None:
            if sm["median_accept_ms"] > 800:
                ui.warn(f"median time to accept {sm['median_accept_ms']} ms — slow (target < 800 ms)"); ui.tip("faceup log --summary"); problems += 1
            else:
                ui.ok(f"median time to accept {sm['median_accept_ms']} ms")

    ui.section("PAM")
    ps = pammod.status()
    if ps["enabled"] and not ps["current"]:
        ui.warn(f"faceup-auth line in {ps['file']} is from an older version (no stdout/seteuid): no on-screen feedback")
        ui.tip("sudo faceup pam enable   (upgrades the line in place)"); problems += 1
    elif ps["enabled"]:
        ui.ok(f"faceup is enabled in {ps['file']} via " + ("the native module pam_faceup.so (messages + keyring token)" if ps.get("native") else "pam_exec (with on-screen feedback)"))
    else:
        ui.warn("faceup-auth not enabled in PAM — authentication never calls the engine"); ui.tip("sudo faceup pam enable")
        problems += 1
    if ps["howdy"]:
        ui.info("pam_howdy is also active — two face steps in a row; `sudo faceup pam enable` offers to disable it")
    from .engine import keyring as kr
    ks = kr.status(user)
    dis = kr.disabled_keyring_lines()
    if dis:
        ui.fail(f"keyring: {len(dis)} pam_gnome_keyring/pam_kwallet line(s) commented out by an earlier faceup experiment ({', '.join(sorted({p.name for p, _ in dis}))}) — even PASSWORD logins leave the keyring locked")
        ui.tip("sudo faceup keyring repair"); problems += 1
    if ps["enabled"]:
        if ps.get("native") and ks["enrolled"]:
            ekey = kr.enrolled_key(user)
            if ekey == "tpm2":
                ui.warn("keyring: unlocked by a TPM2-sealed credential — a firmware TPM adds 2–4 s to every login")
                ui.tip("your disk is encrypted, so the fast host key is safe: sudo faceup keyring enable   (re-seal, no TPM delay)")
            else:
                ui.ok(f"keyring: a face login unlocks it instantly (host-key credential; native PAM module)")
        elif ps.get("native") and ks["enrolled"] is None:
            ui.info("keyring: native module active; credential state needs root to read (sudo faceup doctor)")
        elif cfg["behaviour"].get("skip_login"):
            ui.info("keyring: the first login of a session uses the password (behaviour.skip_login); unlock, sudo and polkit use the face")
        else:
            ui.info("keyring: after a FACE login the login keyring stays locked until first use (it needs the password)")
            ui.tip("sudo faceup keyring enable   (sealed credential + native PAM module)   ·   or keep login on the password: sudo faceup tune behaviour.skip_login true")
        if not ps.get("native"):
            ui.info("PAM line is pam_exec; the native module (built by install.sh when gcc is present) adds the keyring token" + ("" if pammod.native_available() else " — gcc missing?"))
            if pammod.native_available():
                ui.tip("sudo faceup pam enable   (switches to the installed native module)")

    print()
    if problems == 0:
        ui.ok(ui.bold("No problems found."))
    else:
        ui.warn(ui.bold(f"{problems} thing(s) to fix — `sudo faceup setup` walks through all of them."))
    return 1 if problems else 0


# ------------------------------------------------------------------ bench ----
def cmd_bench(args) -> int:
    cfg = cfgmod.load()
    paths, _ = cfgmod.camera_paths(cfg)
    dev = args.device or paths.get("ir") or paths.get("rgb")
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
    from .engine import policy as polmod
    if not args.key:
        print(cfgmod.dump(cfg))
        lvl = polmod.level_for(cfg)
        ui.info(f"policy: {lvl} — {polmod.describe(lvl)}"); return 0
    if args.key == "policy":
        if not args.value:
            for lvl in polmod.LEVELS:
                (ui.ok if lvl == polmod.level_for(cfg) else ui.info)(f"{lvl:<12} {polmod.describe(lvl)}")
            return 0
        try:
            changed = polmod.apply_preset(cfg, args.value)
        except ValueError as e:
            ui.fail(str(e)); return 1
        cfgmod.save(cfg)
        ui.ok(f"policy = {args.value} — {polmod.describe(args.value)}")
        if args.value == "strict":
            ui.tip("strict needs a shadow-test or anti-spoof PASS: run `sudo faceup emitter probe` + `toggle-bench` if not done")
        return 0
    try:
        val = cfgmod.set_key(cfg, args.key, args.value)
    except ValueError as e:
        ui.fail(str(e)); return 1
    cfgmod.save(cfg)
    ui.ok(f"{args.key} = {val}")
    return 0


# ----------------------------------------------------------------- enroll ----
def _display_available() -> bool:
    """A graphical session the registration window can open on (directly or via the sudo/pkexec user's socket)."""
    from . import display
    return bool(os.environ.get(display.STREAM_ENV) or os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY") or display.user_wayland_env())


def _stream_ui(args) -> None:
    """--stream-ui: the manager app draws the window; snapshots go to stdout (display.StreamDisplay)."""
    if getattr(args, "stream_ui", False):
        from . import display
        os.environ[display.STREAM_ENV] = "1"
        display.stream_out()                     # take stdout now: every message goes to the app's console


def cmd_enroll(args) -> int:
    need_root("enroll")
    _stream_ui(args)
    for k in ("fresh", "all", "selfcheck", "guided", "profile"):
        if not hasattr(args, k):
            setattr(args, k, False)
    from .engine.capture import Capture, CaptureError
    from .engine.detect import Models
    from .engine import enroll as en
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
            from . import gui
            if not getattr(args, "stream_ui", False):
                ui.info("A window opens. Follow the arrow to bring your face into each target ring; it registers")
                ui.info("your face and tunes the gate in one pass. Enter skips a pose, Esc cancels.")
            try:
                ui_models = models if cfg["recognition"]["detector"] != "cnn" else Models("yunet", 320, recogniser=models.recogniser_kind)
                from .engine import calibration as cal
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
    from .engine import calibration as cal
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
    from .engine import calibration as cal
    from .engine import lab as labmod
    from .engine.capture import CaptureError
    seconds = seconds or cal.PASS_TIMEOUT_S
    ui.banner("calibration session", f"for {user}: {len(cal.PROMPTS)} guided passes measure your real distances; the gate and the whitening threshold come from them")
    if not text:
        # the window: cameras + models here, embeddings in a worker process (gui.calibration_session)
        from . import gui
        from .engine.capture import Capture
        from .engine.detect import Models
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
    from .engine import calibration as cal
    res = cal.summarise(measures)
    if not res.ok():
        for w in res.warnings: ui.warn(w)
        ui.fail("not enough measurements to calibrate — nothing changed"); return 1
    changed = cal.apply(st, cfg, res, measures, save=True)
    cfgmod.save(cfg)
    _report_calibration(res, changed, st)
    return 0


def _report_calibration(res, changed: dict, st) -> None:
    from .engine import calibration as cal
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
    from .engine import sfm
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
    from .engine.auth import Engine
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


# ------------------------------------------------------------------- auth ----
def cmd_auth(args) -> int:
    """The engine entry point. PAM calls it via /usr/local/bin/faceup-auth; exit 0 = authenticated."""
    from . import feedback
    user = real_user(args)
    cfg = cfgmod.load()
    # golden rule 3 backstop: whatever wedges (a camera stuck in the driver, a hung model load),
    # the KERNEL ends this process — SIGALRM with its default action needs no Python handler,
    # which dlib holding the GIL could starve. pam_exec has no timeout of its own.
    import signal
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(int(cfg["timing"].get("hard_timeout_ms", 4000)) // 1000 + 3)
    if getattr(args, "replay", None):
        os.environ["FACEUP_REPLAY"] = args.replay
    # say "looking…" before the heavy imports: the greeter shows it while cv2/dlib load
    fb = feedback.Feedback(user, cfg["behaviour"], verbose=args.verbose)
    if cfg["behaviour"].get("skip_login") and feedback.is_session_login(user):
        # roadmap K0: the first login of a session is left to the password so pam_gnome_keyring /
        # pam_kwallet can unlock the keyring; unlock, sudo and polkit still use the face
        from .engine.auth import Outcome, log
        out = Outcome(False, "login", user, detail="skip_login: keyring stays unlockable"); log(out)
        return 1
    fb.start()
    from .engine.auth import Engine
    eng = Engine(cfg, verbose=args.verbose)
    eng.events = fb.event
    out = eng.authenticate(user)
    fb.finish(out)
    if args.verbose:
        (ui.ok if out.accepted else ui.fail)(out.logline())
    return 0 if out.accepted else 1


# ----------------------------------------------------------------- daemon ----
UNIT = "faceup.service"


def daemon_state() -> dict:
    """{'installed', 'enabled', 'active', 'answers'} — answers needs root (the socket is 0600)."""
    import shutil
    import subprocess
    st = {"installed": os.path.exists(f"/etc/systemd/system/{UNIT}"), "enabled": False, "active": False, "answers": None}
    if shutil.which("systemctl"):
        for k, verb in (("enabled", "is-enabled"), ("active", "is-active")):
            try:
                st[k] = subprocess.run(["systemctl", verb, "--quiet", UNIT], timeout=3).returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                pass
    if os.geteuid() == 0:
        from . import client
        s = client.connect()
        st["answers"] = s is not None
        if s is not None:
            s.close()
    return st


def cmd_daemon(args) -> int:
    """run (what systemd starts) | status | enable | disable — the resident engine (faceup/daemon.py)."""
    import subprocess
    if args.action == "run":
        need_root("daemon run")
        from . import daemon
        return daemon.serve()
    if args.action == "status":
        st = daemon_state()
        if st["active"]:
            ui.ok("resident engine running — unlocks skip Python start-up and model loading" + ("" if st["answers"] is not False else " (but the socket does not answer)"))
        elif st["installed"]:
            ui.warn("resident engine installed but not running: unlocks load everything from scratch (~1 s slower)")
            ui.tip("sudo faceup daemon enable")
        else:
            ui.info("resident engine not installed (re-run install.sh)")
        return 0 if st["active"] else 1
    need_root(f"daemon {args.action}")
    if args.action == "enable":
        rc = subprocess.run(["systemctl", "enable", "--now", UNIT]).returncode
        (ui.ok if rc == 0 else ui.fail)("resident engine " + ("enabled and started (~150 MB of RAM for the loaded models)" if rc == 0 else "could not be started: journalctl -u faceup"))
        return rc
    rc = subprocess.run(["systemctl", "disable", "--now", UNIT]).returncode
    (ui.ok if rc == 0 else ui.fail)("resident engine stopped; unlocks run the engine from scratch" if rc == 0 else "could not stop faceup.service")
    return rc


# -------------------------------------------------------------------- pam ----
def cmd_pam(args) -> int:
    try:
        return _cmd_pam(args)
    except pammod.PamError as e:
        ui.fail(str(e)); return 1


def _cmd_pam(args) -> int:
    if args.action == "status":
        for k, v in pammod.status().items():
            ui.info(f"{k}: {v}")
        return 0
    need_root("pam")
    if args.action == "enable":
        native = pammod.native_available()
        new = pammod.enable(dry_run=args.dry_run, native=native)
        ui.ok(("would write" if args.dry_run else "enabled") + f" faceup in {pammod.COMMON_AUTH} via "
              + (f"{pammod.MODULE} (native: messages + keyring token)" if native else "pam_exec (no keyring unlock; install gcc and re-run install.sh for the native module)")
              + f" (backup: {pammod.BACKUP})")
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
    from . import attempts
    path = cfgmod.LOG_PATH
    if not os.access(path, os.R_OK):
        ui.fail(f"cannot read {path} (group adm, or run with sudo)"); return 1
    if args.follow:
        ui.info(f"following {path} — Ctrl+C to stop")
        try:
            for r in attempts.follow(path):
                print(json.dumps(r) if args.json else format_attempt(r), flush=True)
        except KeyboardInterrupt:
            print()
        return 0
    rows = attempts.read(path, None if args.summary else args.n)
    if args.summary:
        s = attempts.summary(rows[-args.n:] if args.n and args.n < len(rows) and args.n != 20 else rows)
        if args.json:
            print(json.dumps(s, indent=1)); return 0
        ui.banner("faceup · attempts", f"{s['attempts']} attempts in {path}")
        if not s["attempts"]:
            ui.info("nothing logged yet"); return 0
        (ui.ok if s["accept_rate"] >= 0.9 else ui.warn)(f"accepted {s['accepted']}/{s['attempts']} ({s['accept_rate'] * 100:.0f} %)")
        if s["median_accept_ms"] is not None:
            (ui.ok if s["median_accept_ms"] <= 800 else ui.warn)(
                f"time to accept: median {s['median_accept_ms']} ms · all attempts median {s['median_ms']} ms, p90 {s['p90_ms']} ms"
                + ("" if s["median_accept_ms"] <= 800 else "   (target < 800 ms — see docs/ROADMAP.md §1)"))
        if s["ir_median"] is not None:
            ui.info(f"IR distance on accepts: median {s['ir_median']} (gate {cfgmod.load()['recognition']['certainty']})")
        for reason, n in s["reasons"].items():
            ui.info(f"{reason:<10} {n}")
        for svc, d in s["services"].items():
            ui.info(f"{svc:<14} {d['accepted']}/{d['attempts']} accepted")
        if s["learned"]:
            ui.info(f"learned models added: {s['learned']}" + ("   (churn: `sudo faceup models --forget-learned`)" if s["learned"] > 5 else ""))
        if s["spoof"]:
            ui.warn(f"liveness FAILs: {s['spoof']}")
        return 0
    for r in rows:
        print(json.dumps(r) if args.json else format_attempt(r))
    return 0


def format_attempt(r: dict) -> str:
    """One coloured line per attempt."""
    mark = ui.green("✔") if r["accepted"] else (ui.red("✘") if r["reason"] == "spoof" else ui.yellow("✘"))
    what = "accept" if r["accepted"] else r.get("result", "?")
    parts = [r["time"], mark, f"{what:<12}"]
    for k in ("ir", "rgb"):
        if r.get(k) is not None:
            parts.append(f"{k} {r[k]:.2f}")
    if r["votes"]:
        parts.append(" ".join(v.replace(":PASS", " ✓").replace(":FAIL", " ✗").replace(":ABSTAIN", " –") for v in r["votes"]))
    if r["ms"] is not None:
        parts.append(f"{r['ms']} ms")
    if r["service"]:
        parts.append(ui.dim(r["service"]))
    if r.get("learned"):
        parts.append(ui.dim(f"learned {r['learned']}"))
    if r.get("detail") and not r["service"]:
        parts.append(ui.dim(r["detail"]))
    return "  ".join(parts)


# ----------------------------------------------------------------- models ----
def cmd_models(args) -> int:
    need_root("models")
    user = real_user(args)
    try:
        st = storemod.load(user, _recogniser_name(cfgmod.load()))
    except storemod.StoreError as e:
        ui.fail(str(e)); ui.tip("sudo faceup enroll --fresh"); return 1
    if args.remove:
        n = st.remove_label(args.remove); st.save()
        ui.ok(f"removed {n} model(s) labelled {args.remove}"); return 0
    if getattr(args, "remove_identity", None) is not None:
        n = st.remove_identity(args.remove_identity)
        for sp in st.spectra(): st.fit_whitening(sp)
        st.save(); ui.ok(f"removed {n} model(s) of identity {args.remove_identity!r}"); return 0
    if getattr(args, "remove_condition", None):
        n = st.remove_condition(args.remove_condition, args.identity if args.identity is not None else None)
        for sp in st.spectra(): st.fit_whitening(sp)
        st.save(); ui.ok(f"removed {n} model(s) of condition {args.remove_condition!r}"); return 0
    if args.forget_learned:
        drop = [m for m in st.models if m.learned and (args.identity is None or m.identity == args.identity)]
        n = len(drop)
        st.models = [m for m in st.models if m not in drop]
        for sp in st.spectra():
            st.fit_whitening(sp)
        st.save(); ui.ok(f"forgot {n} learned model(s)"); return 0
    if getattr(args, "json", False):
        import json
        from . import attempts, manage
        print(json.dumps({"user": user, "people": manage.people(st, user, attempts.read(cfgmod.LOG_PATH, 500)),
                          "calibrated": (st.calib.get("calibration") or {}).get("date", "")[:10]}))
        return 0
    if not st.models:
        ui.warn(f"no models for {user}"); return 1
    ui.info(f"{user}: {len(st.models)} models · identities: {', '.join(i or user for i in st.identities())}"
            + (f" · conditions: {', '.join(st.conditions())}" if st.conditions() else ""))
    for sp, e in st.whitening.items():
        ui.info(f"whitening[{sp}]: {e['n']} deviations, threshold {e['threshold']:.2f}")
    for m in st.models:
        tags = " ".join(t for t in ((m.identity and f"[{m.identity}]"), (m.condition and f"@{m.condition}"), (m.learned and "learned")) if t)
        ui.info(f"#{m.id:<3} {m.label:<12} {m.spectrum:<4} quality {m.quality:.2f}  samples {len(m.samples):<2} {tags:<22} {m.time}")
    return 0


# -------------------------------------------------------------------- lab ----
def cmd_lab(args) -> int:
    """Live evaluation: compare unlock policies on the same frames, record subjects, summarise."""
    need_root("lab")
    from .engine import lab as labmod
    from .engine.capture import CaptureError
    user = real_user(args)
    ev = labmod.LiveEvaluator(user)
    if not ev.store.models:
        ui.fail(f"no models for {user} — enroll first"); return 1
    try:
        ev.open()
    except CaptureError as e:
        ui.fail(str(e)); return 1
    if getattr(args, "record_frames", None):
        ui.info(f"recording frame sets to {ev.record_frames(args.record_frames)}  (replay: FACEUP_REPLAY=<dir> faceup auth, or faceup auth --replay <dir>)")
    for n in getattr(ev, "notes", []):
        ui.warn(n)
    if not args.text:
        try:
            from . import display as disp
            screen = disp.open_display("faceup lab", 640, 360, kind="lab", evaluator=ev)
            screen.run(ev)
            ev.close()
            print(labmod.format_summary(ev.summary()))
            return 0
        except Exception as e:                       # window unavailable → text mode
            ui.warn(f"no lab window ({e}); text mode")
    ui.banner("faceup lab · text mode", "Ctrl+C for the summary. Policies: " + " · ".join(f"{p} = {labmod.POLICY_TEXT[p]}" for p in labmod.POLICIES))
    if args.subject:
        ui.info(f"recording subject {args.subject!r} → {ev.start_recording(args.subject)}")
    last_print = 0.0
    try:
        while True:
            s = ev.step()
            if s is None or time.monotonic() - last_print < 0.5:
                continue
            last_print = time.monotonic()
            parts = []
            for sp in ("ir", "rgb"):
                if sp in ev.devices:
                    if s.raw.get(sp) is None:
                        parts.append(f"{sp}: {'face' if s.faces.get(sp) else 'no face'}")
                    else:
                        w = f" w {s.wdist[sp]:.2f}/{s.wthr[sp]:.2f}" if s.wdist.get(sp) is not None else ""
                        parts.append(f"{sp}: raw {s.raw[sp]:.2f} ({s.label.get(sp)}){w} {s.light.get(sp, '')}")
            dec = " ".join(f"{p}={'Y' if s.decisions.get(p) else 'n'}" for p in labmod.POLICIES)
            votes = ",".join(f"{v}:{r}" for v, r, _ in s.votes)
            pose = f" yaw {s.yaw:+.0f} pitch {s.pitch:+.0f}" if s.yaw is not None else ""
            print(f"\r\033[K{' | '.join(parts)} | {votes or 'no votes'} | {dec} | {s.reason}{pose} | {ev.fps:.0f} fps", end="", flush=True)
    except KeyboardInterrupt:
        print()
    ev.close()
    print(labmod.format_summary(ev.summary()))
    return 0


# -------------------------------------------------------------------- app ----
def cmd_app(args) -> int:
    """The manager window. Runs as the user; changes go through pkexec (see app.py)."""
    try:
        from . import app
        return app.main([sys.argv[0]] + (["--selftest"] if getattr(args, "selftest", False) else []))
    except ImportError as e:
        ui.fail(f"the app needs GTK 4 + libadwaita (python3-gi, gir1.2-gtk-4.0, gir1.2-adw-1): {e}")
        ui.tip("everything is also available from the command line: faceup setup / enroll / lab / doctor"); return 1


# ---------------------------------------------------------------- keyring ----
def cmd_keyring(args) -> int:
    """status | enable | disable | repair — let a face login unlock GNOME Keyring / KWallet (DESIGN.md §9b)."""
    from .engine import keyring as kr
    user = real_user(args)
    if args.action == "status":
        st = kr.status(user)
        ui.info(f"credential for {user}: {'sealed and stored' if st['enrolled'] else 'unknown without root' if st['enrolled'] is None else 'not enrolled'} ({st['file']})")
        ui.info(f"sealing: {'TPM2' if st['tpm'] else 'host key'}; root disk {'encrypted' if st['encrypted_root'] else 'NOT encrypted'}; systemd-creds {'ok' if st['systemd_creds'] else 'missing'}")
        ps = pammod.status()
        (ui.ok if ps.get("native") else ui.warn)("PAM: " + ("native module — the token reaches the keyring" if ps.get("native") else "pam_exec line — cannot hand the token to the keyring; install gcc, re-run install.sh, then sudo faceup pam enable"))
        dis = kr.disabled_keyring_lines()
        if dis:
            ui.fail(f"{len(dis)} keyring PAM line(s) were commented out by an earlier faceup experiment ({', '.join(sorted({p.name for p, _ in dis}))}) — password logins do not unlock the keyring either")
            ui.tip("sudo faceup keyring repair")
        else:
            ui.ok("pam_gnome_keyring / pam_kwallet lines untouched")
        return 0
    need_root("keyring")
    if args.action == "repair":
        changed = kr.repair_pam_lines(dry_run=args.dry_run)
        (ui.ok if changed else ui.info)(("would restore" if args.dry_run else "restored") + f" keyring lines in: {', '.join(changed)}" if changed else "nothing to repair")
        return 0
    if args.action == "disable":
        ui.ok("credential removed" if kr.remove(user) else "no credential stored")
        return 0
    if args.action == "enable":
        import getpass
        st = kr.status(user)
        if not st["systemd_creds"]:
            ui.fail("systemd-creds is missing (systemd ≥ 250)"); return 1
        ui.banner("keyring credential", f"for {user}: the login password, sealed with {'the TPM2 (Secure Boot state)' if st['tpm'] else 'the host key'}, released only after a face accept")
        if not st["tpm"] and not st["encrypted_root"] and not args.force_host_key:
            ui.fail("no TPM and the root disk is not encrypted: the sealed password would be readable from a stolen disk"); ui.tip("--force-host-key to accept that risk"); return 1
        key = kr.choose_key(prefer_tpm=args.tpm, force_host_key=args.force_host_key)
        if key == "tpm2":
            ui.info("sealing with the TPM2 (bound to Secure Boot state). Note: a firmware TPM adds 2–4 s to every login while it decrypts.")
        else:
            ui.info("sealing with the host key, protected at rest by your encrypted disk — the keyring unlocks instantly (no TPM delay).")
        if not pammod.status().get("native"):
            ui.warn("PAM still uses pam_exec: the credential will be stored but cannot reach the keyring until the native module is enabled (gcc + install.sh + `sudo faceup pam enable`)")
        if getattr(args, "password_stdin", False):
            pw = sys.stdin.readline().rstrip("\r\n")          # the app feeds it from a password dialog
        else:
            pw = getpass.getpass(f"  ? login password for {user} (used to seal the credential; never logged): ")
        if not pw:
            ui.fail("empty password"); return 1
        ok = kr.verify_password(user, pw)
        if ok is False:
            ui.fail("that is not the current login password"); return 1
        if ok is None:
            ui.warn("could not verify the password against /etc/shadow (unsupported hash?)")
            if not ui.confirm("Store it anyway?", default=False):
                return 1
        try:
            path, kind = kr.seal(user, pw, force_host_key=args.force_host_key, prefer_tpm=args.tpm)
        except RuntimeError as e:
            ui.fail(str(e)); return 1
        finally:
            pw = None
        ui.ok(f"sealed with {kind} → {path}")
        dis = kr.disabled_keyring_lines()
        if dis and ui.confirm(f"{len(dis)} keyring PAM line(s) are commented out from an earlier experiment; restore them now?", default=True):
            ui.ok("restored: " + ", ".join(kr.repair_pam_lines()))
        if kind == "tpm2":
            ui.info("bound to PCR 7: a firmware/Secure Boot change invalidates it — re-run `keyring enable` then")
        ui.tip("test: lock the screen (Super+L), unlock by face, then open Passwords & Keys: the login keyring should be unlocked")
        return 0
    return 1


# ------------------------------------------------------------- recogniser ----
def _recogniser_name(cfg: dict) -> str:
    from .engine import recognise
    if cfg["recognition"].get("recogniser", "dlib") == "arcface":
        return f"arcface:{recognise.installed_variant() or 'unknown'}"
    return recognise.DLIB_NAME


def cmd_recogniser(args) -> int:
    """status | install [mbf|r50] | use dlib|arcface — the embedding network (DESIGN.md §6.4a)."""
    from .engine import recognise
    cfg = cfgmod.load()
    if args.action == "status":
        ui.info(f"configured: {cfg['recognition'].get('recogniser', 'dlib')}")
        ui.info(f"dlib: {'available' if probe.python_shadowed()[1] else 'missing'} (128-d, distro package)")
        ui.info(f"arcface: {('installed, variant ' + recognise.installed_variant()) if recognise.available() else 'not installed'} ({recognise.ARCFACE_MODEL})")
        for k, pk in recognise.PACKS.items():
            ui.info(f"  {k}: {pk['name']}, {pk['size_mb']} MB")
        return 0
    need_root("recogniser")
    if args.action == "install":
        ui.warn(recognise.LICENCE_NOTE)
        if not args.yes and not ui.confirm("Download and install it for personal use?", default=True):
            return 1
        try:
            recognise.install(args.variant, say=ui.info)
        except Exception as e:
            ui.fail(f"install failed: {e}"); return 1
        ui.ok(f"ArcFace {args.variant} ready")
        if cfg["recognition"].get("recogniser") != "arcface":
            ui.tip("switch with: sudo faceup recogniser use arcface   (then re-enroll: models from another recogniser cannot be compared)")
        return 0
    if args.action == "use":
        if args.variant == "arcface" and not recognise.available():
            ui.fail("ArcFace is not installed"); ui.tip("sudo faceup recogniser install"); return 1
        cfg["recognition"]["recogniser"] = args.variant
        cfgmod.save(cfg)
        ui.ok(f"recogniser = {args.variant}")
        ui.tip("re-enroll now: sudo faceup enroll --fresh   (the store is per recogniser; the engine refuses a mismatched one)")
        return 0
    return 1


def cmd_antispoof(args) -> int:
    """status | on | off | test DIR — the bundled anti-spoof CNN voter (liveness/antispoof_cnn.py)."""
    from .engine.liveness import antispoof_cnn as asc
    cfg = cfgmod.load()
    ok, got = asc.verify()
    if args.action == "status":
        (ui.ok if ok else ui.fail)(f"model {cfgmod.ANTISPOOF_MODEL}: " + ("sha256 ok" if ok else ("missing" if not got else f"checksum mismatch {got[:12]}…")))
        ui.info(f"voter: {'on' if cfg['liveness']['antispoof_cnn'] else 'off'} · PASS at p_real ≥ {asc.PASS_P} · FAIL only when {asc.FAIL_MIN_FRAMES}+ frames all ≤ {asc.FAIL_P} · else abstain")
        ui.info("runs on lit colour frames only (an IR or grey frame abstains)")
        return 0 if ok else 1
    if args.action == "test":
        import glob
        import cv2
        import numpy as np
        from .engine.detect import Models
        if not args.dir:
            ui.fail("usage: faceup antispoof test DIR   (a folder of colour PNG frames, e.g. a lab recording)"); return 1
        files = sorted(glob.glob(os.path.join(args.dir, "**", "*.png"), recursive=True))
        m = Models("yunet")
        scores = []
        for f in files:
            img = cv2.imread(f)
            if img is None or img.ndim != 3:
                continue
            box = m.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            sc = asc.score(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), box) if box is not None else None
            if sc is not None:
                scores.append(sc)
        if not scores:
            ui.fail(f"no scorable colour face in {len(files)} frames"); return 1
        a = np.array(scores)
        ui.info(f"{len(a)} faces: p_real min {a.min():.3f} · p5 {np.percentile(a, 5):.3f} · median {np.median(a):.3f} · "
                f"PASS {np.mean(a >= asc.PASS_P):.0%} · FAIL-level {np.mean(a <= asc.FAIL_P):.0%}")
        return 0
    need_root("antispoof")
    if args.action == "on" and not ok:
        ui.fail("the bundled model is missing or corrupt — reinstall faceup"); return 1
    cfg["liveness"]["antispoof_cnn"] = args.action == "on"
    cfgmod.save(cfg)
    ui.ok(f"liveness.antispoof_cnn = {cfg['liveness']['antispoof_cnn']}")
    return 0


# ------------------------------------------------------------------ setup ----
def cmd_setup(args) -> int:
    """The whole chain in one go: cameras → emitter → recogniser → register (window) → keyring → PAM → self-check."""
    need_root("setup")
    ui.banner(f"faceup {__version__} · setup", "guided and reversible — config is rewritten, models are yours, PAM is backed up")
    cfg = cfgmod.load()
    cams = cameras.list_cameras()
    if not cams:
        ui.fail("no camera found"); return 1
    steps = 7

    ui.section(f"1/{steps} Cameras")
    for c in cams: ui.info(c.describe())
    ir_cam, rgb_cam = cameras.pick_pair(cams)
    cfg["cameras"]["ir"] = cameras.spec_for(ir_cam, cams) if ir_cam else ""
    cfg["cameras"]["rgb"] = cameras.spec_for(rgb_cam, cams) if rgb_cam else ""
    if not ir_cam:
        ui.info("No IR camera detected. If this laptop has one that streams as YUYV, set cameras.ir to its id and cameras.force_roles = true.")
    profile = "A (IR + RGB)" if ir_cam and rgb_cam else "B (IR only)" if ir_cam else "C (RGB only)"
    ui.ok(f"profile {profile}: ir={cfg['cameras']['ir'] or '-'} rgb={cfg['cameras']['rgb'] or '-'}  (stable USB ids; /dev numbers may change)")
    if ir_cam and rgb_cam and not ui.confirm("Use both cameras? (recommended: IR for recognition, RGB for liveness)", default=True):
        cfg["cameras"]["rgb"] = ""; rgb_cam = None
    if getattr(args, "cameras_only", False):
        cfgmod.save(cfg); ui.ok(f"cameras written to {cfgmod.CONFIG_PATH}"); return 0
    if not ir_cam and rgb_cam:
        ui.warn("RGB only: no infrared, so darkness and photo spoofs are handled by the colour camera alone.")
        if antispoof_available() and cfg["liveness"]["antispoof_cnn"]:
            ui.ok("anti-spoof CNN (bundled) proves liveness for a still face in daylight")
        elif not antispoof_available() and ui.confirm("Without the anti-spoof model a still face cannot prove liveness. Accept recognition alone (convenience-grade)?", default=True):
            cfg["liveness"]["require_liveness_when_available"] = False

    ui.section(f"2/{steps} IR emitter")
    if ir_cam:
        from .engine import emitter as emitmod
        own = emitmod.config_instruction(cfg)
        if own is not None:
            ui.ok(f"LED control known: unit {own.unit} selector {own.selector}")
        else:
            ui.info("Finding the LED control by measurement (any lighting): sit in front of the camera and keep still.")
            if ui.confirm("Run the emitter probe now?", default=True):
                cfgmod.save(cfg)
                setattr(args, "action", "probe")
                cmd_emitter(args); cfg = cfgmod.load()
        if emitmod.config_instruction(cfg) is not None and int(cfg["liveness"].get("shadow_toggle_ms", -1)) < 0:
            if ui.confirm("Benchmark the LED switch (enables the shadow-test liveness voter)?", default=True):
                cfgmod.save(cfg)
                setattr(args, "action", "toggle-bench")
                cmd_emitter(args); cfg = cfgmod.load()
    else:
        ui.info("no IR camera — skipped")

    ui.section(f"3/{steps} Detector and recogniser")
    from .engine.detect import yunet_available
    from .engine import recognise
    if yunet_available():
        cfg["recognition"]["detector"], cfg["recognition"]["max_height"] = "yunet", 320
        ui.ok("YuNet (OpenCV, ~4 ms per frame, robust to turned/tilted faces)")
    else:
        cfg["recognition"]["detector"], cfg["recognition"]["max_height"] = "hog", 320
        ui.warn("YuNet model missing — falling back to dlib HOG (frontal faces only)")
    if recognise.available():
        ui.ok(f"ArcFace {recognise.installed_variant()} installed; recogniser = {cfg['recognition'].get('recogniser', 'dlib')}")
    else:
        ui.info("dlib recognises well; ArcFace (insightface, run by OpenCV) gives tighter identity clusters — the Windows-Hello-class option.")
        ui.info(recognise.LICENCE_NOTE)
        if ui.confirm("Download ArcFace (MobileFaceNet, 16 MB) and use it?", default=False):
            try:
                recognise.install("mbf", say=ui.info)
                cfg["recognition"]["recogniser"] = "arcface"; ui.ok("recogniser = arcface")
            except Exception as e:
                ui.warn(f"install failed ({e}); staying on dlib")
    if cfg["cameras"]["rgb"]:
        if not antispoof_available():
            ui.warn("anti-spoof CNN model missing — reinstall faceup to restore the daylight liveness voter")
        elif cfg["liveness"]["antispoof_cnn"]:
            ui.ok("anti-spoof CNN (MiniFASNetV2, bundled): a second liveness voter on the colour camera")
        elif ui.confirm("The anti-spoof CNN is off. Turn it on (abstains when unsure, never blocks a real face)?", default=True):
            cfg["liveness"]["antispoof_cnn"] = True; ui.ok("liveness.antispoof_cnn = true")
    cfgmod.save(cfg)
    ui.ok(f"config written to {cfgmod.CONFIG_PATH}")

    ui.section(f"4/{steps} Register")
    if ui.confirm("Register your face now? (one guided window: it registers and tunes the gate in a single sweep)", default=True):
        for k, v in (("fresh", False), ("all", False), ("selfcheck", True), ("guided", False), ("profile", False), ("classic", False), ("identity", ""), ("add", "")):
            setattr(args, k, getattr(args, k, v))
        cmd_enroll(args)
        ui.tip("later: `sudo faceup enroll --add glasses` (a short sweep) · another person: `sudo faceup enroll --identity NAME`")

    ui.section(f"5/{steps} Keyring")
    from .engine import keyring as kr
    ks = kr.status(real_user(args))
    if ks["enrolled"]:
        ui.ok("keyring credential already stored")
    elif ui.confirm("Let a face login unlock your keyring? (stores your login password sealed by the TPM / your encrypted disk)", default=True):
        args.action, args.force_host_key, args.tpm = "enable", False, False
        cmd_keyring(args)

    ui.section(f"6/{steps} PAM")
    ps = pammod.status()
    if ps["enabled"] and ps.get("current"):
        ui.ok("already enabled")
    elif ui.confirm(f"Enable faceup for login/sudo/polkit now? (adds one line to {pammod.COMMON_AUTH}, backup kept)", default=True):
        args.action, args.dry_run = "enable", False
        cmd_pam(args)

    ui.section(f"7/{steps} Policy")
    from .engine import policy as polmod
    lvl = polmod.level_for(cfg)
    ui.info(f"policy: {lvl} — {polmod.describe(lvl)}")
    ui.tip("sudo faceup tune policy strict   for a shared machine · sudo faceup tune policy   lists the levels")
    print()
    ui.ok(ui.bold("Setup complete."))
    ui.tip("verify:  sudo -k && sudo true    ·  every attempt:  faceup log    ·  health:  faceup doctor")
    return 0


def antispoof_available() -> bool:
    from .engine.liveness import antispoof_cnn
    return antispoof_cnn.available()


# ---------------------------------------------------------------- emitter ----
def emitter_install() -> None:
    import tempfile
    import urllib.request
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
        paths, _ = cfgmod.camera_paths(cfg)
        rgb = paths.get("rgb") or next((c.path for c in cameras.list_cameras() if not c.is_ir), "")
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
    if args.action == "probe":
        from .engine import emitter as emitmod
        from .engine.capture import Capture, CaptureError
        from .engine.detect import Models
        cfg = cfgmod.load()
        paths, _ = cfgmod.camera_paths(cfg)
        dev = paths.get("ir") or cam.path
        ui.banner("emitter probe", "finds the IR LED control by measurement — sit in front of the camera, any lighting, keep still")
        m = Models("yunet", 320); last = [None]
        try:
            with Capture({"ir": dev}) as capd:
                capd.warm_up(5)
                def read_face():
                    fr = capd.read_one("ir", flush=1)
                    if fr is None:
                        return None
                    box = m.detect(fr.gray) or last[0]
                    if box is None:
                        return None
                    last[0] = box
                    return fr.gray[max(0, box.y):box.y + box.h, max(0, box.x):box.x + box.w]
                if read_face() is None:
                    ui.fail("no face in the IR view — the probe measures the face, sit in front of the camera"); return 1
                best, results = emitmod.probe(dev, read_face, say=ui.info)
        except CaptureError as ex:
            ui.fail(str(ex)); return 1
        if best is None:
            ui.warn(f"no control changed the face brightness by ≥ {emitmod.SHADOW_MIN_DELTA:.0f} ({len(results)} tried)"); return 1
        ui.ok(f"emitter control: unit {best.unit} selector {best.selector}, on {list(best.on)}, off {list(best.off)}")
        if probe.is_root() or "FACEUP_ETC" in os.environ:
            cfg["emitter"] = {"unit": best.unit, "selector": best.selector, "on": ",".join(str(b) for b in best.on), "off": ",".join(str(b) for b in best.off)}
            cfgmod.save(cfg); ui.ok("written to config [emitter]"); ui.tip("now: sudo faceup emitter toggle-bench")
        else:
            ui.tip("run with sudo to save it: sudo faceup emitter probe")
        return 0
    if args.action == "toggle-bench":
        from .engine import emitter as emitmod
        from .engine.capture import Capture, CaptureError
        cfg = cfgmod.load()
        paths, _ = cfgmod.camera_paths(cfg)
        dev = paths.get("ir") or cam.path
        ins = emitmod.instructions_for(cfg)
        if not ins:
            ui.fail(f"no emitter control known: config [emitter] is empty and {emitmod.TOOL_DIR} has no working instruction")
            ui.tip("sudo faceup emitter probe   (finds the LED control by measurement, in any lighting)"); return 1
        ui.info(f"{len(ins)} instruction(s): " + ", ".join(f"unit {i.unit} sel {i.selector} {list(i.on)}" for i in ins))
        try:
            with Capture({"ir": dev}) as capd:
                capd.warm_up(5)
                em = emitmod.Emitter(dev, ins)
                if not em.available:
                    ui.fail(f"cannot drive the emitter: {em.error}"); return 1
                ui.info("switching the LEDs off/on 3 times — keep still and look at the camera (any lighting)")
                res = emitmod.toggle_bench(em, lambda: (lambda f: None if f is None else f.gray)(capd.read_one("ir", flush=1)))
                em.close()
        except CaptureError as ex:
            ui.fail(str(ex)); return 1
        ui.info(f"IR mean: on {res.on_mean:.0f}, off {res.off_mean:.0f}, delta {res.delta:.0f}; toggle latency {res.latency_ms} ms")
        (ui.ok if res.usable else ui.warn)(res.reason)
        if probe.is_root() or "FACEUP_ETC" in os.environ:
            cfg["liveness"]["shadow_toggle_ms"] = res.latency_ms if res.usable else -1
            cfgmod.save(cfg)
            ui.ok(f"liveness.shadow_toggle_ms = {cfg['liveness']['shadow_toggle_ms']} written" + (" — the shadow-test voter is now active (shadow_test = auto)" if res.usable else ""))
        else:
            ui.tip("run with sudo to record the result in the config")
        return 0 if res.usable else 1
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
    s = sub.add_parser("setup", help="the whole chain: cameras → emitter → recogniser → register → keyring → PAM → policy (root)")
    s.add_argument("-U", "--user"); s.add_argument("--cameras-only", action="store_true", help="detect and write the camera pair, nothing else"); s.set_defaults(fn=cmd_setup)
    ap = sub.add_parser("app", help="the GTK application — every step from one window (sudo for setup actions)")
    ap.add_argument("--selftest", action="store_true", help="render every page once and exit (CI smoke test)"); ap.set_defaults(fn=cmd_app)

    e = sub.add_parser("enroll", help="pose-guided, dual-spectrum enrollment (root)")
    e.add_argument("-U", "--user"); e.add_argument("--fresh", action="store_true", help="delete existing models first")
    e.add_argument("--all", action="store_true", help="include optional poses without asking")
    e.add_argument("--selfcheck", action="store_true", help="run the self-check without asking")
    e.add_argument("--guided", action="store_true", help="scripted 8-pose text flow instead of the window")
    e.add_argument("--classic", action="store_true", help="old free-move coverage-map registration + a separate calibration step")
    e.add_argument("--identity", metavar="NAME", help="register another person on this login under NAME (any registered face unlocks)")
    e.add_argument("--add", metavar="CONDITION", help="short sweep that adds a condition to the existing models: glasses, headset, night, …")
    e.add_argument("--profile", action="store_true", help="print per-stage timings once a second (diagnose a laggy preview)")
    e.add_argument("--stream-ui", action="store_true", help=argparse.SUPPRESS)   # used by `faceup app`
    e.set_defaults(fn=cmd_enroll)

    t = sub.add_parser("tune", help="show config, or set section.key VALUE (root)")
    t.add_argument("key", nargs="?"); t.add_argument("value", nargs="?"); t.set_defaults(fn=cmd_tune)

    b = sub.add_parser("bench", help="time HOG vs CNN detectors on a live frame"); b.add_argument("--device"); b.set_defaults(fn=cmd_bench)

    a = sub.add_parser("auth", help="run the authentication engine once (what PAM calls); exit 0 = accepted")
    a.add_argument("-U", "--user"); a.add_argument("-v", "--verbose", action="store_true")
    a.add_argument("--replay", metavar="DIR", help="authenticate against recorded frame sets instead of the cameras"); a.set_defaults(fn=cmd_auth)

    pm = sub.add_parser("pam", help="enable | disable | status the PAM integration (root)")
    pm.add_argument("action", choices=["enable", "disable", "status"]); pm.add_argument("--dry-run", action="store_true"); pm.set_defaults(fn=cmd_pam)

    lb = sub.add_parser("lab", help="live evaluation window: compare unlock policies, record subjects, summarise (root)")
    lb.add_argument("--record-frames", metavar="DIR", help="save every frame set as PNGs for replay (roadmap L7)")
    lb.add_argument("-U", "--user"); lb.add_argument("--text", action="store_true", help="terminal mode instead of the window")
    lb.add_argument("--subject", help="text mode: record samples under this subject name")
    lb.set_defaults(fn=cmd_lab)

    kr = sub.add_parser("keyring", help="status | enable | disable | repair — let a face login unlock GNOME Keyring / KWallet (root except status)")
    kr.add_argument("action", choices=["status", "enable", "disable", "repair"]); kr.add_argument("-U", "--user")
    kr.add_argument("--force-host-key", action="store_true"); kr.add_argument("--tpm", action="store_true", help="force TPM2 sealing (stronger, but adds 2–4 s per login on a firmware TPM)")
    kr.add_argument("--password-stdin", action="store_true", help="read the login password from stdin (used by the app)")
    kr.add_argument("--dry-run", action="store_true"); kr.set_defaults(fn=cmd_keyring)

    asp = sub.add_parser("antispoof", help="status | on | off | test DIR — the bundled anti-spoof CNN liveness voter (root for on/off)")
    asp.add_argument("action", choices=["status", "on", "off", "test"]); asp.add_argument("dir", nargs="?", default="")
    asp.set_defaults(fn=cmd_antispoof)

    rg = sub.add_parser("recogniser", help="status | install [mbf|r50] | use dlib|arcface — the embedding network (root for install/use)")
    rg.add_argument("action", choices=["status", "install", "use"]); rg.add_argument("variant", nargs="?", default="mbf")
    rg.add_argument("-y", "--yes", action="store_true"); rg.set_defaults(fn=cmd_recogniser)

    lg = sub.add_parser("log", help="recent authentication attempts; --summary for rates and latency, --follow to watch")
    lg.add_argument("-n", type=int, default=20); lg.add_argument("--summary", action="store_true"); lg.add_argument("--follow", "-f", action="store_true")
    lg.add_argument("--json", action="store_true"); lg.set_defaults(fn=cmd_log)

    cb = sub.add_parser("calibrate", help="session (default): measure your real distances and set the gate + whitening threshold · refresh: from recent accepts · pose: rebuild the 3D face model (root)")
    cb.add_argument("action", nargs="?", choices=["session", "refresh", "pose"], default="session")
    cb.add_argument("-U", "--user"); cb.add_argument("--seconds", type=float, default=None, help="max seconds per pass")
    cb.add_argument("--text", action="store_true", help="text prompts instead of the window")
    cb.add_argument("--stream-ui", action="store_true", help=argparse.SUPPRESS); cb.set_defaults(fn=cmd_calibrate)

    dm = sub.add_parser("daemon", help="run | status | enable | disable — the resident engine that keeps the models loaded (fast unlock)")
    dm.add_argument("action", choices=["run", "status", "enable", "disable"]); dm.set_defaults(fn=cmd_daemon)

    m = sub.add_parser("models", help="list or remove registered models (root)")
    m.add_argument("-U", "--user"); m.add_argument("--remove", metavar="LABEL")
    m.add_argument("--forget-learned", action="store_true", help="drop models added by online adaptation")
    m.add_argument("--remove-identity", metavar="NAME"); m.add_argument("--remove-condition", metavar="CONDITION")
    m.add_argument("--identity", metavar="NAME", help="with --remove-condition / --forget-learned: only this person's (default: everyone's)")
    m.add_argument("--json", action="store_true", help="people on this login as JSON (used by the app)")
    m.set_defaults(fn=cmd_models)

    em = sub.add_parser("emitter", help="IR emitter: probe (find the LED control by measurement) | toggle-bench (switch speed → shadow test) | check | test | install | configure")
    em.add_argument("action", choices=["check", "test", "install", "configure", "probe", "toggle-bench"]); em.set_defaults(fn=cmd_emitter)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        print(); return 130


if __name__ == "__main__":
    sys.exit(main())
