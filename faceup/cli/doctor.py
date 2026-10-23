"""faceup doctor · bench · tune — diagnosis and configuration."""
from __future__ import annotations

import os
import subprocess

from .. import __version__, cameras, probe, ui
from ..engine import config as cfgmod
from ..engine import pam as pammod
from ..engine import store as storemod
from .common import need_root, real_user

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
        from ..engine.emitter import config_instruction as emitmod_config_instruction
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
    from .. import hw
    hp = hw.profile()
    ui.info(f"{hp.cpu} · {hp.cores} threads · AVX2 {'yes' if hp.avx2 else 'no'} · OpenCV {hp.opencv} · OpenCL {('yes: ' + hp.opencl_device) if hp.opencl else 'no'}")
    for c in cams:
        try:
            import cv2
            capx = cv2.VideoCapture(int(c.path.replace('/dev/video', '')), cv2.CAP_V4L2)
            from ..engine.brightness import Exposure
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
    elif os.path.realpath(py) != os.path.realpath("/usr/bin/python3"):
        ui.fail(f"`python3` on PATH is {py} and cannot import dlib/cv2 (a pyenv/mise/conda Python shadows the system one)"); problems += 1
    else:   # the system Python itself lacks dlib (not packaged on most distros): optional now
        ui.warn(f"{py} cannot import dlib: the dlib recogniser is unavailable (ArcFace + YuNet still work)")
        ui.tip(__import__("faceup.distro", fromlist=["hint"]).hint("dlib"))
    from ..engine.liveness import antispoof_cnn
    _as_ok, _ = antispoof_cnn.verify()
    if not _as_ok:
        ui.fail(f"anti-spoof CNN model missing or corrupt ({cfgmod.ANTISPOOF_MODEL}) — reinstall faceup (sudo ./install.sh)"); problems += 1
    elif not cfg["liveness"]["antispoof_cnn"]:
        ui.warn("anti-spoof CNN (bundled MiniFASNetV2, daylight liveness) is switched off"); ui.tip("sudo faceup antispoof on")
    else:
        ui.ok(f"anti-spoof CNN: MiniFASNetV2 bundled, checksum ok ({cfgmod.ANTISPOOF_MODEL.name})")
    rec, live = cfg["recognition"], cfg["liveness"]
    from ..engine import recognise
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
    from ..engine import policy as polmod
    from ..engine.auth import Engine as _Eng
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
            from ..engine import calibration as cal
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
    from ..engine import keyring as _kr, detect as _det
    ui.info(f"python {_sys.version.split()[0]} · opencv {__import__('cv2').__version__}"
            f" (YuNet {'yes' if _det.yunet_available() else 'no → HOG'}) · dlib {'yes' if _det.dlib is not None else 'no (ArcFace + YuNet path only)'}")
    try:
        import gi; gi.require_version("Adw", "1"); from gi.repository import Adw  # noqa
        import cairo  # noqa: F401  (appui draws with pycairo: python3-gi-cairo / python3-gobject-cairo)
        ui.info("GTK 4 + libadwaita: yes (windows) ")
    except Exception:
        ui.warn("GTK 4 + libadwaita missing: enroll/calibrate fall back to text (--guided / --text)"); ui.tip(__import__("faceup.distro", fromlist=["hint"]).hint("gtk"))
    ui.info(f"profile: {'A (IR + RGB)' if cfg['cameras']['ir'] and cfg['cameras']['rgb'] else 'B (IR only)' if cfg['cameras']['ir'] else 'C (RGB only)'}"
            f" · gcc {('yes' if os.path.exists('/usr/include/errno.h') else 'yes but no libc headers (' + __import__('faceup.distro', fromlist=['hint']).hint('gcc') + ')') if _sh.which('gcc') else 'no (pam_exec fallback, no keyring token)'} · TPM {'yes' if _kr.has_tpm() else 'no'} · encrypted root {'yes' if _kr.root_is_encrypted() else 'no'}")

    ui.section("Attempts")
    from .. import attempts
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
    from ..engine import keyring as kr
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
    from ..engine import policy as polmod
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
