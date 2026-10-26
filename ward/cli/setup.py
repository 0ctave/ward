"""ward setup · emitter — the guided chain and IR emitter tooling."""
from __future__ import annotations

import json
import os
import subprocess

from .. import __version__, cameras, probe, ui
from ..engine import config as cfgmod
from ..engine import pam as pammod
from .auth import cmd_pam
from .common import EMITTER_REPO, need_root, real_user
from .enroll import cmd_enroll
from .manage import cmd_keyring

# ------------------------------------------------------------------ setup ----
def cmd_setup(args) -> int:
    """The whole chain in one go: cameras → emitter → recogniser → register (window) → keyring → PAM → self-check."""
    need_root("setup")
    ui.banner(f"ward {__version__} · setup", "guided and reversible — config is rewritten, models are yours, PAM is backed up")
    from .enroll import _display_available
    if _display_available():
        ui.tip("Prefer a graphical setup? Open the Ward app (or run `ward`) → it guides every step in one window.")
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
        from ..engine import emitter as emitmod
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
    from ..engine.detect import yunet_available
    from ..engine import recognise
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
            ui.warn("anti-spoof CNN model missing — reinstall ward to restore the daylight liveness voter")
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
        ui.tip("later: `sudo ward enroll --add glasses` (a short sweep) · another person: `sudo ward enroll --identity NAME`")

    ui.section(f"5/{steps} Keyring")
    from ..engine import keyring as kr
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
    elif ui.confirm(f"Enable ward for login/sudo/polkit now? (adds one line to {pammod.COMMON_AUTH}, backup kept)", default=True):
        args.action, args.dry_run = "enable", False
        cmd_pam(args)

    ui.section(f"7/{steps} Policy")
    from ..engine import policy as polmod
    lvl = polmod.level_for(cfg)
    ui.info(f"policy: {lvl} — {polmod.describe(lvl)}")
    ui.tip("sudo ward tune policy strict   for a shared machine · sudo ward tune policy   lists the levels")
    print()
    ui.ok(ui.bold("Setup complete."))
    ui.tip("verify:  sudo -k && sudo true    ·  every attempt:  ward log    ·  health:  ward doctor")
    return 0


def antispoof_available() -> bool:
    from ..engine.liveness import antispoof_cnn
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
        from .. import gui
        from ..engine.capture import Capture, CaptureError
        from ..engine.detect import Models
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
            ui.tip("sudo ward emitter configure")
        return 0 if state == "ok" else 1
    if args.action == "probe":
        from ..engine import emitter as emitmod
        from ..engine.capture import Capture, CaptureError
        from ..engine.detect import Models
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
        if probe.is_root() or "WARD_ETC" in os.environ:
            cfg["emitter"] = {"unit": best.unit, "selector": best.selector, "on": ",".join(str(b) for b in best.on), "off": ",".join(str(b) for b in best.off)}
            cfgmod.save(cfg); ui.ok("written to config [emitter]"); ui.tip("now: sudo ward emitter toggle-bench")
        else:
            ui.tip("run with sudo to save it: sudo ward emitter probe")
        return 0
    if args.action == "toggle-bench":
        from ..engine import emitter as emitmod
        from ..engine.capture import Capture, CaptureError
        cfg = cfgmod.load()
        paths, _ = cfgmod.camera_paths(cfg)
        dev = paths.get("ir") or cam.path
        ins = emitmod.instructions_for(cfg)
        if not ins:
            ui.fail(f"no emitter control known: config [emitter] is empty and {emitmod.TOOL_DIR} has no working instruction")
            ui.tip("sudo ward emitter probe   (finds the LED control by measurement, in any lighting)"); return 1
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
        if probe.is_root() or "WARD_ETC" in os.environ:
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
