"""faceup command line: doctor · setup · enroll · tune · bench · emitter."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.request

from . import __version__, cameras, howdy, probe
from . import ui

EMITTER_REPO = "EmixamPP/linux-enable-ir-emitter"

# Poses that cover how a face actually presents itself at a laptop. Order matters:
# the first ones are the everyday cases, the last ones are optional refinements.
POSES = [
    ("straight",   "Look straight at the camera, neutral face."),
    ("keyboard",   "Look DOWN at the keyboard — this is where your face is when a sudo prompt appears."),
    ("tilt-left",  "Tilt your head ~15° to the left (ear toward shoulder)."),
    ("tilt-right", "Tilt your head ~15° to the right."),
    ("turn-left",  "Turn your head ~20° to the left, eyes on the screen."),
    ("turn-right", "Turn your head ~20° to the right."),
    ("lean-back",  "Sit back further from the screen, relaxed posture."),
    ("close",      "Lean in close, like reading small text."),
]
OPTIONAL_POSES = [
    ("glasses",    "Wearing your glasses (or without them, if you enrolled with them on)."),
    ("headset",    "Wearing headphones / a headset."),
    ("dark",       "Lights OFF — verifies the IR emitter is doing the work."),
]


def need_root(what: str) -> None:
    if not probe.is_root():
        ui.fail(f"{what} needs root (Howdy's config and models are root-owned).")
        ui.tip(f"run:  sudo faceup {' '.join(sys.argv[1:])}")
        sys.exit(2)


# ----------------------------------------------------------------- doctor ----
def cmd_doctor(args) -> int:
    ui.banner(f"faceup {__version__} · doctor", "Everything that decides whether face unlock works on this machine")
    problems = 0

    ui.section("Howdy")
    if howdy.installed():
        ui.ok("Howdy is installed; config at /etc/howdy/config.ini")
    else:
        ui.fail("Howdy not found (need the `howdy` command and /etc/howdy/config.ini)")
        ui.tip("Ubuntu 24.04: sudo add-apt-repository ppa:ubuntuhandbook1/howdy && sudo apt install howdy")
        problems += 1
    if howdy.pam_enabled():
        ui.ok("PAM is wired (pam_howdy in common-auth): login, sudo and polkit can use it")
    else:
        ui.fail("pam_howdy is not referenced by /etc/pam.d — authentication will never call Howdy")
        ui.tip("sudo pam-auth-update   (enable Howdy)")
        problems += 1
    py, dlib_ok = probe.python_shadowed()
    if dlib_ok:
        ui.ok(f"`python3` on PATH ({py}) can import dlib + cv2")
    else:
        ui.fail(f"`python3` on PATH is {py} and cannot import dlib/cv2")
        ui.tip("Howdy's PAM module runs plain `python3`. A pyenv/mise/conda Python shadowing the system one breaks it.")
        ui.tip("fix: remove the global shadow (e.g. `mise unuse -g python`) or ensure that Python has dlib+opencv")
        problems += 1

    ui.section("Cameras")
    cams = cameras.list_cameras()
    if not cams:
        ui.fail("no capture-capable /dev/video* device")
        problems += 1
    for c in cams:
        (ui.ok if c.is_ir else ui.info)(c.describe())
    ir = [c for c in cams if c.is_ir]
    configured = howdy.get("device_path")
    if ir and configured not in [c.path for c in ir]:
        ui.warn(f"An IR camera exists ({ir[0].path}) but Howdy uses device_path = {configured}")
        ui.tip("IR works in the dark and ignores phone-screen photos. `sudo faceup setup` switches it.")
        problems += 1
    elif configured:
        ui.ok(f"Howdy uses device_path = {configured}")
    if not ir:
        ui.info("No IR camera: recognition will depend on room light and can be fooled by a photo.")

    if configured and configured != "none" and os.path.exists(configured):
        ui.section("Lighting right now (" + configured + ")")
        L = probe.lighting(configured)
        line = f"mean {L.mean:.0f}/255, p5 {L.p5:.0f}, p95 {L.p95:.0f} → {L.verdict}"
        if L.verdict == "good":
            ui.ok(line)
        elif L.frames == 0:
            ui.fail("camera gave no frames (busy, or permission denied)")
            problems += 1
        else:
            ui.warn(line)
            if L.verdict in ("dark", "dim"):
                ui.tip("Howdy rejects frames darker than dark_threshold "
                       f"(now {howdy.get('dark_threshold')}). Raise it, or fix the IR emitter.")
            problems += 1

    if any(c.is_ir for c in cams):
        ui.section("IR emitter (linux-enable-ir-emitter)")
        e = probe.emitter_status()
        if not e["installed"]:
            ui.warn("not installed — if the IR preview is black in a dark room, the emitter isn't firing")
            ui.tip("sudo faceup emitter install")
        else:
            (ui.ok if e["enabled"] else ui.warn)("service " + ("enabled (re-applies at boot and resume)" if e["enabled"] else "NOT enabled"))
            if e["configured"] is False:
                ui.fail("installed but NO device configured — the tool has nothing to switch on")
                ui.tip("sudo faceup emitter configure")
                problems += 1
            elif e["configured"] is None:
                ui.info("configuration state unknown (run as root to check)")
            else:
                ui.ok("a device is configured")

    ui.section("Recognition settings")
    for key, hint in (("certainty", "match tolerance; higher = more forgiving (3.5 default, 4.0–4.5 with several poses)"),
                      ("timeout", "seconds to keep trying before password fallback"),
                      ("use_cnn", "CNN detector finds tilted/turned faces; costs CPU"),
                      ("max_height", "frame height Howdy analyses"),
                      ("dark_threshold", "frames darker than this are rejected")):
        ui.info(f"{key} = {howdy.get(key)}   {ui.dim(hint)}")
    user = howdy.real_user()
    m = howdy.models(user)
    if probe.is_root() or m:
        if len(m) >= 4:
            ui.ok(f"{len(m)} models enrolled for {user}: {', '.join(m)}")
        elif m:
            ui.warn(f"only {len(m)} model(s) for {user}: {', '.join(m)} — one pose ≈ one working angle")
            ui.tip("sudo faceup enroll")
            problems += 1
        else:
            ui.fail(f"no models for {user}")
            ui.tip("sudo faceup enroll")
            problems += 1
    else:
        ui.info("model list needs root (sudo faceup doctor)")

    print()
    if problems == 0:
        ui.ok(ui.bold("No problems found."))
    else:
        ui.warn(ui.bold(f"{problems} thing(s) to fix — `sudo faceup setup` walks through all of them."))
    return 1 if problems else 0


# ------------------------------------------------------------------ bench ----
def cmd_bench(args) -> int:
    dev = args.device or howdy.get("device_path") or (cameras.pick_default(cameras.list_cameras()) or cameras.Camera("", "")).path
    if not dev or not os.path.exists(dev):
        ui.fail("no camera to benchmark"); return 1
    ui.banner("detector benchmark", f"on a live frame from {dev}")
    b = probe.benchmark(dev)
    if b is None:
        ui.fail("needs python3-opencv and python3-dlib"); return 1
    ui.info(f"HOG (Howdy default, frontal-only):  {b.hog_ms:6.0f} ms   faces found: {b.faces_hog}")
    for h, ms in b.cnn_ms.items():
        ui.info(f"CNN at max_height={h}:              {ms:6.0f} ms   faces found: {b.faces_cnn}")
    rec = recommend_cnn(b)
    print()
    if rec:
        ui.ok(f"CNN is affordable at max_height={rec}: `sudo faceup tune --cnn {rec}` to detect tilted/turned faces")
    else:
        ui.warn("CNN is too slow on this CPU for a snappy unlock; stay on HOG and enroll more poses instead")
    return 0


def recommend_cnn(b: probe.Bench) -> int | None:
    """Largest frame height where a CNN pass stays under ~0.9 s (≥ 4 attempts in a 4 s timeout)."""
    for h in sorted(b.cnn_ms, reverse=True):
        if b.cnn_ms[h] < 900 and h >= 180:
            return h
    return None


# ------------------------------------------------------------------- tune ----
def cmd_tune(args) -> int:
    need_root("tune")
    values: dict[str, str] = {}
    if args.certainty is not None: values["certainty"] = str(args.certainty)
    if args.timeout is not None:   values["timeout"] = str(args.timeout)
    if args.dark is not None:      values["dark_threshold"] = str(args.dark)
    if args.device:                values["device_path"] = args.device
    if args.cnn is not None:
        values["use_cnn"] = "true"; values["max_height"] = str(args.cnn)
    if args.hog:
        values["use_cnn"] = "false"; values["max_height"] = "320"
    if not values:
        ui.warn("nothing to change — see `faceup tune --help`"); return 1
    changed = howdy.set_many(values)
    for k in values:
        (ui.ok if k in changed else ui.info)(f"{k} = {values[k]}" + ("" if k in changed else "  (unchanged)"))
    return 0


# ----------------------------------------------------------------- enroll ----
def cmd_enroll(args) -> int:
    need_root("enroll")
    user = args.user or howdy.real_user()
    ui.banner("pose-guided enrollment", f"for {user} — Howdy matches against EVERY model, so one model per pose")
    existing = howdy.models(user)
    if existing:
        ui.info(f"already enrolled: {', '.join(existing)}")
        if args.fresh or ui.confirm("Delete them and start clean? (recommended after switching camera)", default=False):
            howdy.run("clear", user=user, yes=True)
            existing = []
    poses = list(POSES)
    if args.all or ui.confirm("Also add the optional poses (glasses, headset, lights off)?", default=False):
        poses += OPTIONAL_POSES
    ui.info("Sit as you normally do. Each capture takes a second; keep still until it says done.")
    done = 0
    for label, instruction in poses:
        if label in existing:
            ui.info(f"{label}: already enrolled, skipping"); continue
        print()
        ui.info(f"{ui.bold(label)} — {instruction}")
        if ui.ask("Press Enter to capture, or s to skip", "").lower().startswith("s"):
            ui.info(f"{label}: skipped"); continue
        rc = howdy.run("add", label, user=user, yes=True)
        if rc == 0:
            ui.ok(f"{label} saved"); done += 1
        else:
            ui.warn(f"{label} failed (no face found?) — try again with better light or a smaller angle")
            if ui.confirm("Retry this pose?", default=True):
                if howdy.run("add", label, user=user, yes=True) == 0:
                    ui.ok(f"{label} saved"); done += 1
    print()
    ui.ok(f"{done} new model(s). Total now: {len(howdy.models(user))}")
    ui.tip("test it:  sudo -k && sudo true   (should unlock by face)")
    return 0


# ------------------------------------------------------------------ setup ----
def cmd_setup(args) -> int:
    need_root("setup")
    ui.banner(f"faceup {__version__} · setup", "guided, reversible (config is backed up before every change)")
    if not howdy.installed():
        ui.fail("Install Howdy first, then re-run. See `faceup doctor`."); return 1
    user = howdy.real_user()

    # 1. camera
    ui.section("1/5 Camera")
    cams = cameras.list_cameras()
    if not cams:
        ui.fail("no camera found"); return 1
    for c in cams: ui.info(c.describe())
    default = cameras.pick_default(cams)
    idx = ui.choose("Which camera should Howdy use? (IR is the right choice when present)",
                    [c.describe() for c in cams], cams.index(default))
    cam = cams[idx]
    values = {"device_path": cam.path, "dark_threshold": "90" if cam.is_ir else "60"}

    # 2. emitter
    if cam.is_ir:
        ui.section("2/5 IR emitter")
        e = probe.emitter_status()
        L = probe.lighting(cam.path)
        ui.info(f"IR frame brightness now: {L.mean:.0f}/255 ({L.verdict})")
        if e["configured"]:
            ui.ok("emitter tool installed and configured")
        else:
            ui.warn("The emitter is what lights your face in the dark. Without it IR only works in bright rooms.")
            ui.tip("Definitive test: lights off → `sudo howdy test` → is your face visible?")
            if ui.confirm("Install/configure linux-enable-ir-emitter now? (interactive, ~2–5 min, well-lit room)", default=not e["installed"]):
                emitter_install(cam) if not e["installed"] else None
                emitter_configure(cam)
    else:
        ui.section("2/5 IR emitter"); ui.info("RGB camera selected — no emitter to configure")

    # 3. detector
    ui.section("3/5 Detector")
    ui.info("Measuring how fast Howdy's detectors run on this CPU…")
    b = probe.benchmark(cam.path)
    if b:
        ui.info(f"HOG: {b.hog_ms:.0f} ms · CNN: " + ", ".join(f"{h}px={ms:.0f} ms" for h, ms in b.cnn_ms.items()))
        rec = recommend_cnn(b)
        if rec and ui.confirm(f"Use the CNN detector at max_height={rec}? (finds tilted/turned faces; ~{b.cnn_ms[rec]/1000:.1f} s per attempt)", default=rec >= 240):
            values.update(use_cnn="true", max_height=str(rec))
        else:
            values.update(use_cnn="false", max_height="320")
    else:
        ui.warn("benchmark unavailable (missing dlib/opencv); leaving detector settings alone")

    # 4. tolerance
    ui.section("4/5 Match tolerance")
    ui.info("certainty: higher = more forgiving. 3.5 is Howdy's default; with several enrolled poses 4.0–4.5 is a good balance.")
    values["certainty"] = ui.ask("certainty", "4.0")
    values["timeout"] = ui.ask("timeout (seconds before password fallback)", "5")
    changed = howdy.set_many(values)
    for k, v in values.items():
        (ui.ok if k in changed else ui.info)(f"{k} = {v}")

    # 5. enroll
    ui.section("5/5 Enroll poses")
    if ui.confirm("Run pose-guided enrollment now?", default=True):
        args.user, args.fresh, args.all = user, False, False
        cmd_enroll(args)

    print()
    ui.ok(ui.bold("Setup complete."))
    ui.tip("verify:  sudo howdy test   then   sudo -k && sudo true")
    ui.tip("re-check anytime:  faceup doctor")
    return 0


# ---------------------------------------------------------------- emitter ----
def emitter_install(cam) -> None:
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
    ui.info("Sit close, room well lit, no other app using the camera. Answer Y in the video window when it flashes.")
    ui.info("Be patient — do not kill it.")
    rc = subprocess.call(["linux-enable-ir-emitter", "--device", cam.path, "--width", str(w), "--height", str(h), "configure"])
    (ui.ok if rc == 0 else ui.warn)("configure finished" if rc == 0 else "configure did not succeed — see the tool's README for --manual / --limit -1")


def cmd_emitter(args) -> int:
    need_root("emitter")
    cams = cameras.list_cameras()
    ir = [c for c in cams if c.is_ir]
    if not ir:
        ui.fail("no IR camera on this machine"); return 1
    cam = ir[0]
    if args.action == "install":
        if probe.emitter_status()["installed"]:
            ui.info("already installed")
        else:
            emitter_install(cam)
        emitter_configure(cam)
    elif args.action == "configure":
        emitter_configure(cam)
    elif args.action == "test":
        L = probe.lighting(cam.path)
        ui.info(f"{cam.path}: mean {L.mean:.0f}/255 → {L.verdict}")
        ui.tip("Run this with the lights OFF: 'dark' means the emitter is not firing.")
    return 0


# ------------------------------------------------------------------- main ----
def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="faceup", description="Make face unlock on Linux (Howdy) actually work.")
    p.add_argument("--version", action="version", version=f"faceup {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="diagnose camera, emitter, PAM, python, settings and models").set_defaults(fn=cmd_doctor)
    sub.add_parser("setup", help="guided setup: camera → emitter → detector → tolerance → enrollment (root)").set_defaults(fn=cmd_setup)

    e = sub.add_parser("enroll", help="pose-guided multi-model enrollment (root)")
    e.add_argument("-U", "--user"); e.add_argument("--fresh", action="store_true", help="delete existing models first")
    e.add_argument("--all", action="store_true", help="include optional poses without asking")
    e.set_defaults(fn=cmd_enroll)

    t = sub.add_parser("tune", help="change recognition settings (root)")
    t.add_argument("--certainty", type=float); t.add_argument("--timeout", type=int)
    t.add_argument("--dark", type=int, help="dark_threshold"); t.add_argument("--device")
    t.add_argument("--cnn", type=int, metavar="MAX_HEIGHT", help="enable CNN detector at this frame height")
    t.add_argument("--hog", action="store_true", help="back to the default HOG detector")
    t.set_defaults(fn=cmd_tune)

    b = sub.add_parser("bench", help="time HOG vs CNN detectors on a live frame")
    b.add_argument("--device"); b.set_defaults(fn=cmd_bench)

    em = sub.add_parser("emitter", help="IR emitter: install | configure | test (root)")
    em.add_argument("action", choices=["install", "configure", "test"]); em.set_defaults(fn=cmd_emitter)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        print(); return 130


if __name__ == "__main__":
    sys.exit(main())
