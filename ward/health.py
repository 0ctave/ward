"""Structured health checks (the data behind `ward doctor` and the app's Status page).

`collect()` returns plain data — sections of items with a level (ok / warn / fail / info), a
sentence, and, when one command fixes it, that command as an argv list (without "ward").
Both the terminal doctor and the GTK app render from this; nothing here prints.
"""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field

from . import cameras, probe
from .engine import config as cfgmod
from .engine import pam as pammod


@dataclass
class Item:
    level: str                      # ok | warn | fail | info
    text: str
    fix: list[str] | None = None    # e.g. ["emitter", "probe"]
    fix_label: str = ""


@dataclass
class Section:
    key: str
    title: str
    items: list[Item] = field(default_factory=list)

    @property
    def worst(self) -> str:
        order = {"fail": 3, "warn": 2, "info": 1, "ok": 0}
        return max((i.level for i in self.items), key=lambda l: order.get(l, 0), default="ok")


def profile_of(cfg: dict) -> str:
    ir, rgb = bool(cfg["cameras"]["ir"]), bool(cfg["cameras"]["rgb"])
    return "A" if ir and rgb else "B" if ir else "C" if rgb else "-"


def collect(user: str | None = None, cfg: dict | None = None, probe_cameras: bool = False) -> dict:
    cfg = cfg or cfgmod.load()
    from .appdata import invoking_user
    user = user or invoking_user()
    out: dict = {"user": user, "profile": profile_of(cfg), "sections": [], "identities": [], "conditions": [],
                 "policy": "", "calibrated": None, "attempts": {}, "root": os.geteuid() == 0}
    S = out["sections"]

    # ---- cameras
    sec = Section("cameras", "Cameras"); S.append(sec)
    try:
        cams, unusable = cameras.list_cameras(include_unusable=True)
    except Exception as e:
        cams, unusable = [], []
        sec.items.append(Item("fail", f"cannot enumerate cameras: {e}"))
    if not cams and not unusable:
        sec.items.append(Item("fail", "no /dev/video* device — no camera driver bound"))
    for c in cams:
        sec.items.append(Item("ok" if c.is_ir else "info", f"{c.path}  {c.describe()}"))
    for u in unusable:
        sec.items.append(Item("warn", f"{u.path} {u.name}: {u.reason}"))
    if cfgmod.CONFIG_PATH.exists():
        paths, notes = cfgmod.camera_paths(cfg)
        sec.items.append(Item("ok", f"profile {out['profile']}: ir = {cfg['cameras']['ir'] or '-'} → {paths.get('ir', '-')}, rgb = {cfg['cameras']['rgb'] or '-'} → {paths.get('rgb', '-')}"))
        for n in notes:
            sec.items.append(Item("warn", n, ["setup"], "Run setup"))
        if any(c.is_ir for c in cams) and not cfg["cameras"]["ir"]:
            sec.items.append(Item("warn", "an IR camera exists but is not configured", ["setup"], "Run setup"))
        if probe_cameras:
            for s in ("ir", "rgb"):
                dev = paths.get(s, "")
                if dev and os.path.exists(dev):
                    L = probe.lighting(dev)
                    face = f", face {L.face_mean:.0f}" if L.face_mean is not None else ", no face in view"
                    lvl = "fail" if L.frames == 0 else ("warn" if (s == "ir" and L.face_mean is not None and L.face_mean < 60) else "ok")
                    sec.items.append(Item(lvl, f"{s} now: frame mean {L.mean:.0f}/255{face}"))
    else:
        sec.items.append(Item("warn", f"no config yet ({cfgmod.CONFIG_PATH})", ["setup"], "Run setup"))

    # ---- emitter
    if cfg["cameras"]["ir"]:
        sec = Section("emitter", "IR emitter"); S.append(sec)
        from .engine import emitter as emitmod
        e = probe.emitter_status()
        own = emitmod.config_instruction(cfg)
        if own is not None:
            sec.items.append(Item("ok", f"LED control known (unit {own.unit} selector {own.selector})"))
            ms = int(cfg["liveness"].get("shadow_toggle_ms", -1))
            if ms < 0:
                sec.items.append(Item("warn", "switch speed not benchmarked — the shadow-test liveness voter is off", ["emitter", "toggle-bench"], "Benchmark"))
            elif ms <= 150:
                sec.items.append(Item("ok", f"switch {ms} ms → shadow test active"))
            else:
                sec.items.append(Item("warn", f"switch {ms} ms — too slow for the shadow test"))
        elif e.get("state") == "working":
            sec.items.append(Item("ok", "linux-enable-ir-emitter has a working instruction"))
            sec.items.append(Item("info", "benchmark the switch to enable the shadow test", ["emitter", "toggle-bench"], "Benchmark"))
        else:
            sec.items.append(Item("warn", "LED control not found yet — the shadow-test voter cannot run", ["emitter", "probe"], "Probe"))
        if e.get("installed") and not e.get("enabled"):
            sec.items.append(Item("info", "linux-enable-ir-emitter service not enabled (fine when the LEDs are on by default)"))

    # ---- engine / recognition / policy
    sec = Section("engine", "Engine"); S.append(sec)
    from .engine import detect as det, recognise, policy as polmod
    from .engine.liveness import antispoof_cnn
    rec, live = cfg["recognition"], cfg["liveness"]
    rk = rec.get("recogniser", "dlib")
    if rk == "arcface":
        sec.items.append(Item("ok" if recognise.available() else "fail",
                              f"recogniser ArcFace {recognise.installed_variant() or ''}".strip() if recognise.available() else "recogniser = arcface but the model is not installed",
                              None if recognise.available() else ["recogniser", "install", "-y"], "" if recognise.available() else "Install"))
    else:
        dlib_ok = det.dlib is not None
        # dlib missing = recogniser=dlib has no embedder (fails at the first embedding): a real problem, not a note
        sec.items.append(Item("info" if dlib_ok else "fail",
                              "recogniser: dlib ResNet (128-d)" if dlib_ok else "recogniser = dlib but python3-dlib is MISSING — face unlock cannot embed; install it or switch to ArcFace",
                              None if dlib_ok else ["recogniser", "install", "-y"], "" if dlib_ok else "Install ArcFace"))
    sec.items.append(Item("info", f"detector {rec['detector']} · gate {rec['certainty']} (rgb {rec['rgb_certainty']}) · fusion {rec['fusion_frames']}"))
    if float(rec["certainty"]) > 0.45:
        sec.items.append(Item("warn", f"gate {rec['certainty']} is loose — calibrate from your real distances", ["calibrate"], "Calibrate"))
    lvl = polmod.level_for(cfg); out["policy"] = lvl
    sec.items.append(Item("info", f"policy {lvl} — {polmod.describe(lvl)}"))
    has_ir, has_rgb = bool(cfg["cameras"]["ir"]), bool(cfg["cameras"]["rgb"])
    from .engine.auth import Engine
    still = []
    if has_ir and has_rgb and live["cross_spectrum"]:
        still.append("cross-spectrum")
    if has_ir and Engine._shadow_enabled(live):
        still.append("shadow test")
    if has_rgb and live["antispoof_cnn"] and antispoof_cnn.available():
        still.append("anti-spoof CNN")
    if has_rgb:
        ok_model, _ = antispoof_cnn.verify()
        if not ok_model:
            sec.items.append(Item("fail", f"anti-spoof CNN model missing or corrupt ({cfgmod.ANTISPOOF_MODEL.name}) — reinstall ward", ["antispoof", "status"], "Check"))
        elif not live["antispoof_cnn"]:
            sec.items.append(Item("warn", "anti-spoof CNN (daylight liveness, bundled) is switched off", ["antispoof", "on"], "Turn on"))
    if still:
        sec.items.append(Item("ok", "liveness for a still face: " + ", ".join(still)))
    elif has_ir:
        sec.items.append(Item("warn", "no liveness voter can PASS for a still face: unlock asks you to move your head", ["emitter", "probe"], "Probe emitter"))
    elif not live.get("require_liveness_when_available", True):
        sec.items.append(Item("info", "RGB only: recognition alone unlocks (convenience-grade)"))
    else:
        sec.items.append(Item("warn", "RGB only: a still face cannot prove liveness — nod, or accept recognition alone",
                              ["tune", "liveness.require_liveness_when_available", "false"], "Recognition alone"))

    # ---- faces (store)
    sec = Section("faces", "Faces"); S.append(sec)
    from .engine import store as storemod
    try:
        st = storemod.load(user, recognise.DLIB_NAME if rk != "arcface" else f"arcface:{recognise.installed_variant() or 'unknown'}")
        if not st.models:
            sec.items.append(Item("fail", f"no face enrolled for {user}", ["enroll"], "Enrol"))
        else:
            out["identities"] = [i or user for i in st.identities()]
            out["conditions"] = st.conditions()
            for sp in sorted(st.spectra()):
                n = len(st.by_spectrum(sp))
                sec.items.append(Item("ok" if n >= 4 else "warn", f"{sp}: {n} models" + (f" ({', '.join(out['identities'])})" if len(out['identities']) > 1 else ""),
                                      None if n >= 4 else ["enroll"], "" if n >= 4 else "Enrol more poses"))
            c = st.calib.get("calibration")
            if c:
                out["calibrated"] = c.get("date", "")[:10]
                sec.items.append(Item("ok", f"calibrated {out['calibrated']} from " + ", ".join(f"{k} {v}" for k, v in c.get("n", {}).items()) + " measurements"))
            else:
                sec.items.append(Item("warn", "thresholds never calibrated on this machine", ["calibrate"], "Calibrate"))
            learned = [m for m in st.models if m.learned]; enrolled = [m for m in st.models if not m.learned]
            if learned and len(learned) >= len(enrolled):
                sec.items.append(Item("warn", f"{len(learned)} learned models vs {len(enrolled)} enrolled — churn", ["models", "--forget-learned"], "Forget learned"))
            if out["conditions"]:
                sec.items.append(Item("info", "conditions: " + ", ".join(out["conditions"])))
    except PermissionError:
        sec.items.append(Item("info", "model store needs root to read"))
    except storemod.StoreError as e:
        sec.items.append(Item("fail", str(e), ["enroll", "--fresh"], "Re-enrol"))

    # ---- keyring + PAM
    sec = Section("access", "Login, sudo, keyring"); S.append(sec)
    from .engine import keyring as kr
    ps = pammod.status()
    if ps["enabled"] and not ps.get("current"):
        sec.items.append(Item("warn", "PAM line is from an older version", ["pam", "enable"], "Upgrade"))
    elif ps["enabled"]:
        sec.items.append(Item("ok", "face unlock enabled for login, sudo and polkit" + (" (native module)" if ps.get("native") else " (pam_exec)")))
        # Pre-flight the user resolution: sudo/login authenticate the human (the login uid), and an
        # unlock only works if THAT user is enrolled. Catches the "sudo authenticated root → no-models"
        # class of bug before the user hits it at a password prompt.
        from .client import _human_user
        human = _human_user()
        if human and human != user:
            sec.items.append(Item("info", f"sudo and login will authenticate {human} (the logged-in user)"))
        try:
            hstore = storemod.load(human, recognise.DLIB_NAME if rk != "arcface" else f"arcface:{recognise.installed_variant() or 'unknown'}")
            if not hstore.models:
                sec.items.append(Item("warn", f"PAM is on but {human} has no enrolled face — sudo/login fall straight to the password",
                                      ["enroll", "--fresh"], "Enrol"))
            else:
                sec.items.append(Item("ok", f"sudo and login unlock as {human} ({len(hstore.models)} models enrolled)"))
        except PermissionError:
            pass
        except storemod.StoreError:
            pass
    else:
        sec.items.append(Item("fail", "face unlock NOT enabled in PAM", ["pam", "enable"], "Enable"))
    if ps.get("howdy"):
        sec.items.append(Item("info", "pam_howdy is also active (two face steps)", ["pam", "enable"], "Retire Howdy"))
    ks = kr.status(user)
    dis = kr.disabled_keyring_lines()
    if dis:
        sec.items.append(Item("fail", "keyring PAM lines were commented out by an earlier experiment", ["keyring", "repair"], "Repair"))
    if ps.get("native") and ks["enrolled"] and ks.get("kind") == "tpm2" and kr.root_is_encrypted():
        sec.items.append(Item("warn", "keyring password sealed with the TPM: 2–4 s per unlock (and it fails after firmware updates) — the encrypted disk's host key is instant",
                              ["keyring", "enable"], "Re-seal"))
    elif ps.get("native") and ks["enrolled"]:
        sec.items.append(Item("ok", "a face login unlocks the keyring"))
    elif ks["enrolled"] is None:
        sec.items.append(Item("info", "keyring credential: state needs root"))
    elif not ps.get("native"):
        sec.items.append(Item("info", "keyring token needs the native module (gcc + install.sh)"))
    elif cfg["behaviour"].get("skip_login"):
        sec.items.append(Item("info", "first login uses the password (skip_login)"))
    else:
        sec.items.append(Item("warn", "after a face login the keyring stays locked until first use", ["keyring", "enable"], "Unlock keyring by face"))

    # ---- resident engine (speed)
    try:
        from .cli import daemon_state
        ds = daemon_state()
        if ds["active"]:
            sec.items.append(Item("ok", "resident engine running: unlocks skip start-up and model loading"))
        elif ds["installed"]:
            sec.items.append(Item("warn", "resident engine stopped: each unlock loads everything first (~1 s slower)", ["daemon", "enable"], "Start"))
    except Exception:
        pass

    # ---- attempts
    sec = Section("attempts", "Attempts"); S.append(sec)
    from . import attempts
    rows = attempts.read(cfgmod.LOG_PATH, 20)
    if rows:
        sm = attempts.summary(rows); out["attempts"] = sm
        sec.items.append(Item("ok" if sm["accept_rate"] >= 0.9 else "warn",
                              f"last {sm['attempts']}: {sm['accepted']} accepted" + (", " + ", ".join(f"{k} {v}" for k, v in sm["reasons"].items()) if sm["reasons"] else "")))
        if sm["median_accept_ms"] is not None:
            sec.items.append(Item("ok" if sm["median_accept_ms"] <= 800 else "info", f"median time to accept {sm['median_accept_ms']} ms"))
    else:
        sec.items.append(Item("info", "no attempts logged yet"))

    # ---- updates (cache only; `ward update` does the network check)
    from . import __version__, update as upd
    rec = upd.cached()
    sec = Section("updates", "Updates"); S.append(sec)
    if rec and upd.is_newer(rec["latest"]):
        sec.items.append(Item("warn", f"a newer Ward is available: {rec['latest']} (you have {__version__})", ["update"], "Check"))
    else:
        sec.items.append(Item("info", f"Ward {__version__} · run `ward update` to check for a newer release"))

    # ---- compatibility
    sec = Section("compat", "Compatibility"); S.append(sec)
    sec.items.append(Item("info", f"python {sys.version.split()[0]} · opencv {__import__('cv2').__version__} (YuNet {'yes' if det.yunet_available() else 'no'}) · dlib {'yes' if det.dlib is not None else 'no'}"))
    try:
        import gi; gi.require_version("Adw", "1"); from gi.repository import Adw  # noqa: F401
        import cairo  # noqa: F401  (appui draws with pycairo)
        sec.items.append(Item("ok", "GTK 4 + libadwaita available"))
    except Exception:
        sec.items.append(Item("warn", "GTK 4 + libadwaita missing: windows fall back to text flows"))
    sec.items.append(Item("info", f"gcc {('yes' if os.path.exists('/usr/include/errno.h') else 'yes, no libc headers') if shutil.which('gcc') else 'no'} · TPM {'yes' if kr.has_tpm() else 'no'} · encrypted root {'yes' if kr.root_is_encrypted() else 'no'}"))
    return out


def problems(report: dict) -> int:
    return sum(1 for s in report["sections"] for i in s.items if i.level in ("warn", "fail"))
