"""ward models · lab · app · keyring · recogniser · antispoof."""
from __future__ import annotations

import json
import os
import sys
import time

from .. import probe, ui
from ..engine import config as cfgmod
from ..engine import pam as pammod
from ..engine import store as storemod
from .common import need_root, real_user

# ----------------------------------------------------------------- models ----
def cmd_models(args) -> int:
    need_root("models")
    user = real_user(args)
    try:
        st = storemod.load(user, _recogniser_name(cfgmod.load()))
    except storemod.StoreError as e:
        ui.fail(str(e)); ui.tip("sudo ward enroll --fresh"); return 1
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
        from .. import appdata, attempts
        print(json.dumps({"user": user, "people": appdata.people(st, user, attempts.read(cfgmod.LOG_PATH, 500)),
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
    from ..engine import lab as labmod
    from ..engine.capture import CaptureError
    user = real_user(args)
    ev = labmod.LiveEvaluator(user)
    if not ev.store.models:
        ui.fail(f"no models for {user} — enroll first"); return 1
    try:
        ev.open()
    except CaptureError as e:
        ui.fail(str(e)); return 1
    if getattr(args, "record_frames", None):
        ui.info(f"recording frame sets to {ev.record_frames(args.record_frames)}  (replay: WARD_REPLAY=<dir> ward auth, or ward auth --replay <dir>)")
    for n in getattr(ev, "notes", []):
        ui.warn(n)
    if not args.text:
        try:
            from .. import display as disp
            screen = disp.open_display("ward lab", 640, 360, kind="lab", evaluator=ev)
            screen.run(ev)
            ev.close()
            print(labmod.format_summary(ev.summary()))
            return 0
        except Exception as e:                       # window unavailable → text mode
            ui.warn(f"no lab window ({e}); text mode")
    ui.banner("ward lab · text mode", "Ctrl+C for the summary. Policies: " + " · ".join(f"{p} = {labmod.POLICY_TEXT[p]}" for p in labmod.POLICIES))
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
        from .. import app
        return app.main([sys.argv[0]] + (["--selftest"] if getattr(args, "selftest", False) else []))
    except ImportError as e:
        ui.fail(f"the app needs GTK 4 + libadwaita (python3-gi, gir1.2-gtk-4.0, gir1.2-adw-1): {e}")
        ui.tip("everything is also available from the command line: ward setup / enroll / lab / doctor"); return 1


# ---------------------------------------------------------------- keyring ----
def cmd_keyring(args) -> int:
    """status | enable | disable | repair — let a face login unlock GNOME Keyring / KWallet (DESIGN.md §9b)."""
    from ..engine import keyring as kr
    user = real_user(args)
    if args.action == "status":
        st = kr.status(user)
        ui.info(f"credential for {user}: {'sealed and stored' if st['enrolled'] else 'unknown without root' if st['enrolled'] is None else 'not enrolled'} ({st['file']})")
        ui.info(f"sealing: {'TPM2' if st['tpm'] else 'host key'}; root disk {'encrypted' if st['encrypted_root'] else 'NOT encrypted'}; systemd-creds {'ok' if st['systemd_creds'] else 'missing'}")
        ps = pammod.status()
        (ui.ok if ps.get("native") else ui.warn)("PAM: " + ("native module — the token reaches the keyring" if ps.get("native") else "pam_exec line — cannot hand the token to the keyring; install gcc, re-run install.sh, then sudo ward pam enable"))
        dis = kr.disabled_keyring_lines()
        if dis:
            ui.fail(f"{len(dis)} keyring PAM line(s) were commented out by an earlier ward experiment ({', '.join(sorted({p.name for p, _ in dis}))}) — password logins do not unlock the keyring either")
            ui.tip("sudo ward keyring repair")
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
            ui.warn("PAM still uses pam_exec: the credential will be stored but cannot reach the keyring until the native module is enabled (gcc + install.sh + `sudo ward pam enable`)")
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
    from ..engine import recognise
    if cfg["recognition"].get("recogniser", "dlib") == "arcface":
        return f"arcface:{recognise.installed_variant() or 'unknown'}"
    return recognise.DLIB_NAME


def cmd_recogniser(args) -> int:
    """status | install [mbf|r50] | use dlib|arcface — the embedding network (DESIGN.md §6.4a)."""
    from ..engine import recognise
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
            ui.tip("switch with: sudo ward recogniser use arcface   (then re-enroll: models from another recogniser cannot be compared)")
        return 0
    if args.action == "use":
        if args.variant == "arcface" and not recognise.available():
            ui.fail("ArcFace is not installed"); ui.tip("sudo ward recogniser install"); return 1
        cfg["recognition"]["recogniser"] = args.variant
        cfgmod.save(cfg)
        ui.ok(f"recogniser = {args.variant}")
        ui.tip("re-enroll now: sudo ward enroll --fresh   (the store is per recogniser; the engine refuses a mismatched one)")
        return 0
    return 1


def cmd_antispoof(args) -> int:
    """status | on | off | test DIR — the bundled anti-spoof CNN voter (liveness/antispoof_cnn.py)."""
    from ..engine.liveness import antispoof_cnn as asc
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
        from ..engine.detect import Models
        if not args.dir:
            ui.fail("usage: ward antispoof test DIR   (a folder of colour PNG frames, e.g. a lab recording)"); return 1
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
        ui.fail("the bundled model is missing or corrupt — reinstall ward"); return 1
    cfg["liveness"]["antispoof_cnn"] = args.action == "on"
    cfgmod.save(cfg)
    ui.ok(f"liveness.antispoof_cnn = {cfg['liveness']['antispoof_cnn']}")
    return 0
