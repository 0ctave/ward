"""faceup auth · daemon · pam · log — the unlock path and its integration."""
from __future__ import annotations

import json
import os
import subprocess

from .. import ui
from ..engine import config as cfgmod
from ..engine import pam as pammod
from .common import need_root, real_user

# ------------------------------------------------------------------- auth ----
def cmd_auth(args) -> int:
    """The engine entry point. PAM calls it via /usr/local/bin/faceup-auth; exit 0 = authenticated."""
    from .. import feedback
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
        from ..engine.auth import Outcome, log
        out = Outcome(False, "login", user, detail="skip_login: keyring stays unlockable"); log(out)
        return 1
    fb.start()
    from ..engine.auth import Engine
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
        from .. import client
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
        from .. import daemon
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
    from .. import attempts
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
