"""ward-auth's fast path: ask the resident engine (ward/daemon.py) for an attempt.

Imports only the standard library and ward.feedback, so the process PAM starts is up in a few
tens of milliseconds instead of ~0.7 s. When no daemon answers — not installed, not running,
crashed, wrong reply — it runs the in-process engine exactly as before (`ward auth`), so a
broken daemon can slow an attempt down but never lock anyone out (golden rule 3).
"""
from __future__ import annotations

import json
import os
import signal
import socket
import sys
from types import SimpleNamespace

CONNECT_TIMEOUT = 0.15


def _vote(d: dict):
    return SimpleNamespace(voter=d.get("voter", "?"), result=SimpleNamespace(value=d.get("result", "ABSTAIN")))


def connect(path: str | None = None):
    from .daemon import SOCK
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(CONNECT_TIMEOUT)
    try:
        s.connect(path or SOCK)
    except OSError:
        s.close()
        return None
    return s


def via_daemon(user: str, verbose: bool = False, path: str | None = None, fb=None, timeout: float = 8.0):
    """The attempt's outcome from the daemon (feedback events relayed to `fb`), or None when the
    daemon is unavailable BEFORE the attempt started (the caller then runs the engine itself)."""
    s = connect(path)
    if s is None:
        return None
    started = False                                        # bound before any I/O: the except reads it
    with s:
        try:
            s.settimeout(timeout)
            f = s.makefile("rwb", buffering=0)
            env = {k: os.environ[k] for k in ("PAM_SERVICE", "PAM_TTY", "PAM_TYPE", "PAM_RHOST", "PAM_RUSER", "PAM_USER",
                                               "SSH_CONNECTION", "DISPLAY", "WAYLAND_DISPLAY", "XDG_SESSION_ID") if k in os.environ}
            f.write((json.dumps({"user": user, "env": env, "verbose": verbose}) + "\n").encode())
            for raw in f:
                msg = json.loads(raw.decode())
                if "outcome" in msg or "event" in msg or "say" in msg:
                    started = True                         # an error before any of these: fall back
                if "outcome" in msg:
                    o = msg["outcome"]
                    o["votes"] = [_vote(v) for v in o.get("votes", [])]
                    return SimpleNamespace(**o)
                if "event" in msg and fb is not None:
                    kind = msg.pop("event")
                    fb.event(kind, **msg)
                elif "say" in msg and verbose:
                    print("  " + msg["say"], flush=True)
                elif "error" in msg:
                    return None if not started else SimpleNamespace(accepted=False, reason="daemon", detail=msg["error"],
                                                                    votes=[], ir_dist=None, rgb_dist=None, elapsed_ms=0, logline="")
        except (OSError, ValueError):
            if started:                                    # mid-attempt the cameras are the daemon's: no retry
                return SimpleNamespace(accepted=False, reason="daemon", detail="daemon connection lost",
                                       votes=[], ir_dist=None, rgb_dist=None, elapsed_ms=0, logline="")
    return None


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    plain = [a for a in argv if a in ("-v", "--verbose")] == argv          # replay / -U … go to the full CLI
    if plain and os.geteuid() == 0:
        verbose = bool(argv)
        user = os.environ.get("PAM_USER") or os.environ.get("SUDO_USER") or os.environ.get("USER") or "root"
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        signal.alarm(12)                                   # backstop: the kernel ends a wedged attempt
        try:
            from . import feedback
            beh = _behaviour()
            if beh.get("skip_login") and feedback.is_session_login(user):
                raise LookupError                          # the full path logs the skip_login decision
            fb = feedback.Feedback(user, beh, verbose=verbose)
            s = connect()
            if s is not None:
                s.close()
                fb.start()
                out = via_daemon(user, verbose, fb=fb)
                if out is not None:
                    fb.finish(out)
                    if verbose:
                        print(("✔ " if out.accepted else "✘ ") + out.logline, flush=True)
                    return 0 if out.accepted else 1
                if fb.spinner:                             # the daemon declined before starting: the engine decides
                    fb.spinner.stop(); fb.spinner = None
        except LookupError:
            pass
        except Exception:
            pass                                           # anything odd: the in-process engine decides
        signal.alarm(0)
    from .cli import main as cli_main
    return cli_main(["auth", *argv])


def _behaviour() -> dict:
    """The [behaviour] section without importing the engine (numpy): config.load is stdlib-only."""
    try:
        from .engine import config as cfgmod
        return cfgmod.load()["behaviour"]
    except Exception:
        return {}


if __name__ == "__main__":
    sys.exit(main())
