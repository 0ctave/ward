"""What the user sees while the camera looks (DESIGN.md §9a).

An authentication attempt is silent by nature — pam_exec runs the engine in the background
of a password prompt. This module gives it a face on whichever channel the caller offers:

- **terminal** (sudo, su, `faceup auth` in a shell): an animated status line on stderr while
  the camera works, then a ✔/✘ line with the distances and the time.
- **PAM conversation** (GDM, SDDM, the KDE lock screen, the GNOME/KDE polkit agents): the PAM
  line carries pam_exec's `stdout` option, which forwards every line we print to stdout as a
  PAM_TEXT_INFO message. The greeters render it under the password field — the same channel
  fprintd uses for "Swipe your finger". One line when the camera starts, one when it is done.
- **desktop notification** after the decision (config `behaviour.notify`): sent on the user's
  own session bus as the user, fire-and-forget, never on the auth path's critical time.

Channel choice is automatic: a terminal on stderr wins (sudo prints PAM info there too, so
both would double up); otherwise stdout lines are used when PAM_TYPE says pam_exec is the
caller. Nothing here can make an attempt fail: every write is wrapped.
"""
from __future__ import annotations

import os
import pwd
import shutil
import subprocess
import sys
import threading
import time

LOOKING = "Looking for your face…"
HOLD_STILL = "Face recognised — move your head a little"      # motion voter needs movement
HOLD = "Face recognised — checking liveness…"
RESULT = {
    "accept": "Face recognised",
    "no-face": "No face seen — enter your password",
    "dark": "Too dark for the camera — enter your password",
    "no-match": "Face not recognised — enter your password",
    "liveness": "Face recognised, liveness not confirmed — enter your password",
    "lockout": "Face unlock paused after repeated failed checks — enter your password",
    "policy": "Face unlock is off here — enter your password",
    "camera": "Camera unavailable — enter your password",
    "timeout": "Face unlock timed out — enter your password",
}
SILENT = {"ssh", "no-models", "store", "login"}        # nothing useful to tell the user in these cases
FRAMES = "◐◓◑◒"
ICON = {"accept": "faceup", "fail": "face-sad"}   # "faceup" is installed by install.sh (data/faceup.svg)


def result_text(reason: str) -> str | None:
    """Human line for an Outcome.reason; None when nothing should be said."""
    base = reason.split(":")[0]
    if base in SILENT:
        return None
    if base == "spoof":
        return "Liveness check failed — enter your password"
    return RESULT.get(base, RESULT["no-face"])


def service_name(service: str) -> str:
    """Friendly name of the PAM service, for the notification body."""
    s = (service or "").lower()
    for key, name in (("sudo", "sudo"), ("gdm", "the lock screen"), ("kde", "the lock screen"), ("sddm", "the login screen"),
                      ("polkit", "an admin prompt"), ("login", "login"), ("su", "su")):
        if key in s:
            return name
    return s or "a prompt"


def _detail(outcome) -> str:
    parts = []
    if outcome.ir_dist is not None:
        parts.append(f"IR {outcome.ir_dist:.2f}")
    if outcome.rgb_dist is not None:
        parts.append(f"RGB {outcome.rgb_dist:.2f}")
    for v in outcome.votes:
        mark = {"PASS": "✓", "FAIL": "✗"}.get(v.result.value, "–")
        parts.append(f"{v.voter} {mark}")
    parts.append(f"{outcome.elapsed_ms / 1000:.1f} s")
    return " · ".join(parts)


class _Spinner(threading.Thread):
    """One status line on a terminal, redrawn in place until stop()."""

    def __init__(self, stream, colour: bool):
        super().__init__(daemon=True, name="faceup-spinner")
        self.stream, self.colour = stream, colour
        self.text = LOOKING
        self._done = threading.Event()
        self.t0 = time.monotonic()

    def _c(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.colour else s

    def run(self) -> None:
        i = 0
        while not self._done.wait(0.1):
            try:
                self.stream.write(f"\r\033[K  {self._c('36', FRAMES[i % len(FRAMES)])} {self.text} {self._c('2', f'{time.monotonic() - self.t0:.1f}s')}")
                self.stream.flush()
            except (OSError, ValueError):
                return
            i += 1

    def stop(self) -> None:
        self._done.set()
        self.join(timeout=0.5)
        try:
            self.stream.write("\r\033[K")
            self.stream.flush()
        except (OSError, ValueError):
            pass


class Feedback:
    def __init__(self, user: str, behaviour: dict | None = None, verbose: bool = False,
                 err=None, out=None, env: dict | None = None):
        self.user = user
        self.beh = behaviour or {}
        self.err = err if err is not None else sys.stderr
        self.out = out if out is not None else sys.stdout
        self.env = os.environ if env is None else env
        self.service = self.env.get("PAM_SERVICE", "")
        self.tty = (not verbose) and self._isatty(self.err)
        self._own_tty = None
        # Under sudo, pam_exec's child has no terminal on stderr (observed: the PAM lines were
        # printed instead of the spinner). PAM_TTY names the caller's pseudo-terminal; write the
        # spinner there — only for terminal services and only on a /dev/pts device, never on a VT
        # (gdm sets PAM_TTY to a console or display name).
        pam_tty = self.env.get("PAM_TTY", "")
        if not self.tty and not verbose and self.service in TERMINAL_SERVICES and pam_tty.startswith("/dev/pts/"):
            try:
                f = open(pam_tty, "w")
                if f.isatty():
                    self._own_tty = f; self.err = f; self.tty = True
                else:
                    f.close()
            except OSError:
                pass
        self.pam = (not self.tty) and "PAM_TYPE" in self.env
        self.colour = self.tty and self.env.get("NO_COLOR") is None
        self.spinner: _Spinner | None = None
        self.hold_said: list[str] = []
        self.lines: list[str] = []          # what went to the PAM channel (tests, verbose)

    @staticmethod
    def _isatty(stream) -> bool:
        try:
            return bool(stream.isatty())
        except (AttributeError, ValueError):
            return False

    # -- channels -----------------------------------------------------------------
    def _pam_line(self, text: str) -> None:
        self.lines.append(text)
        if not self.pam:
            return
        try:
            self.out.write(text + "\n")
            self.out.flush()
        except (OSError, ValueError):
            pass

    def _term_line(self, text: str, colour: str) -> None:
        try:
            self.err.write(f"  {text}\n" if not self.colour else f"  \033[{colour}m{text}\033[0m\n")
            self.err.flush()
        except (OSError, ValueError):
            pass

    # -- lifecycle ----------------------------------------------------------------
    def start(self) -> None:
        if self.tty:
            self.spinner = _Spinner(self.err, self.colour)
            self.spinner.start()
        self._pam_line(LOOKING)

    def event(self, kind: str, **kw) -> None:
        """Engine progress: 'face' (first face seen), 'hold' (recognised, liveness pending; still=bool)."""
        if kind == "hold":
            # said at most twice: the generic line, then the "move" hint once motion reports a still head
            text = HOLD_STILL if kw.get("still") else HOLD
            if text in self.hold_said or (len(self.hold_said) >= 1 and text == HOLD):
                return
            self.hold_said.append(text)
            if self.spinner:
                self.spinner.text = text
            self._pam_line(text)
        elif kind == "face" and self.spinner:
            self.spinner.text = "Face found, recognising…"

    def finish(self, outcome) -> None:
        text = result_text(outcome.reason)
        if self.spinner:
            self.spinner.stop()
            self.spinner = None
            if text:
                if outcome.accepted:
                    self._term_line(f"✔ {text} · {_detail(outcome)}", "32")
                else:
                    self._term_line(f"✘ {text}", "33")
        elif text:
            self._pam_line(text)
        self.notify(outcome)
        if self._own_tty is not None:
            try:
                self._own_tty.close()
            except OSError:
                pass
            self._own_tty = None

    # -- notification -------------------------------------------------------------
    def notify(self, outcome) -> None:
        mode = str(self.beh.get("notify", "accept"))
        if mode not in ("all", "accept", "off"):
            mode = "accept"
        if mode == "off" or (mode == "accept" and not outcome.accepted) or result_text(outcome.reason) is None:
            return
        cmd = notify_command(outcome, self.service)
        if cmd is None:
            return
        try:
            spawn_as_user(self.user, cmd)
        except Exception:
            pass


def notify_command(outcome, service: str = "") -> list[str] | None:
    """The notify-send invocation for an outcome, or None when notify-send is missing."""
    exe = shutil.which("notify-send")
    if exe is None:
        return None
    if outcome.accepted:
        title, body, icon = "Unlocked with your face", f"{service_name(service)} · {_detail(outcome)}", ICON["accept"]
    else:
        title, body, icon = "Face unlock did not succeed", f"{result_text(outcome.reason)} ({service_name(service)})", ICON["fail"]
    # transient: GNOME ignores the expiry time and would keep every attempt in the notification list;
    # a transient one is shown briefly and never stored. -p/-r (below) make each attempt replace the last.
    return [exe, "-a", "faceup", "-i", icon, "-u", "low", "-t", "3000", "-e", "-h", "boolean:transient:true",
            "-h", "string:desktop-entry:faceup", "-p", title, body]


def replacing_shell(cmd: list[str], runtime_dir: str) -> list[str]:
    """Wrap a notify-send argv so it replaces the previous faceup notification instead of stacking:
    the id printed by -p is kept in the user's runtime dir and passed back with -r next time."""
    import shlex
    idfile = shlex.quote(f"{runtime_dir}/faceup-notify.id")
    argv = " ".join(shlex.quote(a) for a in cmd)
    script = (f'f={idfile}; r=$(cat "$f" 2>/dev/null || echo 0); '
              f'case "$r" in ""|*[!0-9]*) r=0;; esac; '
              f'{argv} -r "$r" > "$f" 2>/dev/null')
    return ["/bin/sh", "-c", script]


TERMINAL_SERVICES = ("sudo", "sudo-i", "su", "su-l", "doas")
LOGIN_SERVICES = ("gdm-password", "gdm-autologin", "sddm", "lightdm", "greetd", "login", "ly")


def is_session_login(user: str, env: dict | None = None) -> bool:
    """True when this PAM call is the FIRST login of a session (a greeter service and no
    session bus for the user yet) rather than a screen unlock. Used by behaviour.skip_login."""
    env = os.environ if env is None else env
    if env.get("PAM_SERVICE", "") not in LOGIN_SERVICES:
        return False
    try:
        uid = pwd.getpwnam(user).pw_uid
    except KeyError:
        return False
    return not os.path.exists(f"/run/user/{uid}/bus")


def session_env(user: str) -> tuple[dict[str, str], int, int] | None:
    """Environment to reach `user`'s session bus, with (uid, gid) — None when there is no session."""
    try:
        pw = pwd.getpwnam(user)
    except KeyError:
        return None
    runtime = f"/run/user/{pw.pw_uid}"
    if not os.path.exists(f"{runtime}/bus"):
        return None
    return ({"DBUS_SESSION_BUS_ADDRESS": f"unix:path={runtime}/bus", "XDG_RUNTIME_DIR": runtime,
             "HOME": pw.pw_dir, "USER": user, "LOGNAME": user, "PATH": "/usr/local/bin:/usr/bin:/bin"},
            pw.pw_uid, pw.pw_gid)


def spawn_as_user(user: str, cmd: list[str]) -> bool:
    """Start `cmd` detached as `user` on their session bus. Never waits; never inherits our
    stdout (pam_exec's pipe must close when we exit, or the greeter would hang on our notifier)."""
    se = session_env(user)
    if se is None:
        return False
    env, uid, gid = se
    if cmd and cmd[0].endswith("notify-send"):
        cmd = replacing_shell(cmd, env["XDG_RUNTIME_DIR"])
    kw: dict = {}
    if os.geteuid() == 0 and uid != 0:
        kw = {"user": uid, "group": gid, "extra_groups": []}
    subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True, **kw)
    return True
