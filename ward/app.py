"""The ward manager — one GTK 4 / libadwaita window to run face unlock without a terminal.

Pages: Overview (is it on, does it work, what needs fixing), People (who can unlock this login:
register, re-enrol, conditions, remove), Reliability (rates by light, service and person, failure
reasons, next steps, test unlock), Settings (security policy, behaviour, login integration) and
Advanced (setup steps, lab, maintenance). The app is a DRIVER over the command line: every action
runs the matching `ward …` command as a subprocess whose output streams into the console, so
the CLI stays complete on its own and the two can never disagree.

Privileges: the window runs as the logged-in user — a GUI toolkit never runs as root. Reading the
status, the config and the log needs nothing; each action that touches /etc/ward, PAM or the
keyring runs through `pkexec` (polkit action org.ward.app: administrator password, asked
for each action — not remembered, so no other process can reuse it). The face store is 0600, so the People page asks once to show it and reads it
via `ward models --json`. Started as root (`sudo ward app`) it simply runs everything directly.
"""
from __future__ import annotations

import os
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
CONDITIONS = ("glasses", "sunglasses", "headset", "hat", "beard")   # night needs none: IR covers the dark
from .paths import BIN as INSTALLED, LIB as INSTALLED_LIB
PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENROL = ["enroll", "--fresh", "--selfcheck"]
CAMERA_CMDS = ("enroll", "calibrate")       # drawn inside the app (display.StreamDisplay), not in a second window
PROFILE_TEXT = {"A": "IR + colour camera found", "B": "IR camera found", "C": "colour camera only"}


def strip_ansi(s: str) -> str:
    return ANSI.sub("", s)


def ward_cmd(argv: list[str]) -> list[str]:
    """The command line for `ward …` run by this same interpreter and package."""
    return [sys.executable, "-m", "ward", *argv]


def priv_cmd(argv: list[str], root: bool | None = None, pkg_parent: str = PKG_PARENT) -> list[str]:
    """`ward …` with administrator rights. As root: directly. Otherwise through pkexec — the
    root-owned installed launcher (it carries the polkit action). Never a user-writable checkout as
    root (any process of the user could edit it before the next action) unless the developer opts in
    with WARD_DEV_PKEXEC=1; without an install the command just explains what to do."""
    if root if root is not None else os.geteuid() == 0:
        return ward_cmd(argv)
    if os.environ.get("WARD_DEV_PKEXEC") == "1" and pkg_parent not in ("/usr/local/lib/ward", "/usr/lib/ward", INSTALLED_LIB):
        return ["pkexec", "env", f"PYTHONPATH={pkg_parent}", "NO_COLOR=1", sys.executable, "-m", "ward", *argv]
    if os.path.exists(INSTALLED):
        return ["pkexec", INSTALLED, *argv]
    return ["sh", "-c", "echo 'ward is not installed: run sudo ./install.sh first "
            "(or WARD_DEV_PKEXEC=1 to run this checkout as root)' >&2; exit 1"]


def child_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = PKG_PARENT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("NO_COLOR", "1")
    return env


def pct(x) -> str:
    return "—" if x is None else f"{round(100 * x)} %"


def describe_stats(s: dict) -> str:
    if not s or not s.get("attempts"):
        return "no attempts"
    med = f" · median {s['median_ms'] / 1000:.1f} s" if s.get("median_ms") is not None else ""
    return f"{pct(s['rate'])} of {s['attempts']}{med}"


def main(argv=None) -> int:
    """Load GTK, prepare the environment, then build and run the window (ward/app_window.py).

    The order is the one the single-closure version had: GTK is imported first, the environment
    is set after, the window classes are defined last."""
    import gi
    gi.require_version("Gtk", "4.0"); gi.require_version("Adw", "1")
    from gi.repository import Adw, GLib, Gtk  # noqa: F401
    from .display import user_wayland_env
    for k, v in user_wayland_env().items():
        os.environ.setdefault(k, v)
    os.environ.setdefault("GSK_RENDERER", "ngl")
    ROOT = os.geteuid() == 0
    if ROOT:
        os.environ.setdefault("GTK_A11Y", "none")
    from .app_window import App
    app = App()
    args = list(argv if argv is not None else sys.argv[:1])
    app.selftest = "--selftest" in args
    return app.run([a for a in args if a != "--selftest"])
