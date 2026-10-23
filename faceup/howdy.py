"""Talk to Howdy: config file (comment-preserving edits), models, CLI wrappers."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

CONFIG = Path("/etc/howdy/config.ini")
MODELS_DIRS = [Path("/etc/howdy/models"), Path("/usr/lib/howdy/models"), Path("/var/lib/howdy/models")]
DLIB_DIRS = [Path("/usr/share/dlib"), Path("/usr/lib/howdy/dlib-data"), Path("/usr/share/dlib-models")]


def installed() -> bool:
    return shutil.which("howdy") is not None and CONFIG.exists()


def get(key: str, default: str | None = None) -> str | None:
    try:
        text = CONFIG.read_text()
    except OSError:
        return default
    m = re.search(rf"^\s*{re.escape(key)}\s*=\s*(.*?)\s*$", text, re.M)
    return m.group(1) if m else default


def models(user: str) -> list[str]:
    """Labels of enrolled models for `user` (needs root to read the file)."""
    for d in MODELS_DIRS:
        f = d / f"{user}.dat"
        if f.exists():
            try:
                return [m.get("label", "?") for m in json.loads(f.read_text())]
            except (OSError, ValueError):
                return []
    return []


def dlib_model(name: str) -> Path | None:
    for d in DLIB_DIRS:
        p = d / name
        if p.exists():
            return p
    return None


def run(*args: str, user: str | None = None, yes: bool = False) -> int:
    """howdy [-U user] [-y] <command> [arguments]  — globals go before the command."""
    cmd = ["howdy"]
    if user:
        cmd += ["-U", user]
    if yes:
        cmd.append("-y")
    cmd += list(args)
    return subprocess.call(cmd)


def real_user() -> str:
    return os.environ.get("SUDO_USER") or os.environ.get("USER") or "root"
