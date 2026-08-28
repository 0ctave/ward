"""PAM integration (DESIGN.md §9): one `sufficient` pam_exec line before pam_unix, with backup."""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

COMMON_AUTH = Path(os.environ.get("FACEUP_PAM_FILE", "/etc/pam.d/common-auth"))
BACKUP = COMMON_AUTH.with_name(COMMON_AUTH.name + ".faceup-bak")
ENGINE = "/usr/local/bin/faceup-auth"
LINE = f"auth\tsufficient\t\t\tpam_exec.so quiet {ENGINE}\t# faceup\n"


def status() -> dict:
    text = COMMON_AUTH.read_text() if COMMON_AUTH.exists() else ""
    return {
        "file": str(COMMON_AUTH),
        "enabled": "faceup-auth" in text,
        "howdy": "pam_howdy" in text,
        "backup": BACKUP.exists(),
    }


def render_enabled(text: str) -> str:
    if "faceup-auth" in text:
        return text
    lines = text.splitlines(keepends=True)
    # insert before the first auth line that is a real module (pam_unix or pam_howdy)
    for i, ln in enumerate(lines):
        if re.match(r"^\s*auth\s", ln) and ("pam_unix" in ln or "pam_howdy" in ln):
            return "".join(lines[:i]) + LINE + "".join(lines[i:])
    # fallback: before the first auth line at all
    for i, ln in enumerate(lines):
        if re.match(r"^\s*auth\s", ln):
            return "".join(lines[:i]) + LINE + "".join(lines[i:])
    return text + LINE


def render_disabled(text: str) -> str:
    return "".join(ln for ln in text.splitlines(keepends=True) if "faceup-auth" not in ln)


def enable(dry_run: bool = False) -> str:
    text = COMMON_AUTH.read_text()
    new = render_enabled(text)
    if dry_run or new == text:
        return new
    if not BACKUP.exists():
        shutil.copy2(COMMON_AUTH, BACKUP)
    tmp = COMMON_AUTH.with_suffix(".faceup-tmp")
    tmp.write_text(new)
    os.chmod(tmp, 0o644)
    os.replace(tmp, COMMON_AUTH)
    return new


def disable(dry_run: bool = False) -> str:
    text = COMMON_AUTH.read_text()
    new = render_disabled(text)
    if dry_run or new == text:
        return new
    tmp = COMMON_AUTH.with_suffix(".faceup-tmp")
    tmp.write_text(new)
    os.chmod(tmp, 0o644)
    os.replace(tmp, COMMON_AUTH)
    return new


def disable_howdy(dry_run: bool = False, text: str | None = None) -> str:
    """Comment out pam_howdy lines (two face steps in a row are pointless).
    `text` lets a dry run chain on top of render_enabled() output."""
    if text is None:
        text = COMMON_AUTH.read_text()
    new = "".join(("#" + ln if "pam_howdy" in ln and not ln.lstrip().startswith("#") else ln)
                  for ln in text.splitlines(keepends=True))
    if not dry_run and new != text:
        if not BACKUP.exists():
            shutil.copy2(COMMON_AUTH, BACKUP)
        COMMON_AUTH.write_text(new)
    return new
