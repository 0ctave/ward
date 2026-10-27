"""PAM integration (DESIGN.md §9): one ward line in the shared auth stack, with backup."""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

# the shared auth stack every service includes: Debian/Ubuntu/openSUSE `common-auth`,
# Fedora/RHEL/Arch `system-auth` (both are often symlinks to a managed file — see _write)
CANDIDATES = (Path("/etc/pam.d/common-auth"), Path("/etc/pam.d/system-auth"))


def target() -> Path:
    if os.environ.get("WARD_PAM_FILE"):
        return Path(os.environ["WARD_PAM_FILE"])
    return next((p for p in CANDIDATES if p.exists()), CANDIDATES[0])


COMMON_AUTH = target()
BACKUP = COMMON_AUTH.with_name(COMMON_AUTH.name + ".ward-bak")


class PamError(Exception):
    pass


def manager(path: Path | None = None) -> str:
    """Who owns the stack: 'authselect' (Fedora/RHEL: edits are overwritten by `authselect
    apply-changes`), 'pam-config' (openSUSE), 'pam-auth-update' (Debian), or ''."""
    real = str(Path(path or COMMON_AUTH).resolve())
    if "/authselect/" in real:
        return "authselect"
    if real.endswith("-pc"):
        return "pam-config"
    if os.path.isdir("/usr/share/pam-configs"):
        return "pam-auth-update"
    return ""


def _write(path: Path, text: str) -> None:
    """Atomic, symlink-preserving write of a PAM file: a half-written auth stack locks every
    login out (golden rule 3), and replacing a symlink (openSUSE common-auth → common-auth-pc)
    would silently detach it from its manager."""
    real = Path(path).resolve()
    tmp = real.with_name(real.name + ".ward-tmp")
    with open(tmp, "w") as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    try:
        os.chmod(tmp, real.stat().st_mode & 0o777 if real.exists() else 0o644)
    except OSError:
        os.chmod(tmp, 0o644)
    os.replace(tmp, real)
from ..paths import AUTH as ENGINE, PAM_MODULE as MODULE  # /usr/local (install.sh) or /usr (package)
# seteuid: pam_exec runs the helper with the REAL uid of the PAM application by default — for
# sudo that is the invoking user, who cannot read the root-only model store (observed: exit 1 in
# milliseconds, no log line). With seteuid it runs with the effective uid, i.e. root.
# stdout: every line the engine prints is forwarded as a PAM_TEXT_INFO message, which GDM, SDDM,
# the KDE lock screen and the polkit agents show under the password field (feedback.py).
LINE = f"auth\tsufficient\t\t\tpam_exec.so quiet stdout seteuid {ENGINE}\t# ward\n"
# Native module (DESIGN.md §9b): forwards the helper's lines itself and can hand PAM the sealed
# keyring credential. It is inserted with a `[success=N default=ignore]` jump to the stack's
# pam_permit line — like pam_unix — instead of `sufficient`, so the caller's own lines after the
# @include (pam_gnome_keyring, pam_kwallet) still run and receive the token.


def native_available() -> bool:
    return os.path.exists(MODULE)


def _native_line(skip: int) -> str:
    return f"auth\t[success={skip} default=ignore]\t{MODULE}\t# ward\n"


def status() -> dict:
    text = COMMON_AUTH.read_text() if COMMON_AUTH.exists() else ""
    mine = [ln for ln in text.splitlines() if "ward-auth" in ln or "pam_ward.so" in ln]
    native = [ln for ln in mine if "pam_ward.so" in ln]
    return {
        "file": str(COMMON_AUTH),
        "manager": manager(),
        "enabled": bool(mine),
        "native": bool(native),
        "current": bool(mine) and (bool(native) or all("seteuid" in ln and "stdout" in ln for ln in mine)),   # else `enable` upgrades it
        "howdy": any("pam_howdy" in ln and not ln.lstrip().startswith("#") for ln in text.splitlines()),   # active lines only
        "backup": BACKUP.exists(),
    }


def _is_ours(ln: str) -> bool:
    return "ward-auth" in ln or "pam_ward.so" in ln


def _skip_count(lines: list[str], at: int) -> int:
    """Auth lines between position `at` and the stack's pam_deny (inclusive): what a success must
    jump over to land on pam_permit. Without a pam_deny, jump over pam_unix/pam_sss only."""
    n = 0
    for ln in lines[at:]:
        if not re.match(r"^\s*auth\s", ln):
            continue
        n += 1
        if "pam_deny" in ln:
            return n
    m = sum(1 for ln in lines[at:] if re.match(r"^\s*auth\s", ln) and ("pam_unix" in ln or "pam_sss" in ln))
    return max(1, m)


# Second-factor auth modules: if a required/requisite one sits AFTER the ward line (before pam_deny),
# ward's success-jump [success=N] lands on pam_permit past it, so a face accept would SKIP the second
# factor. We refuse to write that rather than silently weaken 2FA (ROADMAP §6: 2FA modules stay put).
# (pam_faillock is deliberately NOT here: its authfail line only acts on FAILURE — skipping it on a
# success is harmless — and it is in Fedora's default stack, so flagging it would wrongly refuse there.)
SECOND_FACTORS = ("pam_u2f", "pam_google_authenticator", "pam_yubico", "pam_duo", "pam_oath", "pam_otpw")


def bypassed_second_factor(text: str) -> str | None:
    """The name of a required/requisite second-factor module that a face accept would jump over, or
    None. The jump skips auth lines from the ward line through pam_deny inclusive."""
    lines = text.splitlines()
    at = next((i for i, ln in enumerate(lines) if _is_ours(ln)), None)
    if at is not None:
        start = at + 1
    else:                                             # not inserted yet: where render_enabled would put it
        at = next((i for i, ln in enumerate(lines) if re.match(r"^\s*auth\s", ln) and ("pam_unix" in ln or "pam_howdy" in ln)), None)
        if at is None:
            return None
        start = at
    for ln in lines[start:]:
        if not re.match(r"^\s*auth\s", ln):
            continue
        if "pam_deny" in ln:
            break
        if re.search(r"\b(required|requisite)\b", ln):
            hit = next((sf for sf in SECOND_FACTORS if sf in ln), None)
            if hit:
                return hit
    return None


def render_enabled(text: str, native: bool | None = None) -> str:
    """Insert (or upgrade in place) ward's line. native=None → the module if it is installed."""
    native = native_available() if native is None else native
    lines = text.splitlines(keepends=True)
    if any(_is_ours(ln) for ln in lines):
        out = []
        for i, ln in enumerate(lines):
            if not _is_ours(ln):
                out.append(ln); continue
            want = _native_line(_skip_count(lines, i + 1)) if native else LINE
            current_ok = ("pam_ward.so" in ln) if native else (ENGINE in ln and "seteuid" in ln and "stdout" in ln)
            out.append(ln if (current_ok and ("pam_ward.so" in ln) == native and (not native or ln == want)) else want)
        return "".join(out)
    # insert before the first auth line that is a real module (pam_unix or pam_howdy)
    at = next((i for i, ln in enumerate(lines) if re.match(r"^\s*auth\s", ln) and ("pam_unix" in ln or "pam_howdy" in ln)), None)
    if at is None:
        at = next((i for i, ln in enumerate(lines) if re.match(r"^\s*auth\s", ln)), None)
    if at is None:
        return text + (_native_line(1) if native else LINE)
    line = _native_line(_skip_count(lines, at)) if native else LINE
    return "".join(lines[:at]) + line + "".join(lines[at:])


def render_disabled(text: str) -> str:
    return "".join(ln for ln in text.splitlines(keepends=True) if not _is_ours(ln))


def _read() -> str:
    if not COMMON_AUTH.exists():
        raise PamError(f"no shared PAM auth stack found ({' or '.join(str(p) for p in CANDIDATES)}): "
                       "add the ward line to the service files (/etc/pam.d/sudo …) by hand")
    return COMMON_AUTH.read_text()


def enable(dry_run: bool = False, native: bool | None = None, force: bool = False) -> str:
    text = _read()
    sf = bypassed_second_factor(text)
    if sf and not force:
        raise PamError(f"a second-factor auth module ({sf}) sits after where ward goes, and a face accept "
                       f"would jump past it — refusing to silently weaken your two-factor auth. Put the ward "
                       f"line after {sf} by hand, or re-run with --force if you understand the trade-off.")
    new = render_enabled(text, native)
    if dry_run or new == text:
        return new
    if manager() == "authselect":
        raise PamError(f"{COMMON_AUTH} is managed by authselect and would be overwritten: create a custom "
                       "profile (`authselect create-profile ward -b $(authselect current -r | cut -d' ' -f1)`), "
                       "add the ward line to its system-auth, then `authselect select custom/ward`")
    if not BACKUP.exists():
        shutil.copy2(COMMON_AUTH, BACKUP)
    _write(COMMON_AUTH, new)
    return new


def disable(dry_run: bool = False) -> str:
    text = _read()
    new = render_disabled(text)
    if dry_run or new == text:
        return new
    _write(COMMON_AUTH, new)
    return new


def disable_howdy(dry_run: bool = False, text: str | None = None) -> str:
    """Comment out pam_howdy lines (two face steps in a row are pointless).
    `text` lets a dry run chain on top of render_enabled() output."""
    if text is None:
        text = _read()
    new = "".join(("#" + ln if "pam_howdy" in ln and not ln.lstrip().startswith("#") else ln)
                  for ln in text.splitlines(keepends=True))
    if not dry_run and new != text:
        if not BACKUP.exists():
            shutil.copy2(COMMON_AUTH, BACKUP)
        _write(COMMON_AUTH, new)
    return new
