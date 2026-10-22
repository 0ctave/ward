"""Keyring credential (DESIGN.md §9b, roadmap K1): let a face login unlock GNOME Keyring / KWallet.

Those PAM modules unlock the login keyring with the password PAM received. `faceup keyring
enable` stores the login password *sealed* so that pam_faceup.so can hand it to PAM after a
face accept:

- sealed with `systemd-creds encrypt`, bound to the TPM2 (PCR 7 = Secure Boot state) when
  the machine has one; otherwise with the host key, which is only acceptable on an
  encrypted disk (checked; refused otherwise unless forced);
- file `/etc/faceup/keyring/<user>.cred`, 0600 root, named `faceup-keyring-<user>` so a
  credential cannot be replayed for another user;
- verified before sealing against the user's /etc/shadow hash (crypt), so a typo cannot be
  stored;
- released only by the native module, only after the helper reported an accept.

Everything is reversible: `keyring disable` deletes the file; the PAM lines that
pam_gnome_keyring needs are restored by `repair_pam_lines()` if an earlier experiment
commented them out (tag `# faceup-keyring`).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from . import config

CREDS_DIR = config.ETC / "keyring"
CREDS_BIN = shutil.which("systemd-creds") or "/usr/bin/systemd-creds"
PAM_D = Path(os.environ.get("FACEUP_PAM_DIR", "/etc/pam.d"))
TAG = "# faceup-keyring"


def cred_path(user: str) -> Path:
    if not user or "/" in user or user.startswith("."):
        raise ValueError("invalid user name")
    return CREDS_DIR / f"{user}.cred"


def has_tpm() -> bool:
    return os.path.exists("/dev/tpmrm0") or os.path.exists("/dev/tpm0")


def root_is_encrypted() -> bool:
    """True when the root filesystem sits on a dm-crypt device (LUKS)."""
    try:
        out = subprocess.run(["lsblk", "-rno", "TYPE,MOUNTPOINT"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    lines = out.splitlines()
    return any(ln.split()[0] == "crypt" for ln in lines if ln.strip()) and any(ln.endswith(" /") for ln in lines)


def verify_password(user: str, password: str) -> bool | None:
    """Check against /etc/shadow (root only). None = cannot verify (no shadow access or unsupported hash)."""
    try:
        import spwd
        entry = spwd.getspnam(user)
    except (ImportError, KeyError, PermissionError):
        try:
            for ln in open("/etc/shadow"):
                if ln.startswith(user + ":"):
                    h = ln.split(":")[1]; break
            else:
                return None
        except OSError:
            return None
    else:
        h = entry.sp_pwdp
    if not h or h.startswith(("!", "*")):
        return None
    got = _crypt(password, h)
    if got is None:
        return None
    if got is None or len(got) < 10:
        return None
    return got == h


def choose_key(prefer_tpm: bool = False, force_host_key: bool = False) -> str:
    """Which systemd-creds key to seal with.

    Default: the **host key** when the root disk is encrypted — it is protected at rest by full-disk
    encryption and needs no TPM round-trip, so unlocking the keyring is instant (a TPM2 decrypt on a
    firmware TPM adds 2–4 s to every login, felt as a freeze). Fall back to the TPM when there is no
    FDE; `prefer_tpm` forces the TPM (stronger: bound to PCR 7 / Secure Boot); `force_host_key`
    accepts the host key even without FDE."""
    if prefer_tpm and has_tpm():
        return "tpm2"
    if root_is_encrypted() or force_host_key:
        return "host"
    if has_tpm():
        return "tpm2"
    return "host"


def _crypt(password: str, setting: str) -> str | None:
    """crypt(3) via the stdlib module when present (≤ 3.12), else libcrypt through ctypes (3.13+)."""
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import crypt                               # removed in Python 3.13
            return crypt.crypt(password, setting)
    except ImportError:
        pass
    except Exception:
        return None
    try:
        import ctypes, ctypes.util
        lib = None
        # find_library needs ldconfig/gcc (missing in minimal roots); musl has crypt in libc itself
        for name in (ctypes.util.find_library("crypt"), "libcrypt.so.1", "libcrypt.so.2", None):
            try:
                cand = ctypes.CDLL(name)
            except OSError:
                continue
            if hasattr(cand, "crypt"):
                lib = cand; break
        if lib is None:
            return None
        lib.crypt.restype = ctypes.c_char_p
        lib.crypt.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        out = lib.crypt(password.encode(), setting.encode())
        return out.decode() if out else None
    except Exception:
        return None


def seal(user: str, password: str, force_host_key: bool = False, prefer_tpm: bool = False, creds_bin: str | None = None) -> tuple[Path, str]:
    """Encrypt and store; returns (path, key kind 'tpm2' | 'host')."""
    exe = creds_bin or CREDS_BIN
    if not os.path.exists(exe):
        raise RuntimeError("systemd-creds not found (systemd ≥ 250 required)")
    kind = choose_key(prefer_tpm, force_host_key)
    if kind == "host" and not root_is_encrypted() and not force_host_key:
        raise RuntimeError("no TPM and the root disk is not encrypted: a host-key-sealed password would be readable from a stolen disk (use --force-host-key to accept that)")
    p = cred_path(user)
    p.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(p.parent, 0o700)
    tmp = p.with_suffix(".cred.tmp")
    cmd = [exe, "encrypt", f"--name=faceup-keyring-{user}", f"--with-key={kind}", "-", str(tmp)]
    r = subprocess.run(cmd, input=password.encode(), capture_output=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"systemd-creds encrypt failed: {r.stderr.decode(errors='replace').strip()}")
    os.chmod(tmp, 0o600)
    os.replace(tmp, p)
    return p, kind


def unseal(user: str, creds_bin: str | None = None) -> str | None:
    """What pam_faceup.so does; used by `keyring status --check` (root)."""
    exe = creds_bin or CREDS_BIN
    p = cred_path(user)
    if not p.exists():
        return None
    r = subprocess.run([exe, "decrypt", f"--name=faceup-keyring-{user}", str(p), "-"], capture_output=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"systemd-creds decrypt failed: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout.decode().rstrip("\r\n")


def remove(user: str) -> bool:
    p = cred_path(user)
    if p.exists():
        p.unlink(); return True
    return False


def enrolled_key(user: str) -> str | None:
    """Which key an enrolled credential uses ('tpm2' | 'host'), by reading its header. Root only."""
    p = cred_path(user)
    try:
        head = p.read_bytes()[:4096]
    except OSError:
        return None
    # systemd-creds writes a text header naming the key; fall back to a substring probe
    return "tpm2" if b"tpm2" in head else ("host" if b"host" in head or head else None)


def status(user: str) -> dict:
    p = cred_path(user)
    try:
        enrolled: bool | None = p.exists()
    except PermissionError:
        enrolled = None                      # the directory is 0700 root: unknown without root
    return {"file": str(p), "enrolled": enrolled, "tpm": has_tpm(), "encrypted_root": root_is_encrypted(),
            "systemd_creds": os.path.exists(CREDS_BIN), "recommended_key": choose_key()}


# -- PAM lines pam_gnome_keyring / pam_kwallet need ------------------------------------------
def disabled_keyring_lines(pam_dir: Path = PAM_D) -> list[tuple[Path, str]]:
    """Lines an earlier faceup experiment commented out (tagged), per file."""
    found = []
    try:
        files = sorted(p for p in pam_dir.iterdir() if p.is_file() and not p.name.endswith((".faceup-bak", ".pam-old", ".faceup-tmp")))
    except OSError:
        return found
    for p in files:
        try:
            for ln in p.read_text().splitlines():
                if TAG in ln and ln.lstrip().startswith("#"):
                    found.append((p, ln))
        except OSError:
            continue
    return found


def repair_pam_lines(pam_dir: Path = PAM_D, dry_run: bool = False) -> list[str]:
    """Uncomment the tagged lines so password logins unlock the keyring again. Returns what changed."""
    changed = []
    for p, _ in {(p, None) for p, _ in disabled_keyring_lines(pam_dir)}:
        text = p.read_text()
        new = re.sub(r"^(\s*)#\s*(auth|session|password)(\s+.*?)\s*" + re.escape(TAG) + r"[ \t]*$", r"\1\2\3", text, flags=re.M)
        if new != text:
            changed.append(str(p))
            if not dry_run:
                tmp = p.with_suffix(p.suffix + ".faceup-tmp")
                tmp.write_text(new); os.chmod(tmp, 0o644); os.replace(tmp, p)
    return changed
