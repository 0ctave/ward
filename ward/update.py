"""Update check: is a newer Ward released? Best-effort, explicit, never on the auth path.

Security tools go stale dangerously, so Ward can tell you when a newer release exists. Rules:
  * It runs ONLY when you ask (`ward update`) — never in the background, never during an unlock
    (`ward-auth`, the PAM module and the daemon never import this), so it adds no latency or
    network traffic to authentication.
  * One anonymous HTTPS GET to the GitHub releases API, short timeout, no data sent about you, no
    telemetry. Offline or any error → silent (returns None).
  * `ward update` caches the result; `ward doctor` only reads that cache (it never hits the network
    itself), so diagnostics stay fast and offline by default.
Stdlib only (urllib) — no new dependency.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from . import __version__

RELEASES_API = "https://api.github.com/repos/0ctave/ward/releases/latest"
CACHE = Path(os.environ.get("WARD_CACHE") or (Path.home() / ".cache" / "ward")) / "update.json"
CACHE_MAX_AGE_S = 86400          # doctor trusts a cached result for a day


def _parse(ver: str) -> tuple:
    """'v0.3.1' / '0.3.1-rc2' → (0, 3, 1) for comparison; non-numeric parts become 0."""
    core = ver.strip().lstrip("vV").split("-")[0].split("+")[0]
    out = []
    for p in core.split("."):
        try:
            out.append(int(p))
        except ValueError:
            out.append(0)
    return tuple(out) or (0,)


def is_newer(latest: str, current: str | None = None) -> bool:
    return _parse(latest) > _parse(current or __version__)


def fetch_latest(timeout: float = 4.0) -> str | None:
    """The latest release tag from GitHub, or None (offline, no releases yet, rate-limited, error)."""
    import urllib.request
    try:
        req = urllib.request.Request(
            RELEASES_API,
            headers={"Accept": "application/vnd.github+json", "User-Agent": f"ward/{__version__}"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:   # noqa: S310 (fixed https URL)
            data = json.load(r)
        tag = data.get("tag_name")
        return tag or None
    except Exception:
        return None                                              # best-effort: never raise to the caller


def check(write_cache: bool = True) -> dict:
    """Hit the network and return {'current', 'latest', 'checked'}; caches when a tag came back."""
    latest = fetch_latest()
    rec = {"current": __version__, "latest": latest, "checked": int(time.time())}
    if write_cache and latest:
        try:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(rec))
        except OSError:
            pass
    return rec


def cached(max_age_s: int = CACHE_MAX_AGE_S) -> dict | None:
    """A recent cached result (written by `ward update`), or None. Never touches the network."""
    try:
        rec = json.loads(CACHE.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(rec, dict) or not rec.get("latest"):
        return None
    if time.time() - int(rec.get("checked", 0)) > max_age_s:
        return None
    return rec
