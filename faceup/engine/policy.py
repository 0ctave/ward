"""Security policy (DESIGN.md §6.8): presets, per-service overrides, and the spoof lockout.

faceup is convenience-grade authentication with a password behind it. The policy makes the
trade-off explicit and tunable instead of scattered over a dozen keys:

    convenience   raw ≤ gate or whitened path · ≥ 1 PASS, 0 FAIL · learning on
    balanced      + RGB match required when the colour camera sees the face well · ≥ 2 PASS
                    when two voters are available · learning on (strict novelty)     ← default
    strict        gate tightened by 0.05 (never above 0.35), RGB match required, whitened path
                    off · 2 PASS including shadow or the CNN, else reject · learning off

`effective(cfg, service)` resolves the level for one PAM service (`security.policy_sudo`,
`policy_polkit`, `policy_login`, `policy_unlock` override the default) and returns the
recognition/liveness knobs the engine uses for THIS attempt — the config file itself is never
rewritten by an unlock.

Lockout: `security.lockout_failures` liveness FAILs (spoof signals — never "no face" or a
plain no-match) within `security.lockout_minutes` pause face unlock for that user for
`lockout_minutes`. The password path is untouched; a lockout can only ever fall to it.
"""
from __future__ import annotations

import time
from datetime import datetime

LEVELS = ("convenience", "balanced", "strict", "off")

PRESETS = {
    "convenience": {"require_rgb_match": False, "rgb_when_bright": False, "whitened": True,
                    "required_votes": 1, "need_strong_voter": False, "learn": True, "gate_delta": 0.0, "gate_max": 0.45},
    "balanced":    {"require_rgb_match": False, "rgb_when_bright": True, "whitened": True,
                    "required_votes": 2, "need_strong_voter": False, "learn": True, "gate_delta": 0.0, "gate_max": 0.45},
    "strict":      {"require_rgb_match": True, "rgb_when_bright": True, "whitened": False,
                    "required_votes": 2, "need_strong_voter": True, "learn": False, "gate_delta": -0.05, "gate_max": 0.35},
}
STRONG_VOTERS = ("shadow", "cnn")
SERVICE_KEYS = {"sudo": "policy_sudo", "sudo-i": "policy_sudo", "su": "policy_sudo", "su-l": "policy_sudo", "doas": "policy_sudo",
                "polkit-1": "policy_polkit", "gdm-password": "policy_unlock", "kde": "policy_unlock", "sddm": "policy_login",
                "lightdm": "policy_login", "login": "policy_login", "greetd": "policy_login"}


def level_for(cfg: dict, service: str = "") -> str:
    sec = cfg.get("security", {})
    key = SERVICE_KEYS.get((service or "").lower(), "")
    lvl = (sec.get(key) if key else "") or sec.get("policy", "balanced")
    return lvl if lvl in LEVELS else "balanced"


def effective(cfg: dict, service: str = "") -> dict:
    """Resolved knobs for one attempt: {'level', 'certainty', 'rgb_certainty', 'whitened',
    'require_rgb_match', 'rgb_when_bright', 'required_votes', 'need_strong_voter', 'learn', 'off'}."""
    lvl = level_for(cfg, service)
    rec, live, beh = cfg["recognition"], cfg["liveness"], cfg["behaviour"]
    if lvl == "off":
        return {"level": "off", "off": True}
    p = PRESETS[lvl]
    gate = min(float(rec["certainty"]) + p["gate_delta"], p["gate_max"])
    rgb_gate = min(float(rec["rgb_certainty"]) + p["gate_delta"], p["gate_max"])
    return {"level": lvl, "off": False,
            "certainty": round(gate, 3), "rgb_certainty": round(rgb_gate, 3),
            "whitened": bool(rec.get("whitened", True)) and p["whitened"],
            "require_rgb_match": bool(rec.get("require_rgb_match", False)) or p["require_rgb_match"],
            "rgb_when_bright": p["rgb_when_bright"],
            "required_votes": max(int(live.get("required_votes", 2)), p["required_votes"]) if lvl != "convenience" else 1,
            "need_strong_voter": p["need_strong_voter"],
            "learn": bool(beh.get("learn", True)) and p["learn"]}


def apply_preset(cfg: dict, level: str) -> dict:
    """`faceup tune policy X`: record the level; the presets are resolved at attempt time so the
    measured gate (calibration) stays the source of truth. Returns what changed."""
    if level not in LEVELS:
        raise ValueError(f"policy must be one of {', '.join(LEVELS)}")
    sec = cfg.setdefault("security", {})
    old = sec.get("policy", "balanced")
    sec["policy"] = level
    return {"security.policy": (old, level)}


def describe(level: str) -> str:
    return {"convenience": "raw gate or whitened path · one liveness PASS · learning on",
            "balanced": "RGB must agree when it sees you well · two PASSes when two voters exist · learning on",
            "strict": "gate −0.05 (≤ 0.35) · RGB match required · no whitened path · shadow/CNN PASS required · learning off",
            "off": "face unlock disabled for this service"}.get(level, level)


# -- lockout ---------------------------------------------------------------------------------------
def spoof_times(log_path, user: str, since_s: float) -> list[float]:
    """Timestamps (epoch) of liveness FAIL results for `user` in the last `since_s` seconds."""
    out: list[float] = []
    try:
        lines = open(log_path).read().splitlines()[-400:]
    except OSError:
        return out
    now = time.time()
    for ln in lines:
        if f"user={user} " not in ln or "result=spoof" not in ln:
            continue
        try:
            ts = datetime.strptime(ln[:19], "%Y-%m-%dT%H:%M:%S").timestamp()
        except ValueError:
            continue
        if now - ts <= since_s:
            out.append(ts)
    return out


def locked_out(cfg: dict, log_path, user: str) -> tuple[bool, int]:
    """(locked, seconds remaining). N spoof FAILs within lockout_minutes → paused for lockout_minutes."""
    sec = cfg.get("security", {})
    n = int(sec.get("lockout_failures", 0) or 0)
    minutes = float(sec.get("lockout_minutes", 15) or 15)
    if n <= 0:
        return False, 0
    times = spoof_times(log_path, user, minutes * 60)
    if len(times) < n:
        return False, 0
    remaining = int(minutes * 60 - (time.time() - max(times)))
    return True, max(0, remaining)
