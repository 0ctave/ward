"""Configuration: /etc/faceup/config.toml with defaults (DESIGN.md §5.1).

Paths can be redirected with FACEUP_ETC / FACEUP_LOG for tests and unprivileged runs.
"""
from __future__ import annotations

import copy
import os
import re
import tomllib
from pathlib import Path
from typing import Any

ETC = Path(os.environ.get("FACEUP_ETC", "/etc/faceup"))
CONFIG_PATH = ETC / "config.toml"
MODELS_DIR = ETC / "models"
LOG_PATH = Path(os.environ.get("FACEUP_LOG", "/var/log/faceup.log"))
ANTISPOOF_MODEL = ETC / "antispoof.onnx"

DEFAULTS: dict[str, dict[str, Any]] = {
    "cameras": {"ir": "", "rgb": "", "ir_dark_threshold": 25, "rgb_dark_threshold": 40, "force_roles": False},
    "recognition": {
        "certainty": 0.40, "rgb_certainty": 0.38, "fusion_frames": 4,
        "require_rgb_match": False, "detector": "yunet", "max_height": 320, "accel": "cpu",
    },
    "liveness": {
        "required_votes": 2, "cross_spectrum": True, "antispoof_cnn": True,
        "shadow_test": "auto", "motion": True, "motion_challenge": False,
        "require_liveness_when_available": True,
    },
    "timing": {"settle_ms": 0, "window_ms": 2500, "hard_timeout_ms": 4000},
    "behaviour": {"abort_if_ssh": True, "lockout_failures": 0, "log_images": False},
}

_COMMENTS = {
    "cameras": "Stable ids usb:<vid>:<pid>:<iface> (setup writes them), 'auto', or a /dev path; '' = none.",
    "recognition": "certainty = max embedding distance (dlib scale: 0.6 is 'same person', 0.40 is strict).",
    "liveness": "Voters return PASS/FAIL/ABSTAIN; `required_votes` PASSes and zero FAILs are needed.",
    "timing": "Milliseconds. settle_ms > 0 waits for auto-exposure before the window (not needed with YuNet).",
    "behaviour": "log_images saves failed frames under /var/lib/faceup/snapshots (opt-in).",
}


def load() -> dict[str, dict[str, Any]]:
    cfg = copy.deepcopy(DEFAULTS)
    try:
        data = tomllib.loads(CONFIG_PATH.read_text())
    except FileNotFoundError:
        return cfg
    for section, values in data.items():
        if section in cfg and isinstance(values, dict):
            cfg[section].update(values)
    return cfg


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return '"' + v.replace('"', '\\"') + '"'
    return str(v)


def dump(cfg: dict[str, dict[str, Any]]) -> str:
    out = ["# faceup configuration — see docs/DESIGN.md §5.1", ""]
    for section, values in cfg.items():
        out.append(f"[{section}]")
        if section in _COMMENTS:
            out.append(f"# {_COMMENTS[section]}")
        width = max(len(k) for k in values)
        for k, v in values.items():
            out.append(f"{k.ljust(width)} = {_fmt(v)}")
        out.append("")
    return "\n".join(out)


def save(cfg: dict[str, dict[str, Any]]) -> None:
    ETC.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(dump(cfg))
    try:
        os.chmod(CONFIG_PATH, 0o644)
    except OSError:
        pass


def set_key(cfg: dict, dotted: str, raw: str) -> Any:
    """Set `section.key` from a CLI string, coercing to the default's type. Returns the value."""
    if not re.fullmatch(r"[a-z_]+\.[a-z_]+", dotted):
        raise ValueError("use section.key, e.g. recognition.certainty")
    section, key = dotted.split(".")
    if section not in DEFAULTS or key not in DEFAULTS[section]:
        raise ValueError(f"unknown key {dotted}")
    ref = DEFAULTS[section][key]
    if isinstance(ref, bool):
        if raw.lower() not in ("true", "false", "1", "0", "yes", "no"):
            raise ValueError("expected true/false")
        val: Any = raw.lower() in ("true", "1", "yes")
    elif isinstance(ref, int):
        val = int(raw)
    elif isinstance(ref, float):
        val = float(raw)
    else:
        val = raw
    cfg[section][key] = val
    return val


def camera_paths(cfg: dict) -> tuple[dict[str, str], list[str]]:
    """Resolve the configured cameras to today's /dev paths. Returns ({role: path}, notes)."""
    from .. import cameras
    cams = cameras.list_cameras()
    out, notes = {}, []
    for role in ("ir", "rgb"):
        path, note = cameras.resolve(cfg["cameras"][role], role, cams, force=bool(cfg["cameras"].get("force_roles")))
        if path:
            out[role] = path
        if note:
            notes.append(note)
    return out, notes
