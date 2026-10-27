"""Configuration: /etc/ward/config.toml with defaults (DESIGN.md §5.1).

Paths can be redirected with WARD_ETC / WARD_LOG for tests and unprivileged runs.
"""
from __future__ import annotations

import copy
import os
import re
try:
    import tomllib                      # Python ≥ 3.11
except ImportError:                     # 3.10: parse the flat subset ward itself writes
    tomllib = None
from pathlib import Path
from typing import Any

ETC = Path(os.environ.get("WARD_ETC", "/etc/ward"))
CONFIG_PATH = ETC / "config.toml"
MODELS_DIR = ETC / "models"
LOG_PATH = Path(os.environ.get("WARD_LOG", "/var/log/ward.log"))
# bundled (Apache-2.0, 1.7 MB): MiniFASNetV2 — see ward/models/NOTICE and liveness/antispoof_cnn.py
ANTISPOOF_MODEL = Path(__file__).resolve().parents[1] / "models" / "minifasnet_v2.onnx"

DEFAULTS: dict[str, dict[str, Any]] = {
    "cameras": {"ir": "", "rgb": "", "ir_dark_threshold": 25, "rgb_dark_threshold": 40, "force_roles": False},
    "recognition": {
        "certainty": 0.40, "rgb_certainty": 0.38, "fusion_frames": 4,
        "require_rgb_match": False, "detector": "yunet", "max_height": 320, "accel": "cpu", "recogniser": "dlib",
        "whitened": False, "hard_cap": 0.55, "early_reject": 0.65,
    },
    "liveness": {
        "required_votes": 2, "cross_spectrum": True, "antispoof_cnn": True,
        "shadow_test": "auto", "shadow_toggle_ms": -1, "motion": True, "motion_challenge": False,
        "require_liveness_when_available": True,
    },
    "timing": {"settle_ms": 0, "window_ms": 2500, "hard_timeout_ms": 4000},
    "emitter": {"unit": 0, "selector": 0, "on": "", "off": ""},
    "behaviour": {"abort_if_ssh": True, "tty_login": False,
                  "learn": True, "learn_max_dist": 0.30, "learn_min_novelty": 0.25, "learn_per_bin": 2, "learn_max_total": 12,
                  "notify": "accept", "skip_login": False},
    "security": {"policy": "balanced", "policy_sudo": "", "policy_polkit": "", "policy_login": "", "policy_unlock": "",
                 "lockout_failures": 0, "lockout_minutes": 15},
}

_COMMENTS = {
    "cameras": "Stable ids usb:<vid>:<pid>:<iface> (setup writes them), 'auto', or a /dev path; '' = none.",
    "recognition": "recogniser = dlib (distro) | arcface (`ward recogniser install`; tighter identity clusters). certainty = max raw distance (dlib scale for every recogniser; 0.6 is 'same person'). whitened (off by default — measured never to fire: live whitened distances 28–37 vs a threshold ~15, and it rescued no night frame): also accept raw <= hard_cap when the distance whitened by the user's own variation is under its learned threshold. early_reject: end the window as soon as fusion_frames+2 frames all sit above this distance (0 = off).",
    "liveness": "Voters return PASS/FAIL/ABSTAIN; `required_votes` PASSes and zero FAILs are needed. shadow_test = auto runs the emitter shadow test once `ward emitter toggle-bench` has written shadow_toggle_ms (<= 150).",
    "timing": "Milliseconds. settle_ms > 0 waits for auto-exposure before the window (not needed with YuNet).",
    "security": "policy = convenience | balanced | strict (`ward tune policy X`); policy_<service> overrides it for sudo / polkit / login / unlock ('' = default, 'off' = disabled there). lockout_failures liveness FAILs within lockout_minutes pause face unlock (password unaffected); 0 = never.",
    "emitter": "The IR LED control found by `ward emitter probe` (UVC extension unit; on/off = comma-separated bytes). unit 0 = use linux-enable-ir-emitter's instruction instead.",
    "behaviour": "learn: after a confident accept (raw <= learn_max_dist, liveness passed) store the live embedding as a learned model when it is novel (>= learn_min_novelty from existing ones), max learn_per_bin per pose bin and learn_max_total per spectrum; distance is measured against enrolled models only, and a single-frame accept never learns. notify: desktop notification after an attempt — accept | all | off. skip_login: leave the FIRST login of a session to the password; tty_login: allow face unlock on text consoles (off by default).",
}


def load() -> dict[str, dict[str, Any]]:
    cfg = copy.deepcopy(DEFAULTS)
    try:
        text = CONFIG_PATH.read_text()
    except FileNotFoundError:
        return cfg
    data = tomllib.loads(text) if tomllib is not None else parse_flat_toml(text)
    for section, values in data.items():
        if section in cfg and isinstance(values, dict):
            cfg[section].update(values)
    return cfg


def parse_flat_toml(text: str) -> dict:
    """The subset `dump()` writes: [section] headers and `key = value` with quoted strings,
    ints, floats and true/false. Comments and blank lines are skipped. Enough for 3.10."""
    out: dict[str, dict] = {}
    section = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip(); out.setdefault(section, {}); continue
        if section is None or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if "#" in v and not v.startswith('"'):
            v = v.split("#", 1)[0].strip()
        if v.startswith('"') and v.endswith('"'):
            val: Any = v[1:-1].replace('\\"', '"')
        elif v in ("true", "false"):
            val = v == "true"
        else:
            try:
                val = int(v)
            except ValueError:
                try:
                    val = float(v)
                except ValueError:
                    val = v
        out[section][k] = val
    return out


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return '"' + v.replace('"', '\\"') + '"'
    return str(v)


def dump(cfg: dict[str, dict[str, Any]]) -> str:
    out = ["# ward configuration — see docs/DESIGN.md §5.1", ""]
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
