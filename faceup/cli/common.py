"""Shared CLI helpers: root check, invoking user, the scripted pose lists."""
from __future__ import annotations

import os
import sys

from .. import probe, ui

EMITTER_REPO = "EmixamPP/linux-enable-ir-emitter"

POSES = [
    ("straight",   "Look straight at the camera, neutral face."),
    ("keyboard",   "Look DOWN at the keyboard — where your face is when a sudo prompt appears."),
    ("tilt-left",  "Tilt your head ~15° to the left (ear toward shoulder)."),
    ("tilt-right", "Tilt your head ~15° to the right."),
    ("turn-left",  "Turn your head ~20° to the left, eyes on the screen."),
    ("turn-right", "Turn your head ~20° to the right."),
    ("lean-back",  "Sit back further from the screen, relaxed posture."),
    ("close",      "Lean in close, like reading small text."),
]
OPTIONAL_POSES = [
    ("glasses", "Wearing your glasses (or without them, if you registered with them on)."),
    ("headset", "Wearing headphones / a headset."),
    ("dark",    "Lights OFF — verifies the IR emitter is doing the work."),
]


def need_root(what: str) -> None:
    if not probe.is_root() and "FACEUP_ETC" not in os.environ:
        ui.fail(f"{what} needs root (config and models live under /etc/faceup).")
        ui.tip(f"run:  sudo faceup {' '.join(sys.argv[1:])}")
        sys.exit(2)


def real_user(args) -> str:
    from ..manage import invoking_user
    return getattr(args, "user", None) or os.environ.get("PAM_USER") or invoking_user()
