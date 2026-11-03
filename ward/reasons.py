"""Canonical human text for an attempt's failure reason.

Shared by the manager app's Reliability page (`app_window._reliability`) and the CLI's
`ward log --summary` so the two can never drift — a new failure reason gets its label and tip here,
once. `feedback.RESULT` holds the *during-attempt* sentences (a different register, shown under the
password field at the moment of the attempt); keep a new reason in step across both.
"""
from __future__ import annotations

# short label for a failure-reason code as recorded in the log (attempts.read → row["reason"])
LABEL = {
    "no-match": "face not recognised",
    "liveness": "could not prove a live face",
    "spoof": "looked like a photo or screen",
    "no-face": "no face seen",
    "dark": "too dark",
    "timeout": "took too long",
    "lockout": "locked out after spoof attempts",
    "camera": "camera unavailable",
    "policy": "face unlock off here",
    "no-models": "no face registered",
    "store": "could not read the face store",
}

# what the user can do about it ("" = nothing actionable)
TIP = {
    "no-match": "register again in this light",
    "liveness": "move your head slightly during the attempt",
    "spoof": "use your real face, not a photo or screen",
    "no-face": "face the camera",
    "dark": "more light, or an IR camera",
    "timeout": "check the camera isn't busy",
    "no-models": "register your face first",
}


def label(reason: str) -> str:
    """The human label for a reason code, or the code itself if unknown."""
    return LABEL.get(reason, reason)


def tip(reason: str) -> str:
    """What the user can do about a reason, or '' if nothing actionable."""
    return TIP.get(reason, "")
