"""The data behind the faceup manager app: people on this login and how reliably face unlock works.

No GTK here, nothing prints. `people()` turns a model store plus the attempt log into one record
per identity; `reliability()` turns the log into rates, a verdict and concrete next steps (each
with the `faceup …` argv that performs it). `faceup models --json` emits `people()` for the app,
which runs unprivileged and cannot read the 0600 store itself.
"""
from __future__ import annotations

import os
import re
import statistics

NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,31}")


def invoking_user() -> str:
    """The human behind this process: sudo's or pkexec's caller, else the login user."""
    if os.environ.get("SUDO_USER"):
        return os.environ["SUDO_USER"]
    uid = os.environ.get("PKEXEC_UID")
    if uid:
        try:
            import pwd
            return pwd.getpwuid(int(uid)).pw_name
        except (KeyError, ValueError):
            pass
    return os.environ.get("USER") or "root"


def valid_name(name: str, owner: str = "") -> str:
    """'' when `name` can be a new identity, else why not (shown to the user)."""
    name = name.strip()
    if not NAME.fullmatch(name):
        return "use 1–32 letters, digits, spaces, dots, dashes or underscores, starting with a letter or digit"
    if owner and name == owner:
        return f"{owner} is the account owner — re-enrol instead"
    return ""


def _stats(rows: list[dict]) -> dict:
    acc = [r for r in rows if r["accepted"]]
    ms = [r["ms"] for r in acc if r.get("ms") is not None]
    return {"attempts": len(rows), "accepted": len(acc),
            "rate": len(acc) / len(rows) if rows else None,
            "median_ms": int(statistics.median(ms)) if ms else None}


def people(st, owner: str, rows: list[dict] | None = None) -> list[dict]:
    """One record per identity in the store (owner first): models per camera, conditions,
    enrolled date, learned count, and its accepts from the log (`who=` names the matched identity;
    accepts without it are the owner's)."""
    rows = rows or []
    ids = st.identities()
    if "" in ids:
        ids.remove(""); ids.insert(0, "")
    out = []
    for ident in ids:
        ms = [m for m in st.models if m.identity == ident]
        spectra: dict[str, int] = {}
        for m in ms:
            if not m.learned:
                spectra[m.spectrum] = spectra.get(m.spectrum, 0) + 1
        mine = [r for r in rows if r["accepted"] and (r.get("who") or "") == ident]
        times = sorted(m.time for m in ms if m.time and not m.learned)
        out.append({
            "identity": ident, "name": ident or owner, "owner": ident == "",
            "models": spectra, "learned": sum(1 for m in ms if m.learned),
            "conditions": st.conditions(ident), "enrolled": times[0][:10] if times else "",
            "accepts": len(mine), "last_seen": mine[-1]["time"] if mine else "",
        })
    return out


def reliability(rows: list[dict], n_people: int | None = None) -> dict:
    """Overall and split rates (day/night, per service, per identity), failure reasons, a verdict
    and next steps. Only the last 50 attempts count: old failures should not haunt a re-enrolment."""
    rows = rows[-50:]
    out: dict = {"overall": _stats(rows), "verdict": "unknown", "advice": []}
    out["day"] = _stats([r for r in rows if not r.get("dark")])
    out["night"] = _stats([r for r in rows if r.get("dark")])
    svc: dict[str, list] = {}
    for r in rows:
        svc.setdefault(r.get("service") or "terminal", []).append(r)
    out["services"] = {k: _stats(v) for k, v in sorted(svc.items())}
    who: dict[str, int] = {}
    for r in rows:
        if r["accepted"]:
            who[r.get("who") or ""] = who.get(r.get("who") or "", 0) + 1
    out["who"] = who
    reasons: dict[str, int] = {}
    for r in rows:
        if not r["accepted"]:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    out["reasons"] = dict(sorted(reasons.items(), key=lambda kv: -kv[1]))
    o = out["overall"]
    if o["attempts"] < 5:
        out["advice"].append(("Try a few unlocks (or Test unlock) — there is not enough data yet.", None))
        if n_people == 0:
            out["advice"].insert(0, ("Nobody is enrolled: register your face first.", ["enroll", "--fresh", "--selfcheck"]))
        return out
    rate, med = o["rate"], o["median_ms"]
    out["verdict"] = "good" if rate >= 0.9 and (med is None or med <= 2000) else "fair" if rate >= 0.7 else "poor"
    adv = out["advice"]
    if n_people == 0:
        adv.append(("Nobody is enrolled: register your face first.", ["enroll", "--fresh", "--selfcheck"]))
    nm = reasons.get("no-match", 0)
    if nm >= 3 and nm >= 0.15 * o["attempts"]:
        adv.append((f"{nm} attempts did not recognise the face — re-enrol in your usual light, or add a condition (glasses…) you often wear.",
                    ["enroll", "--fresh", "--selfcheck"]))
    n = out["night"]
    if n["attempts"] >= 3 and n["rate"] is not None and n["rate"] < 0.7 <= (out["day"]["rate"] or 0):
        adv.append(("Unlocking in the dark fails more than in daylight — check the IR emitter, then re-enrol.", ["emitter", "probe"]))
    lv = reasons.get("liveness", 0)
    if lv >= 3:
        adv.append((f"{lv} attempts recognised you but could not prove liveness — benchmark the IR emitter so the shadow test can vote.",
                    ["emitter", "toggle-bench"]))
    if reasons.get("spoof", 0):
        adv.append((f"{reasons['spoof']} attempts were flagged as a possible spoof (photo/screen). If that was you, recalibrate.", ["calibrate"]))
    if med is not None and med > 2000:
        adv.append((f"Median unlock takes {med / 1000:.1f} s — power-saver mode and a cold start are the usual causes.", None))
    if not adv and out["verdict"] == "good":
        adv.append(("Face unlock is working well.", None))
    return out
