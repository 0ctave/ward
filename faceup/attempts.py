"""The attempt log as data (roadmap U5): parse `faceup.log` lines, summarise, follow.

One line per attempt is written by engine.auth.Outcome.logline():
    2026-09-05T14:19:34 user=octave result=accept ir=0.26 rgb=0.41 irw=17.51/5.65 votes=cross:PASS,motion:ABSTAIN learned=ir:C/up frames=2 t=1604ms detail='via sudo'
"""
from __future__ import annotations

import re
import shlex
import statistics
import time
from pathlib import Path

_KV = re.compile(r"(\w+)=('(?:[^'\\]|\\.)*'|\S+)")


def parse(line: str) -> dict | None:
    """One log line → dict with typed fields, or None for a line that is not an attempt."""
    line = line.strip()
    if not line or " user=" not in line or " result=" not in line:
        return None
    ts, _, rest = line.partition(" ")
    row: dict = {"time": ts}
    for k, v in _KV.findall(rest):
        if v.startswith("'"):
            try:
                v = shlex.split(v)[0]
            except ValueError:
                v = v.strip("'")
        row[k] = v
    row["accepted"] = row.get("result") == "accept"
    row["reason"] = row.get("result", "?").split(":")[0]
    for k in ("ir", "rgb"):
        if k in row:
            try:
                row[k] = float(row[k])
            except ValueError:
                row[k] = None
    m = re.fullmatch(r"(\d+)ms", str(row.get("t", "")))
    row["ms"] = int(m.group(1)) if m else None
    try:
        row["frames"] = int(row.get("frames", 0))
    except ValueError:
        row["frames"] = 0
    row["votes"] = row.get("votes", "").split(",") if row.get("votes") else []
    d = str(row.get("detail", ""))
    row["service"] = d[4:].split(";")[0].strip() if d.startswith("via ") else ""
    row["dark"] = "ir-only (dark)" in d                  # ran on IR alone because the room was dark
    return row


def read(path: Path, n: int | None = None) -> list[dict]:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    rows = [r for r in (parse(ln) for ln in lines) if r]
    return rows[-n:] if n else rows


def summary(rows: list[dict]) -> dict:
    """Accept rate, latency percentiles, reasons, per-service and per-day counts."""
    out: dict = {"attempts": len(rows), "accepted": sum(r["accepted"] for r in rows)}
    out["accept_rate"] = out["accepted"] / len(rows) if rows else 0.0
    ms = sorted(r["ms"] for r in rows if r["ms"] is not None)
    out["median_ms"] = int(statistics.median(ms)) if ms else None
    out["p90_ms"] = int(ms[min(len(ms) - 1, int(0.9 * len(ms)))]) if ms else None
    acc_ms = sorted(r["ms"] for r in rows if r["accepted"] and r["ms"] is not None)
    out["median_accept_ms"] = int(statistics.median(acc_ms)) if acc_ms else None
    reasons: dict[str, int] = {}
    for r in rows:
        if not r["accepted"]:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    out["reasons"] = dict(sorted(reasons.items(), key=lambda kv: -kv[1]))
    services: dict[str, dict] = {}
    for r in rows:
        s = services.setdefault(r["service"] or "-", {"attempts": 0, "accepted": 0})
        s["attempts"] += 1; s["accepted"] += int(r["accepted"])
    out["services"] = services
    ir = [r["ir"] for r in rows if r["accepted"] and r.get("ir") is not None]
    out["ir_median"] = round(statistics.median(ir), 2) if ir else None
    out["learned"] = sum(1 for r in rows if r.get("learned"))
    out["spoof"] = sum(1 for r in rows if r["reason"] == "spoof")
    return out


def follow(path: Path, poll: float = 0.5):
    """Yield new attempt rows as they are appended (Ctrl+C to stop)."""
    with open(path) as f:
        f.seek(0, 2)
        while True:
            ln = f.readline()
            if not ln:
                time.sleep(poll); continue
            r = parse(ln)
            if r:
                yield r
