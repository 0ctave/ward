"""ward command line — see docs/DESIGN.md §11.

One module per command group; this package re-exports every command so
`from ward import cli`, `cli.main` and `from ward.cli import daemon_state` keep working.
"""
from __future__ import annotations

import argparse
import json  # noqa: F401  (module attributes the single-file cli exposed)
import os  # noqa: F401
import subprocess  # noqa: F401
import sys
import time  # noqa: F401

from .. import __version__, cameras, howdy, probe, ui  # noqa: F401
from ..engine import config as cfgmod  # noqa: F401
from ..engine import pam as pammod  # noqa: F401
from ..engine import store as storemod  # noqa: F401
from .common import (  # noqa: F401
    EMITTER_REPO, OPTIONAL_POSES, POSES, need_root, real_user,
)
from .doctor import (  # noqa: F401
    cmd_update,
    cmd_bench, cmd_doctor, cmd_tune, recommend_cnn,
)
from .enroll import (  # noqa: F401
    _display_available, _finish_calibration, _report_calibration, _stream_ui, calibrate_pose, cmd_calibrate, cmd_enroll, report_pose_calibration, run_calibration_session, selfcheck,
)
from .auth import (  # noqa: F401
    UNIT, _cmd_pam, cmd_auth, cmd_daemon, cmd_log, cmd_pam, daemon_state, format_attempt,
)
from .manage import (  # noqa: F401
    _recogniser_name, cmd_antispoof, cmd_app, cmd_keyring, cmd_lab, cmd_models, cmd_recogniser,
)
from .setup import (  # noqa: F401
    antispoof_available, cmd_emitter, cmd_setup, emitter_configure, emitter_install,
)


BENCHMARKS = {"lfw": "lfw", "spoof": "celeba_spoof", "ir": "ir_pad", "redteam": "redteam"}


def cmd_benchmark(args) -> int:
    """Forward to a harness module's own argparse (ward/{lfw,celeba_spoof,ir_pad,redteam}.py)."""
    import importlib
    mod = importlib.import_module(f"ward.{BENCHMARKS[args.bench_cmd]}")
    return mod.main(args.rest)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        argv = ["app"]          # bare `ward` / the desktop launcher opens the manager window
    # `benchmark <name> …` hands the rest straight to that harness's own argparse (including --help),
    # which REMAINDER-based subparsers mis-route; the parser below still lists it for discoverability.
    if argv and argv[0] == "benchmark" and len(argv) > 1 and argv[1] in BENCHMARKS:
        import importlib
        return importlib.import_module(f"ward.{BENCHMARKS[argv[1]]}").main(argv[2:])
    p = argparse.ArgumentParser(prog="ward", description="Face unlock for Linux that actually works: dual-spectrum recognition with liveness.")
    p.add_argument("--version", action="version", version=f"ward {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="diagnose cameras, emitter, engine, models, PAM"); d.add_argument("-U", "--user"); d.set_defaults(fn=cmd_doctor)
    up = sub.add_parser("update", help="check GitHub for a newer release (explicit; never runs during an unlock)"); up.set_defaults(fn=cmd_update)
    s = sub.add_parser("setup", help="the whole chain: cameras → emitter → recogniser → register → keyring → PAM → policy (root)")
    s.add_argument("-U", "--user"); s.add_argument("--cameras-only", action="store_true", help="detect and write the camera pair, nothing else"); s.set_defaults(fn=cmd_setup)
    ap = sub.add_parser("app", help="the GTK application — every step from one window (sudo for setup actions)")
    ap.add_argument("--selftest", action="store_true", help="render every page once and exit (CI smoke test)"); ap.set_defaults(fn=cmd_app)

    e = sub.add_parser("enroll", help="pose-guided, dual-spectrum enrollment (root)")
    e.add_argument("-U", "--user"); e.add_argument("--fresh", action="store_true", help="delete existing models first")
    e.add_argument("--all", action="store_true", help="include optional poses without asking")
    e.add_argument("--selfcheck", action="store_true", help="run the self-check without asking")
    e.add_argument("--guided", action="store_true", help="scripted 8-pose text flow instead of the window")
    e.add_argument("--classic", action="store_true", help="old free-move coverage-map registration + a separate calibration step")
    e.add_argument("--style", choices=["guided", "ring"], default="guided",
                   help="window enrol style: 'guided' sweep (default) or 'ring' (segmented coverage ring, fill the circle)")
    e.add_argument("--identity", metavar="NAME", help="register another person on this login under NAME (any registered face unlocks)")
    e.add_argument("--add", metavar="CONDITION", help="short sweep that adds a condition to the existing models: glasses, headset, night, …")
    e.add_argument("--profile", action="store_true", help="print per-stage timings once a second (diagnose a laggy preview)")
    e.add_argument("--stream-ui", action="store_true", help=argparse.SUPPRESS)   # used by `ward app`
    e.set_defaults(fn=cmd_enroll)

    t = sub.add_parser("tune", help="show config, or set section.key VALUE (root)")
    t.add_argument("key", nargs="?"); t.add_argument("value", nargs="?"); t.set_defaults(fn=cmd_tune)

    b = sub.add_parser("bench", help="time HOG vs CNN detectors on a live frame"); b.add_argument("--device"); b.set_defaults(fn=cmd_bench)

    bm = sub.add_parser("benchmark", help="offline accuracy on external datasets: recogniser (lfw), anti-spoof (spoof), IR liveness (ir), red-team your own frames (redteam)")
    bms = bm.add_subparsers(dest="bench_cmd", required=True)
    for name, h in (("lfw", "impostor/genuine distances on LFW (recogniser; `-m ward.lfw`)"),
                    ("spoof", "APCER/BPCER on CelebA-Spoof (anti-spoof CNN; `-m ward.celeba_spoof`)"),
                    ("ir", "cross-spectrum PAD on CASIA-SURF/WMCA (`-m ward.ir_pad`)"),
                    ("redteam", "digitally attack your own recorded frames (`-m ward.redteam`)")):
        sp = bms.add_parser(name, help=h, add_help=False)      # options are parsed by the module itself
        sp.add_argument("rest", nargs=argparse.REMAINDER)
    bm.set_defaults(fn=cmd_benchmark)

    a = sub.add_parser("auth", help="run the authentication engine once (what PAM calls); exit 0 = accepted")
    a.add_argument("-U", "--user"); a.add_argument("-v", "--verbose", action="store_true")
    a.add_argument("--replay", metavar="DIR", help="authenticate against recorded frame sets instead of the cameras"); a.set_defaults(fn=cmd_auth)

    pm = sub.add_parser("pam", help="enable | disable | status the PAM integration (root)")
    pm.add_argument("action", choices=["enable", "disable", "status"]); pm.add_argument("--dry-run", action="store_true"); pm.set_defaults(fn=cmd_pam)

    lb = sub.add_parser("lab", help="live evaluation window: compare unlock policies, record subjects, summarise (root)")
    lb.add_argument("--record-frames", metavar="DIR", help="save every frame set as PNGs for replay (roadmap L7)")
    lb.add_argument("-U", "--user"); lb.add_argument("--text", action="store_true", help="terminal mode instead of the window")
    lb.add_argument("--subject", help="text mode: record samples under this subject name")
    lb.set_defaults(fn=cmd_lab)

    kr = sub.add_parser("keyring", help="status | enable | disable | repair — let a face login unlock GNOME Keyring / KWallet (root except status)")
    kr.add_argument("action", choices=["status", "enable", "disable", "repair"]); kr.add_argument("-U", "--user")
    kr.add_argument("--force-host-key", action="store_true"); kr.add_argument("--tpm", action="store_true", help="force TPM2 sealing (stronger, but adds 2–4 s per login on a firmware TPM)")
    kr.add_argument("--password-stdin", action="store_true", help="read the login password from stdin (used by the app)")
    kr.add_argument("--dry-run", action="store_true"); kr.set_defaults(fn=cmd_keyring)

    asp = sub.add_parser("antispoof", help="status | on | off | test DIR — the bundled anti-spoof CNN liveness voter (root for on/off)")
    asp.add_argument("action", choices=["status", "on", "off", "test"]); asp.add_argument("dir", nargs="?", default="")
    asp.set_defaults(fn=cmd_antispoof)

    rg = sub.add_parser("recogniser", help="status | install [mbf|r50] | use dlib|arcface — the embedding network (root for install/use)")
    rg.add_argument("action", choices=["status", "install", "use"]); rg.add_argument("variant", nargs="?", default="mbf")
    rg.add_argument("-y", "--yes", action="store_true"); rg.set_defaults(fn=cmd_recogniser)

    lg = sub.add_parser("log", help="recent authentication attempts; --summary for rates and latency, --follow to watch")
    lg.add_argument("-n", type=int, default=20); lg.add_argument("--summary", action="store_true"); lg.add_argument("--follow", "-f", action="store_true")
    lg.add_argument("--json", action="store_true"); lg.set_defaults(fn=cmd_log)

    cb = sub.add_parser("calibrate", help="session (default): measure your real distances and set the gate + whitening threshold · refresh: from recent accepts · pose: rebuild the 3D face model (root)")
    cb.add_argument("action", nargs="?", choices=["session", "refresh", "pose"], default="session")
    cb.add_argument("-U", "--user"); cb.add_argument("--seconds", type=float, default=None, help="max seconds per pass")
    cb.add_argument("--text", action="store_true", help="text prompts instead of the window")
    cb.add_argument("--stream-ui", action="store_true", help=argparse.SUPPRESS); cb.set_defaults(fn=cmd_calibrate)

    dm = sub.add_parser("daemon", help="run | status | enable | disable — the resident engine that keeps the models loaded (fast unlock)")
    dm.add_argument("action", choices=["run", "status", "enable", "disable"]); dm.set_defaults(fn=cmd_daemon)

    m = sub.add_parser("models", help="list or remove registered models (root)")
    m.add_argument("-U", "--user"); m.add_argument("--remove", metavar="LABEL")
    m.add_argument("--forget-learned", action="store_true", help="drop models added by online adaptation")
    m.add_argument("--remove-identity", metavar="NAME"); m.add_argument("--remove-condition", metavar="CONDITION")
    m.add_argument("--identity", metavar="NAME", help="with --remove-condition / --forget-learned: only this person's (default: everyone's)")
    m.add_argument("--json", action="store_true", help="people on this login as JSON (used by the app)")
    m.set_defaults(fn=cmd_models)

    em = sub.add_parser("emitter", help="IR emitter: probe (find the LED control by measurement) | toggle-bench (switch speed → shadow test) | check | test | install | configure")
    em.add_argument("action", choices=["check", "test", "install", "configure", "probe", "toggle-bench"]); em.set_defaults(fn=cmd_emitter)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        print(); return 130


if __name__ == "__main__":
    sys.exit(main())
