"""faceupd — the resident engine (roadmap R4): Python, OpenCV, dlib and the face models stay loaded.

A cold `faceup auth` spends ~0.7 s starting Python and importing numpy/cv2/dlib, then 0.3–2.5 s
loading the models (power-saver doubles both) before a camera frame can be used. The daemon pays
that once at boot; an attempt then costs only the camera open and the recognition itself.

Protocol (one connection per attempt, newline-delimited JSON on a root-only Unix socket):
    client → {"user": "octave", "env": {"PAM_SERVICE": "sudo", …}, "verbose": false}
    daemon → {"event": "face"} / {"event": "hold", "still": true} / {"say": "…"}  (0..n)
    daemon → {"outcome": {…Outcome fields…}}                                       (last)
The client (faceup/client.py) owns the terminal and the PAM conversation, so feedback stays
where it was; the daemon logs the attempt exactly as an in-process run would.

Safety: the socket lives in /run/faceup (0700 root) and is 0600 root, so only root — PAM — can
ask for an attempt; the user name is the client's to give, as for `faceup-auth`. One attempt runs
at a time (the cameras are exclusive). Anything wrong with the daemon makes the client fall back
to the in-process engine: the daemon is a speed-up, never a dependency (golden rule 3).
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time

SOCK_DIR = os.environ.get("FACEUP_RUN", "/run/faceup")
SOCK = os.path.join(SOCK_DIR, "engine.sock")
PAM_ENV = ("PAM_SERVICE", "PAM_TTY", "PAM_TYPE", "PAM_RHOST", "PAM_RUSER", "PAM_USER", "SSH_CONNECTION", "DISPLAY", "WAYLAND_DISPLAY", "XDG_SESSION_ID")


class Resident:
    """Holds the loaded models; builds a fresh Engine (fresh config) for every attempt."""

    def __init__(self):
        from .engine import config as cfgmod
        self.cfgmod = cfgmod
        self.models = None
        self.key = None
        self.lock = threading.Lock()
        self.warm(cfgmod.load())

    def _models_key(self, cfg: dict) -> tuple:
        rec = cfg["recognition"]
        return (rec["detector"], rec["max_height"], rec.get("accel", "cpu"), rec.get("recogniser", "dlib"))

    def warm(self, cfg: dict) -> None:
        """Load (or reload after `recogniser use` / `tune recognition.detector`) and run each network
        once: the first YuNet inference and the first embedding allocate their buffers (~50–150 ms)."""
        key = self._models_key(cfg)
        if key == self.key and self.models is not None:
            return
        import numpy as np
        from .engine.detect import Models
        m = Models(*key)
        m.preload()
        try:
            m.detect(np.zeros((360, 640), np.uint8))
        except Exception:
            pass
        self.models, self.key = m, key

    def attempt(self, req: dict, send) -> None:
        from .engine.auth import Engine
        cfg = self.cfgmod.load()
        env = {k: v for k, v in (req.get("env") or {}).items() if k in PAM_ENV and isinstance(v, str)}
        saved = {k: os.environ.get(k) for k in PAM_ENV}
        with self.lock:
            try:
                for k in PAM_ENV:
                    if k in env:
                        os.environ[k] = env[k]
                    else:
                        os.environ.pop(k, None)
                self.warm(cfg)
                eng = Engine(cfg, models=self.models, verbose=bool(req.get("verbose")))
                eng.events = lambda kind, **kw: send({"event": kind, **kw})
                if req.get("verbose"):
                    eng.say = lambda msg: send({"say": msg})
                out = eng.authenticate(str(req["user"]))
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
        send({"outcome": outcome_dict(out)})


def outcome_dict(out) -> dict:
    return {"accepted": out.accepted, "reason": out.reason, "user": out.user, "ir_dist": out.ir_dist, "rgb_dist": out.rgb_dist,
            "identity": out.identity, "condition": out.condition, "elapsed_ms": out.elapsed_ms, "frames": out.frames,
            "detail": out.detail, "phases": out.phases, "logline": out.logline(),
            "votes": [{"voter": v.voter, "result": v.result.value} for v in out.votes]}


def serve(path: str = SOCK) -> int:
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    os.chmod(os.path.dirname(path), 0o700)
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    t0 = time.monotonic()
    res = Resident()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    old = os.umask(0o177)
    try:
        srv.bind(path)
    finally:
        os.umask(old)
    srv.listen(4)
    print(f"faceupd: ready in {int((time.monotonic() - t0) * 1000)} ms on {path}", flush=True)
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=handle, args=(res, conn), daemon=True).start()


def handle(res: Resident, conn: socket.socket) -> None:
    with conn:
        try:
            import struct
            _pid, uid, _gid = struct.unpack("3i", conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            if uid != os.geteuid():
                return                                         # only our own uid — root, i.e. PAM — may ask
            conn.settimeout(2.0)
            f = conn.makefile("rwb", buffering=0)
            req = json.loads(f.readline().decode() or "{}")
            conn.settimeout(None)
            if req.get("ping"):
                f.write(b'{"pong": true}\n'); return

            def send(obj):
                try:
                    f.write((json.dumps(obj, default=str) + "\n").encode())
                except OSError:
                    pass                                       # the client left: finish and log anyway
            res.attempt(req, send)
        except Exception as e:                                 # one bad request never stops the daemon
            try:
                conn.sendall((json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n").encode())
            except OSError:
                pass
