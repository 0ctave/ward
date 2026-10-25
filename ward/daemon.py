"""wardd — the resident engine (roadmap R4): Python, OpenCV, dlib and the face models stay loaded.

A cold `ward auth` spends ~0.7 s starting Python and importing numpy/cv2/dlib, then 0.3–2.5 s
loading the models (power-saver doubles both) before a camera frame can be used. The daemon pays
that once at boot; an attempt then costs only the camera open and the recognition itself.

Protocol (one connection per attempt, newline-delimited JSON on a root-only Unix socket):
    client → {"user": "octave", "env": {"PAM_SERVICE": "sudo", …}, "verbose": false}
    daemon → {"event": "face"} / {"event": "hold", "still": true} / {"say": "…"}  (0..n)
    daemon → {"outcome": {…Outcome fields…}}                                       (last)
The client (ward/client.py) owns the terminal and the PAM conversation, so feedback stays
where it was; the daemon logs the attempt exactly as an in-process run would.

Safety: the socket lives in /run/ward (0700 root) and is 0600 root, so only root — PAM — can
ask for an attempt; the user name is the client's to give, as for `ward-auth`. One attempt runs
at a time (the cameras are exclusive). Anything wrong with the daemon makes the client fall back
to the in-process engine: the daemon is a speed-up, never a dependency (golden rule 3).

systemd (data/ward.service, Type=notify): READY=1 once the models are warm, then WATCHDOG=1
from a thread every WatchdogSec/2 — which stops pinging while an attempt is wedged past its
deadline, so systemd kills and restarts us even when no new request arrives to notice. Plain
stdlib datagrams on $NOTIFY_SOCKET (no libsystemd); without it every notify is a no-op.
"""
from __future__ import annotations

import json
import os
import re
import select
import socket
import threading
import time

SOCK_DIR = os.environ.get("WARD_RUN", "/run/ward")
SOCK = os.path.join(SOCK_DIR, "engine.sock")
USER_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.@-]{0,63}\$?")
BUSY_WAIT_S = 1.0          # a queued request waits this long for the cameras, then the client falls back
STUCK_MARGIN_S = 15.0      # an attempt this far past its hard timeout is wedged: exit, systemd restarts us
SEND_TIMEOUT_S = 5.0       # a client that stops reading drops events instead of blocking the attempt
PAM_ENV = ("PAM_SERVICE", "PAM_TTY", "PAM_TYPE", "PAM_RHOST", "PAM_RUSER", "PAM_USER", "SSH_CONNECTION", "DISPLAY", "WAYLAND_DISPLAY", "XDG_SESSION_ID")


def sd_notify(state: str, env=None) -> bool:
    """Send `state` ("READY=1", "WATCHDOG=1", …) to systemd's $NOTIFY_SOCKET. False (never an
    exception) when not run by systemd or the send fails. "@name" is an abstract socket."""
    env = os.environ if env is None else env
    addr = env.get("NOTIFY_SOCKET", "")
    if not addr or addr[0] not in "/@":
        return False
    if addr[0] == "@":
        addr = "\0" + addr[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM | socket.SOCK_CLOEXEC) as s:
            s.connect(addr)
            s.sendall(state.encode())
        return True
    except OSError:
        return False


def watchdog_interval(env=None) -> float | None:
    """Seconds between WATCHDOG=1 pings (half of WatchdogSec), or None when no watchdog is armed
    for this process ($WATCHDOG_USEC absent/invalid, or $WATCHDOG_PID names another process)."""
    env = os.environ if env is None else env
    pid = env.get("WATCHDOG_PID")
    if pid and pid != str(os.getpid()):
        return None
    try:
        usec = int(env.get("WATCHDOG_USEC", ""))
    except ValueError:
        return None
    return usec / 2e6 if usec > 0 else None


def watchdog_loop(res: "Resident", interval: float, stop: threading.Event | None = None, notify=sd_notify) -> None:
    """Ping the watchdog while the daemon is healthy; a wedged attempt silences it."""
    stop = stop or threading.Event()
    while not stop.is_set():
        if not res.wedged():
            notify("WATCHDOG=1")
        stop.wait(interval)


class Resident:
    """Holds the loaded models; builds a fresh Engine (fresh config) for every attempt."""

    def __init__(self):
        from .engine import config as cfgmod
        self.cfgmod = cfgmod
        self.models = None
        self.key = None
        self.lock = threading.Lock()
        self.started = 0.0          # monotonic start of the running attempt (0 = idle)
        self.budget_s = 0.0         # its hard timeout
        self.warm(cfgmod.load())

    def wedged(self, now: float | None = None) -> bool:
        """True while the running attempt is STUCK_MARGIN_S past its hard timeout."""
        started = self.started
        now = time.monotonic() if now is None else now
        return bool(started) and now - started > self.budget_s + STUCK_MARGIN_S

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

    def attempt(self, req: dict, send, alive=lambda: True) -> None:
        from .engine.auth import Engine
        user = req.get("user")
        if not isinstance(user, str) or not USER_RE.fullmatch(user):
            send({"error": "invalid user"}); return            # before the cameras, so the client falls back
        cfg = self.cfgmod.load()
        env = {k: v for k, v in (req.get("env") or {}).items() if k in PAM_ENV and isinstance(v, str)}
        saved = {k: os.environ.get(k) for k in PAM_ENV}
        if not self.lock.acquire(timeout=BUSY_WAIT_S):
            if self.wedged():
                print("wardd: an attempt is wedged (camera driver?): exiting for a restart", flush=True)
                os._exit(1)                                    # Restart=on-failure brings up a fresh daemon
            send({"error": "busy"}); return
        try:
            if not alive():
                return                                         # the client gave up while queued: no phantom attempt
            try:
                for k in PAM_ENV:
                    if k in env:
                        os.environ[k] = env[k]
                    else:
                        os.environ.pop(k, None)
                self.warm(cfg)                                 # a reload (ArcFace r50: 281 MB) is not the attempt's budget
                self.budget_s = float(cfg.get("timing", {}).get("hard_timeout_ms", 8000)) / 1000
                self.started = time.monotonic()                # after the budget: wedged() never sees a stale one
                eng = Engine(cfg, models=self.models, verbose=bool(req.get("verbose")))
                eng.events = lambda kind, **kw: send({"event": kind, **kw})
                if req.get("verbose"):
                    eng.say = lambda msg: send({"say": msg})
                out = eng.authenticate(user)
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
        finally:
            self.started = 0.0
            self.lock.release()
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
    ms = int((time.monotonic() - t0) * 1000)
    print(f"wardd: ready in {ms} ms on {path}", flush=True)
    sd_notify(f"READY=1\nSTATUS=ready in {ms} ms")
    iv = watchdog_interval()
    if iv:
        threading.Thread(target=watchdog_loop, args=(res, iv), daemon=True, name="watchdog").start()
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
            conn.settimeout(SEND_TIMEOUT_S)
            if req.get("ping"):
                f.write(b'{"pong": true}\n'); return
            dead = []

            def send(obj):
                if dead:
                    return
                try:
                    f.write((json.dumps(obj, default=str) + "\n").encode())
                except OSError:                                # incl. timeout: a stalled reader
                    dead.append(1)                             # the client left: finish and log anyway
            res.attempt(req, send, alive=lambda: peer_alive(conn))
        except Exception as e:                                 # one bad request never stops the daemon
            try:
                conn.sendall((json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n").encode())
            except OSError:
                pass


def peer_alive(conn: socket.socket) -> bool:
    """False once the client hung up (EOF or error pending); never blocks."""
    try:
        r, _, x = select.select([conn], [], [conn], 0)
        if x:
            return False
        if r:
            return conn.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) != b""
    except (OSError, ValueError):
        return False
    return True
