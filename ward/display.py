"""Preview window backends.

`GtkDisplay` — native Wayland (GTK 4, GPU-composited via Gdk.MemoryTexture). Works as root:
it connects to the invoking user's Wayland socket, which sudo strips from the environment.
`Cv2Display` — OpenCV HighGUI (Qt/X11) fallback when PyGObject or a display is unavailable.

Both run a caller-supplied `step()` that returns the next RGB canvas (numpy HxWx3) or None
when finished, and deliver key presses through `keys()`. GTK owns the main thread, so
`step()` runs on a worker thread there; the OpenCV backend runs it inline.
"""
from __future__ import annotations

import os
import queue
import threading
from typing import Callable, Optional

import numpy as np

KEY_ENTER, KEY_ESC, KEY_Q = "enter", "esc", "q"
StepFn = Callable[[], Optional[np.ndarray]]


class DisplayUnavailable(Exception):
    pass


# ------------------------------------------------------------------ env ----
def user_wayland_env() -> dict[str, str]:
    """Environment that lets a root process open a window on the invoking user's session.
    sudo keeps DISPLAY but drops WAYLAND_DISPLAY and XDG_RUNTIME_DIR; GTK accepts an absolute
    socket path in WAYLAND_DISPLAY, so no runtime-dir juggling is needed."""
    env: dict[str, str] = {}
    if os.environ.get("WAYLAND_DISPLAY"):
        return env
    uid = os.environ.get("SUDO_UID") or os.environ.get("PKEXEC_UID")
    if not uid:
        return env
    import glob
    import pwd
    socks = sorted(p for p in glob.glob(f"/run/user/{uid}/wayland-*") if not p.endswith(".lock"))
    if socks:
        env["WAYLAND_DISPLAY"] = socks[0]
        env.setdefault("GDK_BACKEND", "wayland,x11")
    elif os.environ.get("DISPLAY") and not os.environ.get("XAUTHORITY"):
        # X11 session: sudo keeps DISPLAY but resets XAUTHORITY, so root is refused by the X server
        try:
            home = pwd.getpwuid(int(uid)).pw_dir
        except (KeyError, ValueError):
            home = ""
        cands = glob.glob(f"/run/user/{uid}/gdm/Xauthority") + glob.glob(f"/run/user/{uid}/xauth_*") \
            + glob.glob(f"/run/user/{uid}/.mutter-Xwaylandauth.*") + ([os.path.join(home, ".Xauthority")] if home else [])
        auth = next((c for c in cands if os.path.exists(c)), None)
        if auth:
            env["XAUTHORITY"] = auth
    # (the user's session bus refuses root, so GTK runs without one — its warning is silenced below)
    return env


# ------------------------------------------------------------------ GTK ----
class GtkDisplay:
    name = "gtk4-wayland"
    snapshot = False          # consumes pre-drawn canvases

    def __init__(self, title: str, width: int, height: int):
        for k, v in user_wayland_env().items():
            os.environ.setdefault(k, v)
        os.environ.setdefault("GSK_RENDERER", "ngl")     # GL renderer: texture uploads stay on the GPU
        if os.geteuid() == 0:
            # GTK4's accessibility backend is what tries to acquire a session bus ("Unable to
            # acquire session bus" as root, where there is none). A camera preview needs no AT-SPI.
            os.environ.setdefault("GTK_A11Y", "none")
        try:
            import gi
            gi.require_version("Gtk", "4.0"); gi.require_version("Gdk", "4.0")
            from gi.repository import Gdk, GLib, Gtk
        except (ImportError, ValueError) as e:
            raise DisplayUnavailable(f"PyGObject/GTK4 not available: {e}")
        self.Gdk, self.GLib, self.Gtk = Gdk, GLib, Gtk
        if not Gtk.init_check():
            raise DisplayUnavailable("GTK could not open a display (no Wayland/X11 session reachable)")
        self.title, self.w, self.h = title, width, height
        self._keys: queue.Queue = queue.Queue()
        self._pending = False
        self._loop = GLib.MainLoop()
        self.backend = "?"

    def keys(self) -> list[str]:
        out = []
        while True:
            try:
                out.append(self._keys.get_nowait())
            except queue.Empty:
                return out

    def _on_key(self, _ctl, keyval, _code, _state) -> bool:
        Gdk = self.Gdk
        if keyval in (Gdk.KEY_Return, Gdk.KEY_KP_Enter):
            self._keys.put(KEY_ENTER)
        elif keyval == Gdk.KEY_Escape:
            self._keys.put(KEY_ESC)
        elif keyval in (Gdk.KEY_q, Gdk.KEY_Q):
            self._keys.put(KEY_Q)
        return True

    def _present(self, canvas: np.ndarray) -> bool:
        Gdk, GLib = self.Gdk, self.GLib
        h, w = canvas.shape[:2]
        data = np.ascontiguousarray(canvas).tobytes()
        tex = Gdk.MemoryTexture.new(w, h, Gdk.MemoryFormat.R8G8B8, GLib.Bytes.new(data), w * 3)
        self.picture.set_paintable(tex)
        self._pending = False
        return False   # one-shot idle

    def run(self, step: StepFn) -> None:
        Gtk, GLib = self.Gtk, self.GLib
        win = Gtk.Window(title=self.title)
        win.set_default_size(self.w, self.h)
        win.set_resizable(False)
        self.picture = Gtk.Picture()
        self.picture.set_can_shrink(False)
        self.picture.set_size_request(self.w, self.h)
        win.set_child(self.picture)
        ctl = Gtk.EventControllerKey()
        ctl.connect("key-pressed", self._on_key)
        win.add_controller(ctl)
        win.connect("close-request", lambda *_: (self._keys.put(KEY_ESC), False)[1])
        win.present()
        try:
            self.backend = win.get_display().get_name()      # e.g. "wayland-0" or ":0"
        except Exception:
            pass

        def worker():
            try:
                while True:
                    canvas = step()
                    if canvas is None:
                        break
                    if not self._pending:            # drop frames rather than queue them
                        self._pending = True
                        GLib.idle_add(self._present, canvas, priority=GLib.PRIORITY_HIGH_IDLE)
            finally:
                GLib.idle_add(self._loop.quit)

        threading.Thread(target=worker, daemon=True, name="ward-ui-step").start()
        self._loop.run()
        win.close()
        # flush pending GTK work so the window really disappears before we return
        ctx = GLib.MainContext.default()
        while ctx.pending():
            ctx.iteration(False)


# --------------------------------------------------------------- OpenCV ----
class Cv2Display:
    name = "opencv-qt"
    snapshot = False

    def __init__(self, title: str, width: int, height: int):
        import cv2
        self.cv2, self.title = cv2, title
        try:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow(title, width, height)
            cv2.imshow(title, np.zeros((height, width, 3), np.uint8))
            cv2.waitKey(1)
        except cv2.error as e:
            raise DisplayUnavailable(str(e))
        self._keys: list[str] = []
        self.backend = "highgui"

    def keys(self) -> list[str]:
        out, self._keys = self._keys, []
        return out

    def run(self, step: StepFn) -> None:
        cv2 = self.cv2
        try:
            while True:
                canvas = step()
                if canvas is None:
                    break
                cv2.imshow(self.title, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
                k = cv2.waitKey(1) & 0xFF
                if k in (13, 10):
                    self._keys.append(KEY_ENTER)
                elif k == 27:
                    self._keys.append(KEY_ESC)
                elif k in (ord("q"), ord("Q")):
                    self._keys.append(KEY_Q)
        finally:
            cv2.destroyAllWindows(); cv2.waitKey(1)


# ---------------------------------------------------------------- stream ----
STREAM_ENV = "WARD_UI_STREAM"
_stream_out = None


def stream_out():
    """The pipe the manager app reads snapshots from: the process's original stdout. Taken once;
    from then on fd 1 points at stderr so every print (CLI messages, library noise) goes to the
    app's console instead of corrupting the snapshot stream."""
    global _stream_out
    if _stream_out is None:
        import sys
        sys.stdout.flush()
        _stream_out = os.fdopen(os.dup(1), "w", buffering=1)
        os.dup2(2, 1)
    return _stream_out


def encode_snapshot(snap: dict, quality: int = 72) -> str:
    """One JSON line: the window's snapshot dict with the camera frame as a base64 JPEG."""
    import base64
    import json

    import cv2
    out = {k: v for k, v in snap.items() if k != "frame"}
    fr = snap.get("frame")
    if fr is not None:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            out["jpg"] = base64.b64encode(buf.tobytes()).decode()

    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        return str(o)
    return json.dumps(out, default=conv)


def decode_snapshot(line: str) -> dict | None:
    """The inverse, in the manager app: the dict back, the frame as an RGB array."""
    import base64
    import json

    import cv2
    try:
        snap = json.loads(line)
    except ValueError:
        return None
    jpg = snap.pop("jpg", None)
    if jpg:
        img = cv2.imdecode(np.frombuffer(base64.b64decode(jpg), np.uint8), cv2.IMREAD_COLOR)
        snap["frame"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    return snap


class StreamDisplay:
    """No window: the snapshots go to the manager app (`ward app`), which draws them inside its
    own window, and its buttons come back as key lines on stdin. This keeps the camera and the
    0600 face store in the privileged process while the user sees one application."""
    name = "stream"
    snapshot = True
    MAX_FPS = 24

    def __init__(self, title: str, width: int, height: int):
        self.title, self.backend = title, "stream"
        self.out = stream_out()
        self._keys: queue.Queue = queue.Queue()
        threading.Thread(target=self._read_keys, daemon=True, name="ward-ui-keys").start()

    def _read_keys(self) -> None:
        import sys
        for line in sys.stdin:
            k = line.strip()
            if k:
                self._keys.put(k)
        self._keys.put(KEY_ESC)                 # the app went away: cancel, never hang

    def keys(self) -> list[str]:
        out = []
        while True:
            try:
                out.append(self._keys.get_nowait())
            except queue.Empty:
                return out

    def push_key(self, k: str) -> None:
        self._keys.put(k)

    def run(self, step) -> None:
        import time
        last, last_phase = 0.0, None
        while True:
            snap = step()
            if snap is None:
                break
            now = time.monotonic()
            if now - last < 1.0 / self.MAX_FPS and snap.get("phase") == last_phase:
                continue                         # throttle the pipe, but never drop a phase change
            last, last_phase = now, snap.get("phase")
            try:
                self.out.write(encode_snapshot(snap) + "\n")
            except (BrokenPipeError, OSError):
                self._keys.put(KEY_ESC)
        try:
            self.out.write('{"phase": "closed"}\n')
        except OSError:
            pass


def open_display(title: str, width: int, height: int, prefer: str = "gtk", kind: str = "raw", canvas=None, **kw):
    """Open a window. kind="enroll" / "emitter" / "lab" / "calibrate": the libadwaita application windows (ward.appui),
    which consume snapshot dicts (`screen.snapshot` is True); kind="raw" or any fallback: a
    canvas display (GTK4 picture, then OpenCV HighGUI). Raises DisplayUnavailable if nothing works."""
    errors = []
    if os.environ.get(STREAM_ENV) and kind == "calibrate":
        return StreamDisplay(title, width, height)
    if kind in ("enroll", "emitter", "lab", "calibrate") and prefer == "gtk":
        try:
            from . import appui
            cls = {"enroll": appui.EnrollWindow, "emitter": appui.EmitterWindow, "lab": appui.LabWindow, "calibrate": appui.CalibrateWindow}[kind]
            return cls(title, width, height, **kw)
        except DisplayUnavailable as e:
            errors.append(f"adwaita: {e}")
        except Exception as e:                       # PyGObject/Adw missing, CSS error…: fall back
            errors.append(f"adwaita: {type(e).__name__}: {e}")
    cw, ch = canvas or (width, height)          # fallback displays show the painted canvas
    order = [GtkDisplay, Cv2Display] if prefer == "gtk" else [Cv2Display, GtkDisplay]
    for cls in order:
        try:
            return cls(title, cw, ch)
        except DisplayUnavailable as e:
            errors.append(f"{cls.name}: {e}")
    raise DisplayUnavailable("; ".join(errors))
