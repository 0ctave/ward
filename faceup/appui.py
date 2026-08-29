"""faceup application windows — GTK 4 + libadwaita, native Wayland.

Two windows share one shell (`_Shell`): the enrollment window and the emitter-check window.
Both implement the display protocol used by `faceup.gui` step functions:

    screen.run(step)      GTK owns the main thread; `step()` runs on a worker thread and returns
                          a *snapshot dict* (see gui.py) or None when finished.
    screen.keys()         key events for the step function: "enter" | "esc" | "q"
    screen.snapshot       True — this display consumes snapshots, not pre-drawn canvases.

Rendering: camera frames are uploaded as Gdk.MemoryTexture (GPU-composited); everything else
(face outline, coverage map) is Cairo on DrawingAreas that redraw only when their data changed.
Colours follow the libadwaita theme (light/dark, accent); semantic colours are used only for
the face-light state and the emitter verdict.
"""
from __future__ import annotations

import math
import os
import queue
import threading
import time

import numpy as np

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Adw", "1")
import cairo  # noqa: E402  (pycairo, part of python3-gi's dependencies)
from gi.repository import Adw, Gdk, GLib, Gtk  # noqa: E402

from .engine import pose as posemod  # noqa: E402

KEY_ENTER, KEY_ESC, KEY_Q = "enter", "esc", "q"

CSS = b"""
.fu-preview { border-radius: 14px; }
.fu-card { background: alpha(currentColor, 0.045); border-radius: 14px; padding: 16px; }
.fu-hint { font-size: 17px; font-weight: 600; }
.fu-dim { opacity: 0.6; }
.fu-pill { border-radius: 999px; padding: 3px 11px; font-size: 12.5px; font-weight: 600;
           background: alpha(currentColor, 0.08); }
.fu-pill.ok   { background-color: @success_bg_color; color: @success_fg_color; }
.fu-pill.warn { background-color: @warning_bg_color; color: @warning_fg_color; }
.fu-pill.bad  { background-color: @error_bg_color; color: @error_fg_color; }
.fu-pill.accent { background-color: @accent_bg_color; color: @accent_fg_color; }
.fu-kbd { font-family: monospace; font-size: 11px; border-radius: 6px; padding: 1px 6px;
          border: 1px solid alpha(currentColor, 0.25); }
.fu-banner { border-radius: 10px; padding: 10px 14px; font-weight: 600; }
.fu-banner.ok   { background-color: @success_bg_color; color: @success_fg_color; }
.fu-banner.bad  { background-color: @error_bg_color; color: @error_fg_color; }
.fu-banner.warn { background-color: @warning_bg_color; color: @warning_fg_color; }
.fu-banner.info { background-color: alpha(currentColor, 0.06); }
.fu-progress trough { min-height: 5px; }
.fu-progress progress { min-height: 5px; }
"""

_css_loaded = False


def _load_css() -> None:
    global _css_loaded
    if _css_loaded:
        return
    prov = Gtk.CssProvider()
    prov.load_from_data(CSS)
    Gtk.StyleContext.add_provider_for_display(Gdk.Display.get_default(), prov, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
    _css_loaded = True


def _rgba(widget: Gtk.Widget, fallback=(0.21, 0.52, 0.89)) -> tuple[float, float, float]:
    """Foreground colour of a widget as (r, g, b) — used to read the theme's accent."""
    try:
        c = widget.get_color()
        if c.alpha > 0:
            return (c.red, c.green, c.blue)
    except Exception:
        pass
    return fallback


def _texture(frame: np.ndarray) -> Gdk.Texture:
    h, w = frame.shape[:2]
    data = np.ascontiguousarray(frame).tobytes()
    return Gdk.MemoryTexture.new(w, h, Gdk.MemoryFormat.R8G8B8, GLib.Bytes.new(data), w * 3)


# ------------------------------------------------------------------ widgets ----
class Preview(Gtk.Overlay):
    """Camera picture with a soft rounded face outline drawn on top."""

    def __init__(self, width: int, height: int):
        super().__init__()
        self.w, self.h = width, height
        self.picture = Gtk.Picture()
        self.picture.set_content_fit(Gtk.ContentFit.COVER)
        self.picture.set_size_request(width, height)
        self.picture.add_css_class("fu-preview")
        self.picture.set_overflow(Gtk.Overflow.HIDDEN)
        self.set_child(self.picture)
        self.area = Gtk.DrawingArea()
        self.area.set_can_target(False)
        self.area.set_draw_func(self._draw)
        self.add_overlay(self.area)
        self.box = None            # (x, y, w, h) in frame pixels
        self.frame_size = (width, height)
        self.state = "track"       # track | sampled | dark | none
        self._sampled_t = 0.0
        self.accent = (0.21, 0.52, 0.89)

    def set_frame(self, frame: np.ndarray) -> None:
        self.frame_size = (frame.shape[1], frame.shape[0])
        self.picture.set_paintable(_texture(frame))

    def set_box(self, box, state: str) -> None:
        changed = (box != self.box) or (state != self.state)
        self.box, self.state = box, state
        if state == "sampled":
            self._sampled_t = time.monotonic()
        if changed or state == "sampled":
            self.area.queue_draw()

    def _draw(self, area, cr: cairo.Context, W: int, H: int) -> None:
        if self.box is None:
            return
        fw, fh = self.frame_size
        # picture uses COVER: uniform scale, centred crop
        s = max(W / fw, H / fh)
        ox, oy = (W - fw * s) / 2, (H - fh * s) / 2
        x, y, w, h = self.box
        x, y, w, h = ox + x * s, oy + y * s, w * s, h * s
        r = min(w, h) * 0.18
        col = {"dark": (0.88, 0.11, 0.14), "sampled": (0.15, 0.64, 0.41)}.get(self.state, self.accent)
        glow = 1.0
        if self.state == "sampled":
            glow = max(0.0, 1.0 - (time.monotonic() - self._sampled_t) / 0.5)
        cr.new_sub_path()
        cr.arc(x + w - r, y + r, r, -math.pi / 2, 0)
        cr.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
        cr.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
        cr.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
        cr.close_path()
        cr.set_source_rgba(*col, 0.35 + 0.4 * glow)
        cr.set_line_width(6.0)
        cr.stroke_preserve()
        cr.set_source_rgba(*col, 0.95)
        cr.set_line_width(1.6)
        cr.stroke()
        # corner ticks for a "viewfinder" feel
        t = min(w, h) * 0.12
        cr.set_line_width(2.2)
        for (cx, cy, dx, dy) in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
            cr.move_to(cx, cy + dy * t); cr.line_to(cx, cy); cr.line_to(cx + dx * t, cy)
        cr.stroke()


class CoverageView(Gtk.DrawingArea):
    """2-D pose map: fixed required ellipse, coverage as soft accent-coloured contours, the
    current pose as an eased dot with a roll tick, the target as a pulsing ring."""

    def __init__(self, width: int, height: int):
        super().__init__()
        self.set_size_request(width, height)
        self.set_draw_func(self._draw)
        self.cov = None
        self.pose = None           # display-space (yaw, pitch, roll)
        self.shown = None          # eased pose
        self.target = None
        self.sgn = -1.0
        self.accent = (0.21, 0.52, 0.89)
        self.fg = (0.5, 0.5, 0.5)
        self._surface = None
        self._surface_key = None
        self._anim = None
        self._phase = 0.0

    def update(self, cov, pose, target, sgn: float) -> None:
        self.cov, self.pose, self.target, self.sgn = cov, pose, target, sgn
        if pose is not None and self.shown is None:
            self.shown = pose
        if self._anim is None:
            self._anim = GLib.timeout_add(16, self._tick)
        self.queue_draw()

    def _tick(self) -> bool:
        self._phase = (self._phase + 0.05) % 1.0
        if self.pose is not None and self.shown is not None:
            y, p, r = self.shown
            ty, tp, tr = self.pose
            self.shown = (y + 0.35 * (ty - y), p + 0.35 * (tp - p), r + 0.35 * (tr - r))
        elif self.pose is None:
            self.shown = None
        self.queue_draw()
        return True

    def stop(self) -> None:
        if self._anim is not None:
            GLib.source_remove(self._anim); self._anim = None

    # -- geometry helpers
    def _to_px(self, yaw, pitch, W, H, pad):
        y0, y1 = posemod.YAW_EDGES[0], posemod.YAW_EDGES[-1]
        p0, p1 = posemod.PITCH_EDGES[0], posemod.PITCH_EDGES[-1]
        x = pad + (self.sgn * yaw - y0) / (y1 - y0) * (W - 2 * pad) if self.sgn < 0 else pad + (yaw - y0) / (y1 - y0) * (W - 2 * pad)
        y = pad + (p1 - pitch) / (p1 - p0) * (H - 2 * pad)
        return x, y

    def _coverage_surface(self, cov):
        m = np.minimum(cov.map / cov.FULL, 1.0)
        key = (float(m.sum()), m.shape)
        if self._surface is not None and self._surface_key == key:
            return self._surface
        m = m[::-1]                                    # pitch "up" at the top
        if self.sgn < 0:
            m = m[:, ::-1]                             # mirror: user's left on the left
        req = cov.required[::-1]
        if self.sgn < 0:
            req = req[:, ::-1]
        a = np.where(req, 0.12 + 0.75 * m, 0.30 * m)   # alpha: soft, stronger inside the ellipse
        h, w = a.shape
        r, g, b = self.accent
        buf = np.zeros((h, w, 4), np.uint8)
        buf[..., 0] = (b * 255 * a).astype(np.uint8)   # premultiplied BGRA for cairo ARGB32
        buf[..., 1] = (g * 255 * a).astype(np.uint8)
        buf[..., 2] = (r * 255 * a).astype(np.uint8)
        buf[..., 3] = (255 * a).astype(np.uint8)
        surf = cairo.ImageSurface.create_for_data(bytearray(buf.tobytes()), cairo.FORMAT_ARGB32, w, h, w * 4)
        self._surface, self._surface_key = surf, key
        return surf

    def _draw(self, area, cr: cairo.Context, W: int, H: int) -> None:
        pad = 22
        fr, fg, fb = self.fg
        # frame
        cr.set_source_rgba(fr, fg, fb, 0.06)
        cr.rectangle(0, 0, W, H); cr.fill()
        if self.cov is None:
            return
        cov = self.cov
        # coverage field
        surf = self._coverage_surface(cov)
        sw, sh = surf.get_width(), surf.get_height()
        cr.save()
        cr.rectangle(pad, pad, W - 2 * pad, H - 2 * pad); cr.clip()
        cr.translate(pad, pad)
        cr.scale((W - 2 * pad) / sw, (H - 2 * pad) / sh)
        pat = cairo.SurfacePattern(surf); pat.set_filter(cairo.FILTER_BILINEAR)
        cr.set_source(pat); cr.paint()
        cr.restore()
        # quiet grid: centre lines
        cx0, cy0 = self._to_px(0, 0, W, H, pad)
        cr.set_source_rgba(fr, fg, fb, 0.18); cr.set_line_width(1)
        cr.move_to(cx0, pad); cr.line_to(cx0, H - pad); cr.move_to(pad, cy0); cr.line_to(W - pad, cy0); cr.stroke()
        # required ellipse (fixed outline)
        ecx, ecy = self._to_px(cov.centre[0], cov.centre[1], W, H, pad)
        ax = cov.axes[0] / (posemod.YAW_EDGES[-1] - posemod.YAW_EDGES[0]) * (W - 2 * pad)
        ay = cov.axes[1] / (posemod.PITCH_EDGES[-1] - posemod.PITCH_EDGES[0]) * (H - 2 * pad)
        cr.save(); cr.translate(ecx, ecy); cr.scale(ax, ay); cr.arc(0, 0, 1, 0, 2 * math.pi); cr.restore()
        cr.set_source_rgba(fr, fg, fb, 0.55); cr.set_line_width(1.5); cr.stroke()
        # captions
        cr.set_source_rgba(fr, fg, fb, 0.55)
        cr.select_font_face("Cantarell", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL); cr.set_font_size(11)
        for text, (tx, ty) in (("left", (6, H / 2 + 4)), ("right", (W - 30, H / 2 + 4)), ("up", (W / 2 - 8, 14)), ("down", (W / 2 - 14, H - 7))):
            cr.move_to(tx, ty); cr.show_text(text)
        # target ring (pulsing)
        if self.target is not None:
            tx, ty = self._to_px(self.target[0], self.target[1], W, H, pad)
            pulse = 0.5 + 0.5 * math.sin(self._phase * 2 * math.pi)
            r, g, b = self.accent
            cr.set_source_rgba(r, g, b, 0.25 + 0.35 * pulse); cr.set_line_width(2 + 2 * pulse)
            cr.arc(tx, ty, 11 + 3 * pulse, 0, 2 * math.pi); cr.stroke()
            cr.set_source_rgba(r, g, b, 0.9); cr.set_line_width(1.5)
            cr.arc(tx, ty, 10, 0, 2 * math.pi); cr.stroke()
        # current pose dot with roll tick
        if self.shown is not None:
            y, p, roll = self.shown
            px, py = self._to_px(y, p, W, H, pad)
            r, g, b = self.accent
            cr.set_source_rgba(1, 1, 1, 0.9); cr.arc(px, py, 7.5, 0, 2 * math.pi); cr.fill()
            cr.set_source_rgba(r, g, b, 1.0); cr.arc(px, py, 5.5, 0, 2 * math.pi); cr.fill()
            a = math.radians(roll)
            cr.set_line_width(2.2); cr.move_to(px, py); cr.line_to(px + 14 * math.cos(a), py + 14 * math.sin(a)); cr.stroke()


def _pill(text: str = "", cls: str = "") -> Gtk.Label:
    lbl = Gtk.Label(label=text)
    lbl.add_css_class("fu-pill")
    if cls:
        lbl.add_css_class(cls)
    return lbl


def _set_pill(lbl: Gtk.Label, text: str, cls: str) -> None:
    lbl.set_label(text)
    for c in ("ok", "warn", "bad", "accent"):
        lbl.remove_css_class(c)
    if cls:
        lbl.add_css_class(cls)


# -------------------------------------------------------------------- shell ----
class _Shell:
    """Common window plumbing: Adw window + header, key queue, worker thread driving `step`."""
    snapshot = True

    def __init__(self, title: str, subtitle: str, width: int, height: int):
        from .display import user_wayland_env
        for k, v in user_wayland_env().items():
            os.environ.setdefault(k, v)
        os.environ.setdefault("GSK_RENDERER", "ngl")
        if os.geteuid() == 0:
            os.environ.setdefault("GTK_A11Y", "none")
        if not Gtk.init_check():
            from .display import DisplayUnavailable
            raise DisplayUnavailable("GTK could not open a display")
        Adw.init()
        _load_css()
        self.win = Adw.Window(title=title)
        self.win.set_default_size(width, height)
        self.win.set_resizable(False)
        self.header = Adw.HeaderBar()
        self.title = Adw.WindowTitle(title=title, subtitle=subtitle)
        self.header.set_title_widget(self.title)
        self.root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.root.append(self.header)
        self.win.set_content(self.root)
        self._keys: queue.Queue = queue.Queue()
        self._pending = False
        self._loop = GLib.MainLoop()
        ctl = Gtk.EventControllerKey(); ctl.connect("key-pressed", self._on_key); self.win.add_controller(ctl)
        self.win.connect("close-request", lambda *_: (self._keys.put(KEY_ESC), False)[1])
        self.backend = "?"
        self.name = "adwaita"

    def _on_key(self, _c, keyval, _code, _state) -> bool:
        if keyval in (Gdk.KEY_Return, Gdk.KEY_KP_Enter):
            self._keys.put(KEY_ENTER)
        elif keyval == Gdk.KEY_Escape:
            self._keys.put(KEY_ESC)
        elif keyval in (Gdk.KEY_q, Gdk.KEY_Q):
            self._keys.put(KEY_Q)
        return True

    def keys(self) -> list[str]:
        out = []
        while True:
            try:
                out.append(self._keys.get_nowait())
            except queue.Empty:
                return out

    def push_key(self, k: str) -> None:
        self._keys.put(k)

    def _apply(self, snap) -> bool:      # overridden
        self._pending = False
        return False

    def run(self, step) -> None:
        self.win.present()
        try:
            self.backend = self.win.get_display().get_name()
        except Exception:
            pass

        def worker():
            try:
                while True:
                    snap = step()
                    if snap is None:
                        break
                    if not self._pending:
                        self._pending = True
                        GLib.idle_add(self._apply, snap, priority=GLib.PRIORITY_HIGH_IDLE)
            finally:
                GLib.idle_add(self._loop.quit)

        threading.Thread(target=worker, daemon=True, name="faceup-ui-step").start()
        self._loop.run()
        self.on_close()
        self.win.close()
        ctx = GLib.MainContext.default()
        while ctx.pending():
            ctx.iteration(False)

    def on_close(self) -> None:
        pass


# --------------------------------------------------------------- enrollment ----
class EnrollWindow(_Shell):
    """Enrollment: preview + coverage map on the left, guidance / status / actions on the right."""

    def __init__(self, title: str = "faceup", width: int = 640, height: int = 360, sgn: float = -1.0):
        super().__init__(title, "Enrollment", 960, 620)
        self.sgn = sgn
        pw, ph = 620, int(620 * height / width)
        body = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=18)
        body.set_margin_top(14); body.set_margin_bottom(16); body.set_margin_start(18); body.set_margin_end(18)
        self.root.append(body)

        left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self.preview = Preview(pw, ph)
        left.append(self.preview)
        self.coverage = CoverageView(pw, 175)
        self.coverage.add_css_class("fu-preview"); self.coverage.set_overflow(Gtk.Overflow.HIDDEN)
        left.append(self.coverage)
        body.append(left)

        right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        right.set_size_request(280, -1)
        body.append(right)

        card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8); card.add_css_class("fu-card")
        self.phase_lbl = Gtk.Label(xalign=0); self.phase_lbl.add_css_class("fu-dim"); self.phase_lbl.add_css_class("caption")
        self.hint_lbl = Gtk.Label(xalign=0, wrap=True); self.hint_lbl.add_css_class("fu-hint")
        self.hint_lbl.set_max_width_chars(24)
        self.calib_bar = Gtk.ProgressBar(); self.calib_bar.add_css_class("fu-progress"); self.calib_bar.set_visible(False)
        card.append(self.phase_lbl); card.append(self.hint_lbl); card.append(self.calib_bar)
        right.append(card)

        stat = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8); stat.add_css_class("fu-card")
        cov_row = Gtk.Box(spacing=8)
        self.cov_lbl = Gtk.Label(xalign=0, label="Coverage"); self.cov_lbl.add_css_class("fu-dim")
        self.cov_val = Gtk.Label(xalign=1, label="0 %"); self.cov_val.add_css_class("title-4"); self.cov_val.set_hexpand(True)
        cov_row.append(self.cov_lbl); cov_row.append(self.cov_val)
        self.cov_bar = Gtk.ProgressBar(); self.cov_bar.add_css_class("fu-progress")
        pills = Gtk.FlowBox(); pills.set_selection_mode(Gtk.SelectionMode.NONE); pills.set_column_spacing(6); pills.set_row_spacing(6)
        pills.set_max_children_per_line(3); pills.set_homogeneous(False)
        self.p_models = _pill("0 poses", "accent"); self.p_light = _pill("no face"); self.p_fps = _pill("– fps")
        self.p_fps.add_css_class("fu-dim")
        for p in (self.p_models, self.p_light, self.p_fps):
            pills.append(p)
        stat.append(cov_row); stat.append(self.cov_bar); stat.append(pills)
        right.append(stat)

        self.detail = Gtk.Label(xalign=0, wrap=True); self.detail.add_css_class("caption"); self.detail.add_css_class("fu-dim")
        self.detail.set_max_width_chars(34)
        right.append(self.detail)

        spacer = Gtk.Box(); spacer.set_vexpand(True); right.append(spacer)
        keys = Gtk.Box(spacing=10)
        for k, t in (("Enter", "finish"), ("Esc", "cancel")):
            kb = Gtk.Label(label=k); kb.add_css_class("fu-kbd"); keys.append(kb)
            tl = Gtk.Label(label=t); tl.add_css_class("fu-dim"); tl.add_css_class("caption"); keys.append(tl)
        right.append(keys)
        actions = Gtk.Box(spacing=8)
        self.btn_cancel = Gtk.Button(label="Cancel"); self.btn_cancel.connect("clicked", lambda *_: self.push_key(KEY_ESC))
        self.btn_finish = Gtk.Button(label="Finish"); self.btn_finish.add_css_class("suggested-action")
        self.btn_finish.connect("clicked", lambda *_: self.push_key(KEY_ENTER))
        self.btn_finish.set_hexpand(True); self.btn_cancel.set_hexpand(True)
        actions.append(self.btn_cancel); actions.append(self.btn_finish)
        right.append(actions)

        # read the theme accent once the window is realised
        probe = Gtk.Label(); probe.add_css_class("accent"); right.append(probe)
        probe.set_visible(False)
        self._probe = probe
        self.win.connect("realize", lambda *_: self._read_theme())

    def _read_theme(self) -> None:
        self.preview.accent = self.coverage.accent = _rgba(self._probe)
        self.coverage.fg = _rgba(self.hint_lbl, (0.5, 0.5, 0.5))

    def _apply(self, s: dict) -> bool:
        try:
            self.preview.set_frame(s["frame"])
            light = s.get("light", "")
            gate = s.get("gate", "")
            box_state = "dark" if light == "dark" else ("sampled" if s.get("sampled") else ("track" if s.get("box") else "none"))
            self.preview.set_box(s.get("box"), box_state)
            self.coverage.update(s["coverage"], s.get("pose"), s.get("target"), self.sgn)
            phase = s.get("phase", "enrolling")
            self.title.set_subtitle({"calibrating": "Calibrating neutral pose", "enrolling": "Enrollment", "done": "Complete"}.get(phase, phase))
            if phase == "calibrating":
                self.phase_lbl.set_label("STEP 1 · NEUTRAL POSE")
                self.hint_lbl.set_label("Look at the middle of the screen — not the camera — and hold still.")
                self.calib_bar.set_visible(True); self.calib_bar.set_fraction(s.get("calib_progress", 0.0))
            elif phase == "done":
                self.phase_lbl.set_label("STEP 3 · DONE")
                self.hint_lbl.set_label("Coverage complete. Press Finish to save your face models.")
                self.calib_bar.set_visible(False)
            else:
                self.phase_lbl.set_label("STEP 2 · MOVE YOUR HEAD")
                self.hint_lbl.set_label(s.get("hint", ""))
                self.calib_bar.set_visible(False)
            frac = s.get("fraction", 0.0)
            self.cov_val.set_label(f"{frac * 100:.0f} %"); self.cov_bar.set_fraction(min(1.0, frac))
            _set_pill(self.p_models, f"{s.get('n_models', 0)} poses", "accent")
            light_cls = {"ok": "ok", "dim": "warn", "bright": "warn", "dark": "bad", "blown": "bad"}.get(light, "")
            _set_pill(self.p_light, ("face " + light) if light else (gate or "no face"), light_cls)
            _set_pill(self.p_fps, f"{s.get('fps', 0):.0f} fps", "")
            pose = s.get("pose")
            det = f"IR {s.get('ir_mean', 0):.0f}"
            if pose:
                det += f"   ·   yaw {pose[0]:+.0f}°   pitch {pose[1]:+.0f}°   roll {pose[2]:+.0f}°"
            if gate and gate not in ("", "sampling"):
                det += f"   ·   {gate}"
            self.detail.set_label(det)
        finally:
            self._pending = False
        return False

    def on_close(self) -> None:
        self.coverage.stop()


# ------------------------------------------------------------------ emitter ----
class EmitterWindow(_Shell):
    """IR and RGB side by side with their brightness, and the verdict as a banner."""

    def __init__(self, title: str = "faceup", width: int = 1280, height: int = 360):
        super().__init__(title, "IR emitter check", 980, 520)
        body = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        body.set_margin_top(14); body.set_margin_bottom(16); body.set_margin_start(18); body.set_margin_end(18)
        self.root.append(body)
        intro = Gtk.Label(xalign=0, wrap=True, label="Turn the room lights off. If the colour camera goes dark while the IR camera still shows your face, the emitter is working.")
        intro.add_css_class("fu-dim"); body.append(intro)
        row = Gtk.Box(spacing=12)
        self.ir = Preview(460, 259); self.rgb = Preview(460, 259)
        for prev, name in ((self.ir, "IR"), (self.rgb, "RGB")):
            col = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
            head = Gtk.Box(spacing=8)
            head.append(Gtk.Label(label=name, xalign=0))
            pill = _pill("–"); head.append(pill)
            prev.pill = pill
            col.append(head); col.append(prev); row.append(col)
        body.append(row)
        self.banner = Gtk.Label(xalign=0, wrap=True); self.banner.add_css_class("fu-banner"); self.banner.add_css_class("info")
        self.banner.set_label("Waiting for the cameras…")
        body.append(self.banner)
        foot = Gtk.Box(spacing=10)
        kb = Gtk.Label(label="Esc"); kb.add_css_class("fu-kbd"); foot.append(kb)
        fl = Gtk.Label(label="close"); fl.add_css_class("fu-dim"); fl.add_css_class("caption"); foot.append(fl)
        sp = Gtk.Box(); sp.set_hexpand(True); foot.append(sp)
        btn = Gtk.Button(label="Done"); btn.add_css_class("suggested-action"); btn.connect("clicked", lambda *_: self.push_key(KEY_Q)); foot.append(btn)
        body.append(foot)

    def _apply(self, s: dict) -> bool:
        try:
            if s.get("ir") is not None:
                self.ir.set_frame(s["ir"]); self.ir.set_box(s.get("box"), "track" if s.get("box") else "none")
                _set_pill(self.ir.pill, f"mean {s.get('ir_mean', 0):.0f}", "ok" if s.get("ir_mean", 0) >= 45 else "warn")
            if s.get("rgb") is not None:
                self.rgb.set_frame(s["rgb"])
                _set_pill(self.rgb.pill, f"mean {s.get('rgb_mean', 0):.0f}", "")
            state = s.get("state", "unknown")
            cls = {"ok": "ok", "dead": "bad", "inconclusive": "warn"}.get(state, "info")
            for c in ("ok", "bad", "warn", "info"):
                self.banner.remove_css_class(c)
            self.banner.add_css_class(cls)
            self.banner.set_label(s.get("msg", ""))
        finally:
            self._pending = False
        return False
