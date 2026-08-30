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
.fu-surface { border-radius: 14px; }
.fu-card { background: alpha(currentColor, 0.04); border-radius: 14px; padding: 20px; }
.fu-eyebrow { font-size: 11px; font-weight: 700; letter-spacing: 0.08em; opacity: 0.55; }
.fu-hint { font-size: 17px; font-weight: 600; line-height: 1.3; }
.fu-dim { opacity: 0.55; }
.fu-sep { background: alpha(currentColor, 0.10); min-height: 1px; }
.fu-pill { border-radius: 999px; padding: 3px 10px; font-size: 12px; font-weight: 600;
           background: alpha(currentColor, 0.08); }
.fu-pill.ok   { background-color: @success_bg_color; color: @success_fg_color; }
.fu-pill.warn { background-color: @warning_bg_color; color: @warning_fg_color; }
.fu-pill.bad  { background-color: @error_bg_color; color: @error_fg_color; }
.fu-pill.accent { background-color: @accent_bg_color; color: @accent_fg_color; }
.fu-kbd { font-family: monospace; font-size: 10.5px; border-radius: 5px; padding: 1px 6px;
          border: 1px solid alpha(currentColor, 0.22); opacity: 0.8; }
.fu-banner { border-radius: 10px; padding: 10px 14px; font-weight: 600; }
.fu-banner.ok   { background-color: @success_bg_color; color: @success_fg_color; }
.fu-banner.bad  { background-color: @error_bg_color; color: @error_fg_color; }
.fu-banner.warn { background-color: @warning_bg_color; color: @warning_fg_color; }
.fu-banner.info { background-color: alpha(currentColor, 0.06); }
.fu-progress trough { min-height: 4px; }
.fu-progress progress { min-height: 4px; }
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
        self.picture.add_css_class("fu-surface")
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
    """2-D pose map. The coverage field (a 2° Gaussian grid) is upsampled to pixel resolution
    with bicubic interpolation and drawn as a soft luminous accent field with one faint
    iso-contour at the "covered" threshold; a hairline marks the required ellipse; the
    current pose is an eased white dot with a roll tick; the target a blinking accent dot with a breathing shadow.
    Everything handed in is display space already — no sign handling here."""
    PULSE_PERIOD = 1.2       # s
    FADE = 0.25              # s, cross-fade when the field changes

    def __init__(self, width: int, height: int):
        super().__init__()
        self.set_size_request(width, height)
        self.set_draw_func(self._draw)
        self.cov = None
        self.pose = None
        self.shown = None
        self.target = None
        self.accent = (0.21, 0.52, 0.89)
        self.fg = (0.5, 0.5, 0.5)
        self._surf = None            # current field surface
        self._surf_key = None
        self._prev_surf = None       # previous field, for the cross-fade
        self._fade_t0 = 0.0
        self._anim = None
        self._t0 = time.monotonic()

    def update(self, cov, pose, target, sgn: float = 1.0) -> None:
        self.cov, self.pose, self.target = cov, pose, target
        if pose is not None and self.shown is None:
            self.shown = pose
        if self._anim is None:
            self._anim = GLib.timeout_add(16, self._tick)
        self.queue_draw()

    def _tick(self) -> bool:
        if self.pose is not None and self.shown is not None:
            y, p, r = self.shown; ty, tp, tr = self.pose
            self.shown = (y + 0.3 * (ty - y), p + 0.3 * (tp - p), r + 0.3 * (tr - r))
        elif self.pose is None:
            self.shown = None
        self.queue_draw()
        return True

    def stop(self) -> None:
        if self._anim is not None:
            GLib.source_remove(self._anim); self._anim = None

    def _to_px(self, yaw, pitch, W, H, pad):
        y0, y1 = posemod.YAW_EDGES[0], posemod.YAW_EDGES[-1]
        p0, p1 = posemod.PITCH_EDGES[0], posemod.PITCH_EDGES[-1]
        return pad + (yaw - y0) / (y1 - y0) * (W - 2 * pad), pad + (p1 - pitch) / (p1 - p0) * (H - 2 * pad)

    def _field(self, cov, iw: int, ih: int):
        """Pixel-resolution ARGB32 surface of the coverage field; cached on a cheap map hash.
        The raw (unclipped) Gaussian field is upsampled bicubically, so there is no plateau
        edge to alias; the ellipse mask is analytic at pixel resolution and feathered."""
        import cv2
        raw = (cov.map / cov.FULL)[::-1].astype(np.float32)                 # pitch "up" at the top
        key = (round(float(raw.sum()), 3), round(float(raw.max()), 3), iw, ih, self.accent)
        if self._surf is not None and self._surf_key == key:
            return self._surf
        big = np.maximum(cv2.resize(raw, (iw, ih), interpolation=cv2.INTER_CUBIC), 0.0)
        # analytic, feathered ellipse mask in pixel space
        y0, y1 = posemod.YAW_EDGES[0], posemod.YAW_EDGES[-1]
        p0, p1 = posemod.PITCH_EDGES[0], posemod.PITCH_EDGES[-1]
        yaws = y0 + (np.arange(iw) + 0.5) / iw * (y1 - y0)
        pitches = p1 - (np.arange(ih) + 0.5) / ih * (p1 - p0)
        cy, cp = cov.centre; ay, ap = cov.axes
        d = np.sqrt(((yaws[None, :] - cy) / ay) ** 2 + ((pitches[:, None] - cp) / ap) ** 2)
        inside = np.clip((1.06 - d) / 0.12, 0.0, 1.0)                       # 1 inside, feathered ~1.5° at the rim
        inside = inside * inside * (3 - 2 * inside)
        # luminous ramp without plateau: asymptotic in the field value; quieter outside the ellipse
        lum = 1.0 - np.exp(-1.4 * big)
        alpha = (0.05 + 0.50 * lum) * inside + (0.20 * lum) * (1.0 - inside)
        # one faint iso-line where the field crosses "covered" (value 1)
        contour = np.exp(-((big - 1.0) / 0.06) ** 2) * (big > 0.5)
        alpha = np.clip(alpha + 0.16 * contour * inside, 0.0, 0.9).astype(np.float32)
        r, g, b = self.accent
        buf = np.empty((ih, iw, 4), np.uint8)
        buf[..., 0] = (b * 255 * alpha).astype(np.uint8)                   # premultiplied BGRA
        buf[..., 1] = (g * 255 * alpha).astype(np.uint8)
        buf[..., 2] = (r * 255 * alpha).astype(np.uint8)
        buf[..., 3] = (255 * alpha).astype(np.uint8)
        surf = cairo.ImageSurface.create_for_data(bytearray(buf.tobytes()), cairo.FORMAT_ARGB32, iw, ih, iw * 4)
        if self._surf is not None:
            self._prev_surf, self._fade_t0 = self._surf, time.monotonic()
        self._surf, self._surf_key = surf, key
        return surf

    def _draw(self, area, cr: cairo.Context, W: int, H: int) -> None:
        pad = 20
        fr, fg, fb = self.fg
        cr.set_source_rgba(fr, fg, fb, 0.04); cr.rectangle(0, 0, W, H); cr.fill()
        if self.cov is None:
            return
        cov = self.cov
        iw, ih = W - 2 * pad, H - 2 * pad
        surf = self._field(cov, iw, ih)
        t = time.monotonic()
        k = min(1.0, (t - self._fade_t0) / self.FADE) if self._prev_surf is not None else 1.0
        cr.save(); cr.rectangle(pad, pad, iw, ih); cr.clip()
        if self._prev_surf is not None and k < 1.0:
            cr.set_source_surface(self._prev_surf, pad, pad); cr.paint_with_alpha(1.0 - k)
        cr.set_source_surface(surf, pad, pad); cr.paint_with_alpha(k)
        cr.restore()
        if k >= 1.0:
            self._prev_surf = None
        # required ellipse: hairline
        ecx, ecy = self._to_px(cov.centre[0], cov.centre[1], W, H, pad)
        ax = cov.axes[0] / (posemod.YAW_EDGES[-1] - posemod.YAW_EDGES[0]) * iw
        ay = cov.axes[1] / (posemod.PITCH_EDGES[-1] - posemod.PITCH_EDGES[0]) * ih
        cr.new_path(); cr.save(); cr.translate(ecx, ecy); cr.scale(ax, ay); cr.arc(0, 0, 1, 0, 2 * math.pi); cr.restore()
        cr.set_source_rgba(fr, fg, fb, 0.38); cr.set_line_width(1.0); cr.stroke()
        # quiet captions
        cr.set_source_rgba(fr, fg, fb, 0.42)
        cr.select_font_face("Cantarell", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL); cr.set_font_size(10)
        for text, (tx, ty) in (("left", (8, H / 2 + 3)), ("right", (W - 30, H / 2 + 3)), ("up", (W / 2 - 6, 13)), ("down", (W / 2 - 13, H - 6))):
            cr.move_to(tx, ty); cr.show_text(text)
        r, g, b = self.accent
        # target: a blinking accent dot with a breathing soft shadow ("look here")
        if self.target is not None:
            tx, ty = self._to_px(self.target[0], self.target[1], W, H, pad)
            pulse = 0.5 + 0.5 * math.sin((t - self._t0) / self.PULSE_PERIOD * 2 * math.pi)   # 0..1
            halo = 14 + 8 * pulse
            grad = cairo.RadialGradient(tx, ty, 2, tx, ty, halo)
            grad.add_color_stop_rgba(0.0, r, g, b, 0.45 * pulse + 0.10)
            grad.add_color_stop_rgba(1.0, r, g, b, 0.0)
            cr.new_path(); cr.set_source(grad); cr.arc(tx, ty, halo, 0, 2 * math.pi); cr.fill()
            cr.new_path(); cr.set_source_rgba(r, g, b, 0.35 + 0.65 * pulse)                # the blink
            cr.arc(tx, ty, 5.5, 0, 2 * math.pi); cr.fill()
            cr.new_path(); cr.set_source_rgba(1, 1, 1, 0.55 * pulse); cr.set_line_width(1.0)
            cr.arc(tx, ty, 5.5, 0, 2 * math.pi); cr.stroke()
        # pose dot with roll tick
        if self.shown is not None:
            y, p, roll = self.shown
            px, py = self._to_px(y, p, W, H, pad)
            cr.new_path(); cr.set_source_rgba(1, 1, 1, 0.85); cr.arc(px, py, 6.5, 0, 2 * math.pi); cr.fill()
            cr.new_path(); cr.set_source_rgba(r, g, b, 1.0); cr.arc(px, py, 4.8, 0, 2 * math.pi); cr.fill()
            a = math.radians(roll)
            cr.set_source_rgba(r, g, b, 0.9); cr.set_line_width(1.8)
            cr.move_to(px + 6 * math.cos(a), py + 6 * math.sin(a)); cr.line_to(px + 13 * math.cos(a), py + 13 * math.sin(a)); cr.stroke()


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
        self.win.set_focus(None)          # no widget grabs focus on open (a stray keystroke replaced the subject text)
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
    """Enrollment: preview and pose map on the left (same width and radius), one guidance card
    on the right with the coverage progress, and the actions at the bottom."""

    def __init__(self, title: str = "faceup", width: int = 640, height: int = 360, sgn: float = 1.0):
        super().__init__(title, "Enrollment", 960, 640)
        pw, ph = 600, int(600 * height / width)
        body = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=24)
        for side in ("top", "bottom", "start", "end"):
            getattr(body, f"set_margin_{side}")(24)
        body.set_margin_top(16)
        self.root.append(body)

        left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        self.preview = Preview(pw, ph)
        left.append(self.preview)
        self.coverage = CoverageView(pw, 190)
        self.coverage.add_css_class("fu-surface"); self.coverage.set_overflow(Gtk.Overflow.HIDDEN)
        left.append(self.coverage)
        body.append(left)

        right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        right.set_size_request(264, -1)
        body.append(right)

        card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12); card.add_css_class("fu-card")
        self.phase_lbl = Gtk.Label(xalign=0); self.phase_lbl.add_css_class("fu-eyebrow")
        self.hint_lbl = Gtk.Label(xalign=0, wrap=True); self.hint_lbl.add_css_class("fu-hint")
        self.hint_lbl.set_max_width_chars(22)
        self.calib_bar = Gtk.ProgressBar(); self.calib_bar.add_css_class("fu-progress"); self.calib_bar.set_visible(False)
        sep = Gtk.Box(); sep.add_css_class("fu-sep"); sep.set_margin_top(4); sep.set_margin_bottom(4)
        cov_row = Gtk.Box(spacing=8)
        cov_lbl = Gtk.Label(xalign=0, label="Coverage"); cov_lbl.add_css_class("fu-dim")
        self.cov_val = Gtk.Label(xalign=1, label="0 %"); self.cov_val.add_css_class("title-3"); self.cov_val.set_hexpand(True)
        cov_row.append(cov_lbl); cov_row.append(self.cov_val)
        self.cov_bar = Gtk.ProgressBar(); self.cov_bar.add_css_class("fu-progress")
        pill_row = Gtk.Box(spacing=8)
        self.p_light = _pill("Looking for your face")
        pill_row.append(self.p_light)
        self.detail = Gtk.Label(xalign=0); self.detail.add_css_class("caption"); self.detail.add_css_class("fu-dim")
        for wdg in (self.phase_lbl, self.hint_lbl, self.calib_bar, sep, cov_row, self.cov_bar, pill_row, self.detail):
            card.append(wdg)
        right.append(card)

        spacer = Gtk.Box(); spacer.set_vexpand(True); right.append(spacer)
        keys = Gtk.Box(spacing=8)
        for k, t in (("Enter", "finish"), ("Esc", "cancel")):
            kb = Gtk.Label(label=k); kb.add_css_class("fu-kbd"); keys.append(kb)
            tl = Gtk.Label(label=t); tl.add_css_class("fu-dim"); tl.add_css_class("caption"); tl.set_margin_end(6); keys.append(tl)
        right.append(keys)
        actions = Gtk.Box(spacing=8)
        self.btn_cancel = Gtk.Button(label="Cancel"); self.btn_cancel.connect("clicked", lambda *_: self.push_key(KEY_ESC))
        self.btn_finish = Gtk.Button(label="Finish"); self.btn_finish.connect("clicked", lambda *_: self.push_key(KEY_ENTER))
        self.btn_finish.set_hexpand(True); self.btn_cancel.set_hexpand(True)
        actions.append(self.btn_cancel); actions.append(self.btn_finish)
        right.append(actions)

        probe = Gtk.Label(); probe.add_css_class("accent"); probe.set_visible(False); right.append(probe)
        self._probe = probe
        self.win.connect("realize", lambda *_: self._read_theme())

    def _read_theme(self) -> None:
        self.preview.accent = self.coverage.accent = _rgba(self._probe)
        self.coverage.fg = _rgba(self.hint_lbl, (0.5, 0.5, 0.5))

    def _apply(self, s: dict) -> bool:
        try:
            self.preview.set_frame(s["frame"])
            light = s.get("light", "")
            box_state = "dark" if light == "dark" else ("sampled" if s.get("sampled") else ("track" if s.get("box") else "none"))
            self.preview.set_box(s.get("box"), box_state)
            self.coverage.update(s["coverage"], s.get("pose"), s.get("target"))
            phase = s.get("phase", "enrolling")
            frac = s.get("fraction", 0.0)
            self.title.set_subtitle({"calibrating": "Calibrating your neutral pose", "enrolling": "Move your head slowly",
                                     "done": "Coverage complete"}.get(phase, phase))
            if phase == "calibrating":
                self.phase_lbl.set_label("STEP 1 OF 2 · NEUTRAL POSE")
                self.hint_lbl.set_label("Look at the middle of the screen, not the camera, and hold still.")
                self.calib_bar.set_visible(True); self.calib_bar.set_fraction(s.get("calib_progress", 0.0))
            elif phase == "done":
                self.phase_lbl.set_label("DONE")
                self.hint_lbl.set_label("Every direction is covered. Finish to save your face models.")
                self.calib_bar.set_visible(False)
            else:
                self.phase_lbl.set_label("STEP 2 OF 2 · COVERAGE")
                self.hint_lbl.set_label(s.get("hint", ""))
                self.calib_bar.set_visible(False)
            self.cov_val.set_label(f"{frac * 100:.0f} %"); self.cov_bar.set_fraction(min(1.0, frac))
            if frac >= 0.97:
                self.btn_finish.add_css_class("suggested-action")
            else:
                self.btn_finish.remove_css_class("suggested-action")
            light_txt = {"ok": "Face well lit", "dim": "Face a little dim", "bright": "Face a little bright",
                         "dark": "Face too dark", "blown": "Face overexposed"}.get(light)
            gate = s.get("gate", "")
            if light_txt is None:
                light_txt = {"no face": "Looking for your face", "too far": "Come a little closer", "moving": "Hold still a moment"}.get(gate, "Looking for your face")
            light_cls = {"ok": "ok", "dim": "warn", "bright": "warn", "dark": "bad", "blown": "bad"}.get(light, "")
            _set_pill(self.p_light, light_txt, light_cls)
            self.detail.set_label(f"{s.get('n_models', 0)} poses stored  ·  {s.get('spectrum', 'ir').upper()} brightness {s.get('ir_mean', 0):.0f}  ·  {s.get('fps', 0):.0f} fps")
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


# ---------------------------------------------------------------------- lab ----
class MetricBar(Gtk.DrawingArea):
    """A thin horizontal bar for one distance: the value as an accent bar, the gate as a mark,
    optionally a hard cap as a second (dimmer) mark. `ratio=True` draws a 0..2 scale with the
    threshold at 1 (whitened distance / threshold: < 1 accepts)."""

    def __init__(self, ratio: bool = False):
        super().__init__()
        self.set_size_request(-1, 10)
        self.set_hexpand(True)
        self.set_draw_func(self._draw)
        self.ratio = ratio
        self.value: float | None = None
        self.gate: float | None = None
        self.cap: float | None = None
        self.scale_max = 2.0 if ratio else 0.8
        self.accent = (0.21, 0.52, 0.89)
        self.fg = (0.5, 0.5, 0.5)

    def set(self, value, gate=None, cap=None) -> None:
        self.value, self.gate, self.cap = value, gate, cap
        self.queue_draw()

    def _x(self, v: float, W: float) -> float:
        return max(0.0, min(1.0, v / self.scale_max)) * W

    def _draw(self, area, cr: cairo.Context, W: int, H: int) -> None:
        fr, fg, fb = self.fg
        y, h = H / 2 - 2, 4
        cr.set_source_rgba(fr, fg, fb, 0.12)
        cr.rectangle(0, y, W, h); cr.fill()
        if self.value is not None:
            ok = (self.value <= (self.gate if self.gate is not None else 1.0)) if not self.ratio else self.value <= 1.0
            r, g, b = (0.15, 0.64, 0.41) if ok else (0.88, 0.11, 0.14)
            cr.set_source_rgba(r, g, b, 0.85)
            cr.rectangle(0, y, self._x(self.value, W), h); cr.fill()
        marks = [(1.0, 0.9)] if self.ratio else [(self.gate, 0.9), (self.cap, 0.4)]
        for v, a in marks:
            if v is None:
                continue
            x = self._x(v, W)
            cr.set_source_rgba(fr, fg, fb, a)
            cr.rectangle(x - 0.75, 0, 1.5, H); cr.fill()


class LabWindow(_Shell):
    """Live evaluation instrument: the decision under three unlock policies on the same frames,
    every distance the engine sees, liveness votes, threshold sliders, and a recorded-session
    summary per subject (yourself, other people, a photo…)."""

    def __init__(self, title: str = "faceup lab", width: int = 640, height: int = 360, evaluator=None):
        super().__init__(title, "Live evaluation", 1080, 700)
        from .engine import lab as labmod
        self.labmod = labmod
        self.ev = evaluator
        pw, ph = 470, int(470 * height / width)
        body = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=20)
        for side in ("bottom", "start", "end"):
            getattr(body, f"set_margin_{side}")(20)
        body.set_margin_top(12)
        self.root.append(body)

        # ---- left: preview + pose caption + light pill
        left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.preview = Preview(pw, ph)
        left.append(self.preview)
        under = Gtk.Box(spacing=8)
        # camera switch: which stream the preview shows (the engine always evaluates both)
        self.view = "ir"
        sw = Gtk.Box(); sw.add_css_class("linked")
        self.btn_ir = Gtk.ToggleButton(label="IR"); self.btn_rgb = Gtk.ToggleButton(label="RGB", group=self.btn_ir)
        self.btn_ir.set_active(True)
        for b, sp in ((self.btn_ir, "ir"), (self.btn_rgb, "rgb")):
            b.add_css_class("flat"); b.connect("toggled", lambda w, sp=sp: setattr(self, "view", sp) if w.get_active() else None)
            sw.append(b)
        under.append(sw)
        self.brighten = Gtk.CheckButton(label="Brighten dark frames"); self.brighten.set_active(True)
        self.brighten.add_css_class("caption"); under.append(self.brighten)
        self.p_light = _pill("Looking for a face"); under.append(self.p_light)
        self.pose_lbl = Gtk.Label(xalign=0); self.pose_lbl.add_css_class("caption"); self.pose_lbl.add_css_class("fu-dim")
        self.pose_lbl.set_hexpand(True); under.append(self.pose_lbl)
        self.fps_lbl = Gtk.Label(xalign=1); self.fps_lbl.add_css_class("caption"); self.fps_lbl.add_css_class("fu-dim"); under.append(self.fps_lbl)
        left.append(under)

        # ---- session card (under the preview: it belongs to the operator, not the measurement)
        sess = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10); sess.add_css_class("fu-card")
        eyebrow = Gtk.Label(xalign=0, label="SESSION"); eyebrow.add_css_class("fu-eyebrow"); sess.append(eyebrow)
        row = Gtk.Box(spacing=8)
        self.subject = Gtk.Entry(); self.subject.set_placeholder_text("Subject (me, a friend, phone photo…)")
        self.subject.set_text("me"); self.subject.set_hexpand(True); row.append(self.subject)
        self.btn_rec = Gtk.ToggleButton(label="Record"); self.btn_rec.connect("toggled", self._on_record); row.append(self.btn_rec)
        sess.append(row)
        self.rec_lbl = Gtk.Label(xalign=0, wrap=True); self.rec_lbl.add_css_class("caption"); self.rec_lbl.add_css_class("fu-dim")
        self.rec_lbl.set_label("Not recording. Tag who is in front of the camera, then record a few seconds per pose and lighting.")
        sess.append(self.rec_lbl)
        self.summary_grid = Gtk.Grid(column_spacing=14, row_spacing=4)
        sess.append(self.summary_grid)
        srow = Gtk.Box(spacing=8)
        self.btn_copy = Gtk.Button(label="Copy summary"); self.btn_copy.connect("clicked", self._copy_summary); srow.append(self.btn_copy)
        self.copied = Gtk.Label(xalign=0); self.copied.add_css_class("caption"); self.copied.add_css_class("fu-dim"); srow.append(self.copied)
        sess.append(srow)
        left.append(sess)
        body.append(left)

        # ---- right column
        right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14)
        right.set_size_request(520, -1)
        body.append(right)

        # decision strip
        dec = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8); dec.add_css_class("fu-card")
        eb = Gtk.Label(xalign=0, label="DECISION"); eb.add_css_class("fu-eyebrow"); dec.append(eb)
        self.dec_rows = {}
        for key, name in (("strict", "Strict"), ("whitened", "Whitened"), ("whitened+liveness", "Whitened + liveness")):
            r = Gtk.Box(spacing=12)
            state = Gtk.Label(label="—", xalign=0); state.add_css_class("title-2"); state.set_size_request(118, -1)
            col = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
            nm = Gtk.Label(xalign=0, label=name); nm.add_css_class("heading")
            cap = Gtk.Label(xalign=0, wrap=True, label=labmod.POLICY_TEXT[key]); cap.add_css_class("caption"); cap.add_css_class("fu-dim")
            col.append(nm); col.append(cap); col.set_hexpand(True)
            r.append(state); r.append(col)
            dec.append(r)
            self.dec_rows[key] = state
        self.reason_lbl = Gtk.Label(xalign=0); self.reason_lbl.add_css_class("caption"); self.reason_lbl.add_css_class("fu-dim")
        dec.append(self.reason_lbl)
        right.append(dec)

        # metrics
        met = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8); met.add_css_class("fu-card")
        eb2 = Gtk.Label(xalign=0, label="MEASUREMENTS"); eb2.add_css_class("fu-eyebrow"); met.append(eb2)
        grid = Gtk.Grid(column_spacing=10, row_spacing=6)
        for c, text in enumerate(("", "raw distance", "nearest pose", "whitened / threshold")):
            h = Gtk.Label(xalign=0, label=text); h.add_css_class("caption"); h.add_css_class("fu-dim"); grid.attach(h, c, 0, 1, 1)
        self.metric_rows = {}
        for i, sp in enumerate(("ir", "rgb")):
            name = Gtk.Label(xalign=0, label=sp.upper()); name.add_css_class("heading"); grid.attach(name, 0, 1 + i, 1, 1)
            rawbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            rawval = Gtk.Label(xalign=0, label="–"); rawval.add_css_class("caption")
            rawbar = MetricBar(); rawbox.append(rawval); rawbox.append(rawbar); rawbox.set_size_request(150, -1)
            grid.attach(rawbox, 1, 1 + i, 1, 1)
            lab = Gtk.Label(xalign=0, label="–"); lab.add_css_class("caption"); lab.set_size_request(86, -1); grid.attach(lab, 2, 1 + i, 1, 1)
            wbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            wval = Gtk.Label(xalign=0, label="–"); wval.add_css_class("caption")
            wbar = MetricBar(ratio=True); wbox.append(wval); wbox.append(wbar); wbox.set_size_request(150, -1)
            grid.attach(wbox, 3, 1 + i, 1, 1)
            self.metric_rows[sp] = (rawval, rawbar, lab, wval, wbar, name)
        met.append(grid)
        vrow = Gtk.Box(spacing=8)
        vl = Gtk.Label(xalign=0, label="Liveness"); vl.add_css_class("caption"); vl.add_css_class("fu-dim"); vrow.append(vl)
        self.votes_box = Gtk.Box(spacing=6); vrow.append(self.votes_box)
        met.append(vrow)
        self.votes_caption = Gtk.Label(xalign=0, wrap=True); self.votes_caption.add_css_class("caption"); self.votes_caption.add_css_class("fu-dim")
        met.append(self.votes_caption)
        sep = Gtk.Box(); sep.add_css_class("fu-sep"); met.append(sep)
        self.scales = {}
        for key, text, lo, hi in (("certainty", "Certainty (raw gate)", 0.25, 0.60), ("hard_cap", "Hard cap (whitened path)", 0.40, 0.70)):
            srow = Gtk.Box(spacing=10)
            sl = Gtk.Label(xalign=0, label=text); sl.add_css_class("caption"); sl.set_size_request(170, -1); srow.append(sl)
            sc = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, lo, hi, 0.01); sc.set_hexpand(True); sc.set_draw_value(False)
            sc.set_value(float(self.ev.thresholds[key]) if self.ev else (0.4 if key == "certainty" else 0.55))
            val = Gtk.Label(xalign=1, label=f"{sc.get_value():.2f}"); val.add_css_class("caption"); val.set_size_request(36, -1)
            sc.connect("value-changed", self._on_scale, key, val)
            srow.append(sc); srow.append(val); met.append(srow)
            self.scales[key] = sc
        arow = Gtk.Box(spacing=8)
        self.btn_apply = Gtk.Button(label="Apply to config"); self.btn_apply.connect("clicked", self._apply_config); arow.append(self.btn_apply)
        self.apply_lbl = Gtk.Label(xalign=0); self.apply_lbl.add_css_class("caption"); self.apply_lbl.add_css_class("fu-dim"); arow.append(self.apply_lbl)
        met.append(arow)
        right.append(met)

        spacer = Gtk.Box(); spacer.set_vexpand(True); right.append(spacer)
        foot = Gtk.Box(spacing=10)
        kb = Gtk.Label(label="Esc"); kb.add_css_class("fu-kbd"); foot.append(kb)
        fl = Gtk.Label(label="close · summary is printed in the terminal"); fl.add_css_class("fu-dim"); fl.add_css_class("caption"); foot.append(fl)
        sp = Gtk.Box(); sp.set_hexpand(True); foot.append(sp)
        btn = Gtk.Button(label="Close"); btn.connect("clicked", lambda *_: self.push_key(KEY_Q)); foot.append(btn)
        right.append(foot)

        probe = Gtk.Label(); probe.add_css_class("accent"); probe.set_visible(False); right.append(probe)
        self._probe = probe
        self.win.connect("realize", lambda *_: self._read_theme())
        self._last_sample = None
        self._summary_t = 0.0
        self._n_rec = 0

    # -- theme / helpers
    def _read_theme(self) -> None:
        acc = _rgba(self._probe); fg = _rgba(self.reason_lbl, (0.5, 0.5, 0.5))
        self.preview.accent = acc
        for rawval, rawbar, lab, wval, wbar, name in self.metric_rows.values():
            rawbar.accent = wbar.accent = acc; rawbar.fg = wbar.fg = fg

    def _on_scale(self, scale, key, val_lbl) -> None:
        v = float(scale.get_value()); val_lbl.set_label(f"{v:.2f}")
        if self.ev is not None:
            self.ev.thresholds[key] = v
            if self._last_sample is not None:
                self._last_sample.decisions, self._last_sample.reason = self.ev.decide(self._last_sample)
                self._show_decisions(self._last_sample)
        self.apply_lbl.set_label("")

    def _apply_config(self, *_):
        try:
            from .engine import config as cfgmod
            cfg = cfgmod.load()
            cfg["recognition"]["certainty"] = round(float(self.scales["certainty"].get_value()), 3)
            cfg["recognition"]["hard_cap"] = round(float(self.scales["hard_cap"].get_value()), 3)
            cfgmod.save(cfg)
            self.apply_lbl.set_label(f"Saved: certainty {cfg['recognition']['certainty']}, hard cap {cfg['recognition']['hard_cap']}")
        except Exception as e:
            self.apply_lbl.set_label(f"Could not save: {e}")

    def _on_record(self, btn) -> None:
        if self.ev is None:
            return
        if btn.get_active():
            subject = self.subject.get_text().strip() or "me"
            path = self.ev.start_recording(subject)
            btn.set_label("Stop"); btn.add_css_class("destructive-action")
            self.rec_lbl.set_label(f"Recording “{subject}” → {path}")
        else:
            self.ev.stop_recording()
            btn.set_label("Record"); btn.remove_css_class("destructive-action")
            self.rec_lbl.set_label(f"Stopped. {len(self.ev.records)} samples in this session.")

    def _copy_summary(self, *_):
        if self.ev is None:
            return
        text = self.labmod.format_summary(self.ev.summary())
        try:
            self.win.get_clipboard().set(text)
            self.copied.set_label("Copied")
        except Exception as e:
            self.copied.set_label(str(e))

    def _show_decisions(self, s) -> None:
        no_face = s is None or not any(s.faces.values())
        for key, lbl in self.dec_rows.items():
            for c in ("success", "error"):
                lbl.remove_css_class(c)
            if no_face or s.raw.get(s.primary) is None and s.raw.get("rgb") is None:
                lbl.set_label("—")
            elif s.decisions.get(key):
                lbl.set_label("ACCEPT"); lbl.add_css_class("success")
            else:
                lbl.set_label("REJECT"); lbl.add_css_class("error")
        self.reason_lbl.set_label("" if s is None else s.reason)

    def _refresh_summary(self) -> None:
        grid = self.summary_grid
        child = grid.get_first_child()
        while child is not None:
            nxt = child.get_next_sibling(); grid.remove(child); child = nxt
        rows = self.ev.summary() if self.ev else []
        heads = ("subject", "n", "strict", "whitened", "w+live", "raw med", "raw min", "w/thr")
        for c, h in enumerate(heads):
            l = Gtk.Label(label=h, xalign=0 if c == 0 else 1); l.add_css_class("caption"); l.add_css_class("fu-dim"); grid.attach(l, c, 0, 1, 1)
        if not rows:
            l = Gtk.Label(label="No recorded samples yet.", xalign=0); l.add_css_class("caption"); l.add_css_class("fu-dim"); grid.attach(l, 0, 1, len(heads), 1)
            return
        fmt = lambda v, f="{:.2f}": "–" if v is None else f.format(v)
        for i, r in enumerate(rows, start=1):
            cells = (r["subject"], str(r["n"]), f"{r['strict']*100:.0f} %", f"{r['whitened']*100:.0f} %", f"{r['whitened+liveness']*100:.0f} %",
                     fmt(r["raw_median"]), fmt(r["raw_min"]), fmt(r["w_median"]))
            for c, text in enumerate(cells):
                l = Gtk.Label(label=text, xalign=0 if c == 0 else 1)
                l.add_css_class("caption"); l.add_css_class("numeric")
                grid.attach(l, c, i, 1, 1)

    # -- run: the evaluator IS the step source
    def run(self, ev) -> None:
        self.ev = ev
        lab = self.labmod

        def step():
            for k in self.keys():
                if k in (KEY_ESC, KEY_Q):
                    return None
            s = ev.step()
            if s is None:
                return {}
            view = self.view if self.view in ev.devices else ev.primary
            fr = ev.last_frames.get(view)
            if fr is None:                                   # the other camera's frame was not in this set
                obs_v = ev.last_obs.get(view) or []
                fr = obs_v[-1].frame if obs_v else None
            frame = None
            if fr is not None:
                rgb = fr.rgb
                if self.brighten.get_active() and fr.gray.mean() < 60:
                    from .engine import brightness
                    rgb = brightness.normalise_rgb(rgb, fr.gray) if view == "rgb" else \
                        np.repeat(brightness.normalise(fr.gray, float(fr.gray.mean()))[:, :, None], 3, axis=2)
                frame = rgb[:, ::-1]
            obs = ev.last_obs.get(view) or []
            box = obs[-1].box if obs and obs[-1].box is not None else None
            if box is not None and frame is not None:
                w = frame.shape[1]
                box = (w - box.x - box.w, box.y, box.w, box.h)
            return {"sample": s, "frame": frame, "box": box, "fps": ev.fps, "n_rec": len(ev.records), "view": view}

        super().run(step)

    def _apply(self, snap: dict) -> bool:
        try:
            if not snap:
                return False
            s = snap.get("sample")
            if snap.get("frame") is not None:
                self.preview.set_frame(np.ascontiguousarray(snap["frame"]))
            view = snap.get("view", s.primary if s else "ir")
            light = s.light.get(view, "") if s else ""
            self.preview.set_box(snap.get("box"), "dark" if light == "dark" else ("track" if snap.get("box") else "none"))
            self._last_sample = s
            self._show_decisions(s)
            if s is not None:
                for sp, (rawval, rawbar, lab, wval, wbar, name) in self.metric_rows.items():
                    present = sp in self.ev.devices
                    for wdg in (rawval, rawbar, lab, wval, wbar, name):
                        wdg.set_opacity(1.0 if present else 0.35)
                    gate = self.ev.thresholds["certainty" if sp == "ir" else "rgb_certainty"]
                    cap = self.ev.thresholds["hard_cap"]
                    r = s.raw.get(sp)
                    rawval.set_label("–" if r is None else f"{r:.2f}  (gate {gate:.2f})")
                    rawbar.set(r, gate, cap)
                    lab.set_label(s.label.get(sp) or "–")
                    w, thr = s.wdist.get(sp), s.wthr.get(sp)
                    if w is None or not thr:
                        wval.set_label("–" if w is None else f"{w:.2f}"); wbar.set(None)
                    else:
                        wval.set_label(f"{w:.2f} / {thr:.2f}"); wbar.set(w / thr)
                # votes
                child = self.votes_box.get_first_child()
                while child is not None:
                    nxt = child.get_next_sibling(); self.votes_box.remove(child); child = nxt
                caps = []
                for voter, res, reason in s.votes:
                    cls = {"PASS": "ok", "FAIL": "bad"}.get(res, "")
                    self.votes_box.append(_pill(f"{voter} {res.lower()}", cls))
                    if reason:
                        caps.append(f"{voter}: {reason}")
                if not s.votes:
                    self.votes_box.append(_pill("no face"))
                self.votes_caption.set_label("  ·  ".join(caps))
                light_txt = {"ok": "Face well lit", "dim": "Face a little dim", "bright": "Face a little bright",
                             "dark": "Face too dark", "blown": "Face overexposed"}.get(light, "Looking for a face" if not any(s.faces.values()) else "Face found")
                _set_pill(self.p_light, light_txt, {"ok": "ok", "dim": "warn", "bright": "warn", "dark": "bad", "blown": "bad"}.get(light, ""))
                self.pose_lbl.set_label("" if s.yaw is None else f"yaw {s.yaw:+.0f}°  pitch {s.pitch:+.0f}°  ·  primary {s.primary.upper()}")
            self.fps_lbl.set_label(f"{snap.get('fps', 0):.0f} fps")
            now = time.monotonic()
            n = snap.get("n_rec", 0)
            if self.btn_rec.get_active():
                self.rec_lbl.set_label(f"Recording “{self.ev.subject}” · {n} samples → {getattr(self.ev, 'record_path', '')}")
            if now - self._summary_t >= 1.0:
                self._summary_t = now
                self._refresh_summary()
        finally:
            self._pending = False
        return False
