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
