#!/usr/bin/env python3
"""Render Ward's app pages to PNG, headless and face-free (no camera, no personal data).

    python3 docs/assets/screenshots.py                 # all face-free pages → docs/assets/
    python3 docs/assets/screenshots.py --theme light --pages overview,reliability

It drives the real `ward.app` window with sample data and snapshots each page's widget tree, so the
screenshots always match the current UI. Only pages without a camera preview are rendered (the
register/setup-camera page is deliberately excluded). Regenerate after UI changes and commit the PNGs.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

PAGES_FACE_FREE = ("overview", "reliability", "settings", "advanced")


def sample_report(user: str = "you"):
    """A realistic, invented status report (profile A, face unlock on, a healthy log). No real data."""
    from ward import health
    from ward.engine import config as cfgmod
    cfg = cfgmod.load()
    cfg["cameras"]["ir"] = "usb:0001:0001:02"; cfg["cameras"]["rgb"] = "usb:0001:0001:00"
    now = "2026-10-25T09:%02d:%02d"
    rows = []
    for i in range(22):
        acc = i not in (7, 15)                         # a couple of misses for a realistic rate
        rows.append({"time": now % (10 + i % 40, i * 2 % 60), "accepted": acc,
                     "result": "accept" if acc else "no-match", "reason": "accept" if acc else "no-match",
                     "who": "", "dark": i % 5 == 0, "ms": 780 + (i * 37) % 600 if acc else None,
                     "service": ["sudo", "gdm-password", "polkit-1"][i % 3],
                     "ir": round(0.12 + (i % 7) * 0.02, 2) if acc else 0.44,
                     "rgb": round(0.18 + (i % 5) * 0.02, 2) if (acc and i % 2) else None,
                     "votes": ["cross:PASS"] + (["shadow:PASS"] if i % 3 == 0 else [])})
    sections = [
        health.Section("cameras", "Cameras", [health.Item("ok", "/dev/video2 IR (greyscale) 640x360"),
                                              health.Item("info", "/dev/video0 RGB 1920x1080")]),
        health.Section("engine", "Engine", [health.Item("ok", "recogniser: dlib ResNet · gate 0.38"),
                                            health.Item("ok", "liveness for a still face: cross-spectrum, shadow test")]),
        health.Section("access", "Login, sudo, keyring", [health.Item("ok", "face unlock enabled for login, sudo and polkit (native module)"),
                                                          health.Item("ok", "a face login unlocks the keyring")]),
    ]
    return {"user": user, "profile": "A", "sections": sections, "identities": ["", "Alex"],
            "conditions": ["glasses"], "policy": "balanced", "calibrated": "2026-10-25",
            "attempts": {}, "root": False, "cfg": cfg, "rows": rows, "pam": True,
            "daemon": {"installed": True, "active": True}}


def sample_people(user: str = "you"):
    return [{"identity": "", "name": user, "owner": True, "models": {"ir": 6, "rgb": 6},
             "learned": 3, "conditions": ["glasses"], "enrolled": "2026-10-25",
             "accepts": 128, "last_seen": "2026-10-25T09:12:03"},
            {"identity": "Alex", "name": "Alex", "owner": False, "models": {"ir": 6},
             "learned": 0, "conditions": [], "enrolled": "2026-10-24", "accepts": 11, "last_seen": "2026-10-25T08:40:00"}]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Render Ward's face-free app pages to PNG.")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--pages", default=",".join(PAGES_FACE_FREE), help="comma-separated: " + ",".join(PAGES_FACE_FREE))
    ap.add_argument("--theme", choices=["dark", "light"], default="dark")
    ap.add_argument("--width", type=int, default=1040)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--user", default="you")
    a = ap.parse_args(argv)
    pages = [p.strip() for p in a.pages.split(",") if p.strip() in PAGES_FACE_FREE]
    os.makedirs(a.out, exist_ok=True)

    from ward import app as A
    A.priv_cmd = lambda argv, root=None, pkg_parent=None: A.ward_cmd(argv)   # no pkexec in the harness
    import gi
    gi.require_version("Gtk", "4.0"); gi.require_version("Adw", "1")
    from gi.repository import Adw, Gio, GLib, Graphene, Gtk

    rep = sample_report(a.user)
    _init = Adw.Application.__init__

    def hook(self, *args, **kw):
        _init(self, *args, **kw)
        self.set_flags(self.get_flags() | Gio.ApplicationFlags.NON_UNIQUE)
        sm = Adw.StyleManager.get_default()
        sm.set_color_scheme(Adw.ColorScheme.FORCE_DARK if a.theme == "dark" else Adw.ColorScheme.FORCE_LIGHT)

        def go():
            self.people = sample_people(a.user)
            self.report = rep
            self.win.set_default_size(a.width, a.height)
            import time
            ctx = GLib.MainContext.default()
            for key in pages:
                getattr(self, "_" + key)(rep)
                self.stack.set_visible_child_name(key)
                for _ in range(45):
                    ctx.iteration(False); time.sleep(0.02)
                w = self.win; W, H = w.get_width(), w.get_height()
                s = Gtk.Snapshot(); Gtk.WidgetPaintable.new(w).snapshot(s, W, H)
                node = s.to_node()
                if node is None:
                    print(f"  ! {key}: nothing to snapshot", file=sys.stderr); continue
                tex = w.get_renderer().render_texture(node, Graphene.Rect().init(0, 0, W, H))
                path = os.path.join(a.out, f"app-{key}.png")
                tex.save_to_png(path); print("wrote", path)
            GLib.timeout_add(300, lambda: (self.quit(), False)[1])
            return False
        self.connect("activate", lambda *_: GLib.timeout_add(600, go))

    Adw.Application.__init__ = hook
    return A.main([sys.argv[0]])


if __name__ == "__main__":
    sys.exit(main())
