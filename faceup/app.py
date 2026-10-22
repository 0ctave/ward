"""The faceup application — one GTK 4 / libadwaita window for the whole chain.

Pages: Status (health checks with a Fix button each), Setup (the seven steps), Faces
(identities and conditions), Attempts (the log), Lab (the diagnostic bench), Settings
(policy, behaviour, recogniser). The app is a DRIVER over the command line: read-only data
comes from `health.collect()` / the store / the log, and every action runs the matching
`faceup …` command as a subprocess whose output streams into the console pane. The CLI
therefore stays complete on its own, and the two can never disagree.

Privileges: setup actions need root (store, config, PAM, keyring). Launch with `sudo faceup app`
or from the launcher (`pkexec faceup app`, polkit action org.faceup.app). Without root the app
still shows status and the log, and says what it cannot do.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
CONDITIONS = ("glasses", "sunglasses", "headset", "hat", "beard")   # night needs none: IR covers the dark


def strip_ansi(s: str) -> str:
    return ANSI.sub("", s)


def faceup_cmd(argv: list[str]) -> list[str]:
    """The command line for `faceup …` run by this same interpreter and package."""
    return [sys.executable, "-m", "faceup", *argv]


def child_env() -> dict:
    env = dict(os.environ)
    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = pkg_parent + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("NO_COLOR", "1")
    return env


def main(argv=None) -> int:
    import gi
    gi.require_version("Gtk", "4.0"); gi.require_version("Adw", "1")
    from gi.repository import Adw, Gdk, GLib, Gtk, Pango  # noqa: F401
    from .display import user_wayland_env
    for k, v in user_wayland_env().items():
        os.environ.setdefault(k, v)
    os.environ.setdefault("GSK_RENDERER", "ngl")
    if os.geteuid() == 0:
        os.environ.setdefault("GTK_A11Y", "none")
    from . import appui  # CSS + shared widgets
    from . import health
    from .engine import config as cfgmod, policy as polmod

    def adw_at_least(minor: int) -> bool:
        return Adw.get_major_version() > 1 or Adw.get_minor_version() >= minor

    class _Dialog(Gtk.Window):
        """Adw.MessageDialog's interface for libadwaita < 1.2 (heading, body, extra child, responses)."""

        def __init__(self, parent, heading, body):
            super().__init__(transient_for=parent, modal=True, title=heading, resizable=False)
            self._handlers, self._default = [], None
            box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
            for side_ in ("top", "bottom", "start", "end"):
                getattr(box, f"set_margin_{side_}")(18)
            h = Gtk.Label(label=heading, xalign=0); h.add_css_class("title-3"); box.append(h)
            box.append(Gtk.Label(label=body, xalign=0, wrap=True))
            self._extra = Gtk.Box(); box.append(self._extra)
            self._buttons = Gtk.Box(spacing=8, halign=Gtk.Align.END); box.append(self._buttons)
            self.set_child(box)

        def set_extra_child(self, w):
            self._extra.append(w)

        def add_response(self, rid, label):
            b = Gtk.Button(label=label); b.connect("clicked", lambda *_: self._respond(rid)); self._buttons.append(b)
            b._rid = rid

        def set_response_appearance(self, rid, _appearance):
            for b in list(self._iter_buttons()):
                if b._rid == rid:
                    b.add_css_class("suggested-action")

        def set_default_response(self, rid):
            self._default = rid

        def _iter_buttons(self):
            c = self._buttons.get_first_child()
            while c is not None:
                yield c; c = c.get_next_sibling()

        def connect(self, sig, cb, *a):
            if sig == "response":
                self._handlers.append(cb); return 0
            return super().connect(sig, cb, *a)

        def _respond(self, rid):
            for cb in self._handlers:
                cb(self, rid)
            self.close()

    def message_dialog(parent, heading, body):
        if adw_at_least(2):
            return Adw.MessageDialog(transient_for=parent, heading=heading, body=body)
        return _Dialog(parent, heading, body)

    class Console(Gtk.Box):
        """Where the commands' output goes."""

        def __init__(self):
            super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=6)
            head = Gtk.Box(spacing=8)
            self.title = Gtk.Label(xalign=0, label="Console"); self.title.add_css_class("fu-eyebrow"); head.append(self.title)
            self.state = appui._pill("idle"); head.append(self.state)
            sp = Gtk.Box(); sp.set_hexpand(True); head.append(sp)
            clear = Gtk.Button(label="Clear"); clear.add_css_class("flat"); clear.connect("clicked", lambda *_: self.buf.set_text("")); head.append(clear)
            self.append(head)
            self.view = Gtk.TextView(); self.view.set_editable(False); self.view.set_monospace(True); self.view.set_cursor_visible(False)
            self.view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR); self.buf = self.view.get_buffer()
            sc = Gtk.ScrolledWindow(); sc.set_child(self.view); sc.set_min_content_height(150); sc.set_vexpand(True)
            sc.add_css_class("fu-surface"); self.append(sc)

        def write(self, text: str) -> None:
            end = self.buf.get_end_iter(); self.buf.insert(end, text)
            self.view.scroll_to_mark(self.buf.get_insert(), 0.0, True, 0.0, 1.0)

    class Runner:
        """Runs one `faceup …` command at a time; streams output to the console; callback at the end."""

        def __init__(self, console: Console):
            self.console = console; self.proc = None

        @property
        def busy(self) -> bool:
            return self.proc is not None and self.proc.poll() is None

        def run(self, argv: list[str], stdin_text: str | None = None, on_done=None) -> bool:
            if self.busy:
                self.console.write("… a command is still running\n"); return False
            self.console.write(f"$ faceup {' '.join(argv)}\n")
            appui._set_pill(self.console.state, "running", "accent")
            try:
                self.proc = subprocess.Popen(faceup_cmd(argv), stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
                                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=child_env(), bufsize=1)
            except OSError as e:
                self.console.write(f"cannot start: {e}\n"); appui._set_pill(self.console.state, "error", "bad"); return False
            if stdin_text is not None:
                try:
                    self.proc.stdin.write(stdin_text + "\n"); self.proc.stdin.close()
                except OSError:
                    pass

            def reader(proc=self.proc):
                for line in proc.stdout:
                    GLib.idle_add(self.console.write, strip_ansi(line))
                rc = proc.wait()
                def finish():
                    appui._set_pill(self.console.state, "done" if rc == 0 else f"exit {rc}", "ok" if rc == 0 else "warn")
                    if on_done:
                        on_done(rc)
                    return False
                GLib.idle_add(finish)
            threading.Thread(target=reader, daemon=True).start()
            return True

    def level_icon(level: str) -> str:
        return {"ok": "emblem-ok-symbolic", "warn": "dialog-warning-symbolic", "fail": "dialog-error-symbolic"}.get(level, "dialog-information-symbolic")

    class App(Adw.Application):
        def __init__(self):
            super().__init__(application_id="org.faceup.App")
            self.connect("activate", self.on_activate)
            self.report = None

        # ---- window
        def on_activate(self, *_):
            appui._load_css()
            self.win = Adw.ApplicationWindow(application=self, title="faceup")
            self.win.set_default_size(1120, 760)
            self.console = Console(); self.runner = Runner(self.console)
            root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
            header = Adw.HeaderBar(); header.set_title_widget(Adw.WindowTitle(title="faceup", subtitle="face unlock for Linux"))
            refresh = Gtk.Button(icon_name="view-refresh-symbolic"); refresh.set_tooltip_text("Refresh"); refresh.connect("clicked", lambda *_: self.refresh())
            header.pack_end(refresh)
            root.append(header)
            if os.geteuid() != 0:
                msg = "Not running as root: status and log only. For setup actions run  sudo faceup app"
                if adw_at_least(3):
                    banner = Adw.Banner(title=msg); banner.set_revealed(True)
                else:                                        # libadwaita < 1.3 (Ubuntu 22.04 ships 1.1)
                    banner = Gtk.Label(label=msg, wrap=True); banner.add_css_class("warning")
                    for side_ in ("top", "bottom"):
                        getattr(banner, f"set_margin_{side_}")(6)
                root.append(banner)
            self.stack = Gtk.Stack(); self.stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
            side = Gtk.StackSidebar(); side.set_stack(self.stack); side.set_size_request(190, -1)
            body = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
            body.append(side)
            content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
            scroller = Gtk.ScrolledWindow(); scroller.set_child(self.stack); scroller.set_vexpand(True)
            content.append(scroller)
            self.console.set_margin_start(18); self.console.set_margin_end(18); self.console.set_margin_bottom(12)
            content.append(self.console)
            content.set_hexpand(True)
            body.append(content); body.set_vexpand(True)
            root.append(body)
            self.win.set_content(root)
            self.pages = {}
            for key, title in (("status", "Status"), ("setup", "Setup"), ("faces", "Faces"), ("attempts", "Attempts"), ("lab", "Lab"), ("settings", "Settings")):
                page = Adw.PreferencesPage(); page.set_margin_top(6)
                self.pages[key] = page; self.stack.add_titled(page, key, title)
            self.win.present()
            self.refresh()

        # ---- data
        def refresh(self):
            def work():
                try:
                    rep = health.collect(probe_cameras=False)
                except Exception as e:
                    rep = {"sections": [health.Section("error", "Error", [health.Item("fail", str(e))])], "identities": [], "conditions": [], "policy": "", "attempts": {}, "user": "?", "profile": "-", "calibrated": None, "root": os.geteuid() == 0}
                GLib.idle_add(self.render, rep)
            threading.Thread(target=work, daemon=True).start()

        def _clear(self, page):
            child = page.get_first_child()
            # PreferencesPage holds groups; remove them all
            groups = list(getattr(self, "_groups", {}).get(page, []))
            for g in groups:
                page.remove(g)
            self._groups = getattr(self, "_groups", {}); self._groups[page] = []

        def _group(self, page, title, desc=""):
            g = Adw.PreferencesGroup(title=title, description=desc)
            page.add(g); self._groups.setdefault(page, []).append(g); return g

        def _row(self, group, title, subtitle="", icon=None, button=None, on_click=None, suffix_widget=None):
            r = Adw.ActionRow(title=title, subtitle=subtitle)
            if icon:
                r.add_prefix(Gtk.Image.new_from_icon_name(icon))
            if suffix_widget is not None:
                r.add_suffix(suffix_widget)
            if button:
                b = Gtk.Button(label=button); b.set_valign(Gtk.Align.CENTER); b.add_css_class("flat" if button in ("Open", "Refresh") else "suggested-action")
                if on_click:
                    b.connect("clicked", lambda *_: on_click())
                r.add_suffix(b)
            group.add(r); return r

        def run_cmd(self, argv, stdin_text=None, then_refresh=True):
            self.runner.run(argv, stdin_text, on_done=(lambda rc: self.refresh()) if then_refresh else None)

        def ask_text(self, title, body, placeholder, cb, password=False):
            dlg = message_dialog(self.win, title, body)
            entry = Gtk.PasswordEntry() if password else Gtk.Entry()
            if not password:
                entry.set_placeholder_text(placeholder)
            entry.set_margin_top(8); dlg.set_extra_child(entry)
            dlg.add_response("cancel", "Cancel"); dlg.add_response("ok", "Continue"); dlg.set_response_appearance("ok", Adw.ResponseAppearance.SUGGESTED)
            dlg.set_default_response("ok")
            def on_resp(d, resp):
                if resp == "ok" and entry.get_text().strip():
                    cb(entry.get_text().strip())
            dlg.connect("response", on_resp); dlg.present()

        def ask_choice(self, title, body, choices, cb):
            dlg = message_dialog(self.win, title, body)
            for c in choices:
                dlg.add_response(c, c.capitalize())
            dlg.add_response("other", "Other…"); dlg.add_response("cancel", "Cancel")
            def on_resp(d, resp):
                if resp == "other":
                    self.ask_text(title, "Name the condition", "e.g. sunglasses", cb)
                elif resp not in ("cancel",):
                    cb(resp)
            dlg.connect("response", on_resp); dlg.present()

        # ---- render
        def render(self, rep):
            self.report = rep
            root = rep.get("root", False)
            try:
                self._render(rep, root)
            finally:
                if getattr(self, "selftest", False):
                    n = sum(len(s.items) for s in rep["sections"])
                    print(f"selftest: rendered {len(rep['sections'])} sections / {n} items, pages: {', '.join(self.pages)}", flush=True)
                    GLib.timeout_add(300, lambda: (self.quit(), False)[1])

        def _render(self, rep, root):
            # Status
            p = self.pages["status"]; self._clear(p)
            hdr = self._group(p, f"Profile {rep.get('profile', '-')} · {rep.get('user', '')}",
                              f"{health.problems(rep)} thing(s) to fix" if health.problems(rep) else "Everything looks fine")
            for sec in rep["sections"]:
                g = self._group(p, sec.title)
                for it in sec.items:
                    self._row(g, it.text, "", level_icon(it.level), it.fix_label or ("Fix" if it.fix else None) if (it.fix and root) else None,
                              (lambda fix=it.fix: self.run_cmd(fix)) if it.fix else None)
            # Setup
            p = self.pages["setup"]; self._clear(p)
            g = self._group(p, "Set up face unlock", "Seven steps, each one reversible. Or run  sudo faceup setup  in a terminal.")
            emit_known = any(s.key == "emitter" and s.items and s.items[0].level == "ok" for s in rep["sections"])
            has_ir = rep.get("profile") in ("A", "B")
            steps = [
                ("1 · Cameras", f"profile {rep.get('profile', '-')}; setup picks the IR and RGB pair by stream format", "Detect", ["setup", "--cameras-only"]),
                ("2 · IR emitter", "find the LED control by measurement, then benchmark its switch (enables the shadow-test voter)", "Probe + bench", ["emitter", "probe"]),
                ("3 · Recogniser", "dlib (distro) or ArcFace via OpenCV — tighter identity clusters", "Install ArcFace", ["recogniser", "install", "-y"]),
                ("4 · Enrol", "one guided window in your usual light: enrols, sets the gate" + (", also works in the dark" if has_ir else "") + "; glasses are offered at the end", "Enrol", ["enroll", "--fresh", "--selfcheck"]),
                ("5 · Keyring", "a face login unlocks GNOME Keyring / KWallet (login password sealed by TPM / encrypted disk)", "Enable", None),
                ("6 · PAM", "one line in the shared auth stack (common-auth / system-auth); the password path is never touched", "Enable", ["pam", "enable"]),
                ("7 · Policy", f"currently {rep.get('policy', '')}", "Choose", None),
            ]
            for title, sub, label, cmd in steps:
                if title.startswith("2") and not has_ir:
                    continue                                 # no IR camera: no emitter to probe
                if title.startswith("5"):
                    self._row(g, title, sub, None, label if root else None, lambda: self.keyring_enable())
                elif title.startswith("7"):
                    self._row(g, title, sub, None, "Choose" if root else None, lambda: self.stack.set_visible_child_name("settings"))
                elif title.startswith("2"):
                    def probe_then_bench():
                        self.runner.run(["emitter", "probe"], on_done=lambda rc: self.run_cmd(["emitter", "toggle-bench"]) if rc == 0 else self.refresh())
                    self._row(g, title, sub, None, label if root else None, probe_then_bench)
                else:
                    self._row(g, title, sub, None, label if root else None, (lambda c=cmd: self.run_cmd(c)) if cmd else None)
            # Faces
            p = self.pages["faces"]; self._clear(p)
            g = self._group(p, "Who can unlock this login", "Any registered face unlocks; the log says who.")
            ids = rep.get("identities", [])
            if not ids:
                self._row(g, "Nobody enrolled yet", "", "dialog-warning-symbolic", "Enrol" if root else None, lambda: self.run_cmd(["enroll", "--fresh", "--selfcheck"]))
            for name in ids:
                mine = name == rep.get("user")
                self._row(g, name + (" (you)" if mine else ""), "calibrated " + (rep.get("calibrated") or "never") if mine else "named identity", "avatar-default-symbolic",
                          "Re-enrol" if root else None, (lambda n=name, m=mine: self.run_cmd(["enroll", "--fresh"] if m else ["enroll", "--fresh", "--identity", n])))
                if not mine and root:
                    self._row(g, f"Remove {name}", "", "user-trash-symbolic", "Remove", lambda n=name: self.run_cmd(["models", "--remove-identity", n]))
            if root:
                self._row(g, "Add another person", "registers a new named identity with the same guided window", "list-add-symbolic", "Add…",
                          lambda: self.ask_text("Add a person", "Their name (any registered face unlocks this login)", "Alice", lambda n: self.run_cmd(["enroll", "--identity", n])))
            g = self._group(p, "Conditions", "A short four-pose sweep on top of the enrolment — glasses, headset… (the dark needs none with an IR camera). Your calibrated gate is never loosened by these.")
            for c in rep.get("conditions", []):
                self._row(g, c, "", "emblem-ok-symbolic", "Remove" if root else None, lambda cc=c: self.run_cmd(["models", "--remove-condition", cc]))
            if root:
                self._row(g, "Add a condition", "e.g. with glasses or a headset", "list-add-symbolic", "Add…",
                          lambda: self.ask_choice("Add a condition", "Wear it / set it up first, then start.", list(CONDITIONS), lambda c: self.run_cmd(["enroll", "--add", c])))
                self._row(g, "Recalibrate thresholds", "eight guided passes derive the gate from your real distances", "view-refresh-symbolic", "Calibrate", lambda: self.run_cmd(["calibrate"]))
            # Attempts
            p = self.pages["attempts"]; self._clear(p)
            sm = rep.get("attempts") or {}
            g = self._group(p, "Recent unlocks", (f"{sm.get('accepted', 0)}/{sm.get('attempts', 0)} accepted · median {sm.get('median_accept_ms')} ms" if sm else "nothing logged yet"))
            from . import attempts as att
            for r in reversed(att.read(cfgmod.LOG_PATH, 30)):
                what = "accept" if r["accepted"] else r.get("result", "?")
                extra = " · ".join(x for x in ("night (IR only)" if r.get("dark") else "", f"ir {r['ir']:.2f}" if r.get("ir") is not None else "", f"rgb {r['rgb']:.2f}" if r.get("rgb") is not None else "",
                                                " ".join(r["votes"]) if r["votes"] else "", f"{r['ms']} ms" if r["ms"] else "", r["service"], r.get("detail", "") if not (r["service"] or r.get("dark")) else "") if x)
                self._row(g, f"{r['time'][11:]}  {what}", extra, "emblem-ok-symbolic" if r["accepted"] else "dialog-error-symbolic")
            self._row(g, "Summary in the terminal", "faceup log --summary", None, "Open", lambda: self.run_cmd(["log", "--summary"], then_refresh=False))
            # Lab
            p = self.pages["lab"]; self._clear(p)
            g = self._group(p, "The lab", "Live evaluation instrument: every distance the engine sees, the liveness votes, and the decision under three policies on the same frames. Record yourself, a friend, a phone photo — get accept rates per subject. This is how you diagnose and improve faceup.")
            self._row(g, "Open the lab window", "sudo faceup lab", "utilities-system-monitor-symbolic", "Open" if root else None, lambda: self.run_cmd(["lab"], then_refresh=False))
            self._row(g, "Record frames for replay", "saves every frame set as PNGs under the lab dir: a regression corpus that runs without hardware", "media-record-symbolic", "Open" if root else None,
                      lambda: self.run_cmd(["lab", "--record-frames", os.path.join(os.environ.get("FACEUP_ETC", "/var/lib/faceup"), "lab", "frames")], then_refresh=False))
            self._row(g, "Replay a recording through the engine", "faceup auth --replay DIR -v", "media-playback-start-symbolic", "Open" if root else None,
                      lambda: self.ask_text("Replay", "Directory with index.json and the PNG frames", "/var/lib/faceup/lab/frames", lambda d: self.run_cmd(["auth", "--replay", d, "-v"], then_refresh=False)))
            self._row(g, "Text mode", "faceup lab --text  (SSH, servers)", "utilities-terminal-symbolic")
            # Settings
            p = self.pages["settings"]; self._clear(p)
            cfg = cfgmod.load()
            g = self._group(p, "Security policy", "Presets only ever tighten the calibrated gate. Per-service overrides below.")
            for lvl in ("convenience", "balanced", "strict"):
                self._row(g, lvl, polmod.describe(lvl), "emblem-ok-symbolic" if rep.get("policy") == lvl else None,
                          "Use" if (root and rep.get("policy") != lvl) else None, lambda l=lvl: self.run_cmd(["tune", "policy", l]))
            for svc, key in (("sudo", "policy_sudo"), ("polkit", "policy_polkit"), ("login", "policy_login"), ("unlock", "policy_unlock")):
                cur = cfg.get("security", {}).get(key, "") or "default"
                dd = Gtk.DropDown.new_from_strings(["default", "convenience", "balanced", "strict", "off"])
                dd.set_selected(["default", "convenience", "balanced", "strict", "off"].index(cur)); dd.set_valign(Gtk.Align.CENTER)
                dd.set_sensitive(root)
                dd.connect("notify::selected", lambda d, _p, k=key: self.run_cmd(["tune", f"security.{k}", "" if d.get_selected_item().get_string() == "default" else d.get_selected_item().get_string()], then_refresh=False))
                self._row(g, f"{svc}", f"security.{key}", None, None, None, dd)
            g = self._group(p, "Behaviour")
            for key, title, sub in (("behaviour.notify", "Desktop notification", "accept | all | off"), ("behaviour.skip_login", "First login uses the password", "keeps the keyring unlockable without the native module"),
                                    ("behaviour.tty_login", "Face unlock on text consoles", "off by default"), ("recognition.recogniser", "Recogniser", "dlib | arcface (re-enrol after switching)")):
                sec_, k = key.split(".")
                val = str(cfg[sec_].get(k, "")).lower()
                self._row(g, title, f"{key} = {val}   ·   {sub}", None, "Change" if root else None,
                          lambda kk=key, vv=val: self.ask_text(kk, "New value", vv, lambda v: self.run_cmd(["tune", kk, v])))
            g = self._group(p, "Maintenance")
            if root:
                self._row(g, "Forget learned models", "drops models added by online adaptation; enrolled ones stay", "edit-clear-symbolic", "Forget", lambda: self.run_cmd(["models", "--forget-learned"]))
                self._row(g, "Show the config", "faceup tune", "document-properties-symbolic", "Open", lambda: self.run_cmd(["tune"], then_refresh=False))
                self._row(g, "Doctor (terminal report)", "faceup doctor", "utilities-terminal-symbolic", "Open", lambda: self.run_cmd(["doctor"], then_refresh=False))

        def keyring_enable(self):
            self.ask_text("Unlock the keyring by face", "Your login password is verified and stored sealed (encrypted disk / TPM). It is released only after a face accept.", "",
                          lambda pw: self.run_cmd(["keyring", "enable", "--password-stdin"], stdin_text=pw), password=True)

    app = App()
    args = list(argv if argv is not None else sys.argv[:1])
    app.selftest = "--selftest" in args
    return app.run([a for a in args if a != "--selftest"])
