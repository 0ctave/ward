"""The ward manager's GTK 4 / libadwaita window classes (see ward/app.py for the design).

Imported only by `app.main()` after it has prepared the environment (the invoking user's
Wayland socket, GSK_RENDERER, GTK_A11Y for root), because importing this module imports GTK.
ward/app.py itself stays GTK-free so tests can import its helpers anywhere.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from types import SimpleNamespace

import gi
gi.require_version("Gtk", "4.0"); gi.require_version("Adw", "1")
from gi.repository import Adw, GLib, Gtk  # noqa: E402

from . import appdata, appui, attempts as att, health  # noqa: E402,F401
from .app import (  # noqa: E402,F401
    CAMERA_CMDS, CONDITIONS, ENROL, PROFILE_TEXT, child_env, describe_stats, setup_steps, ward_cmd, priv_cmd, strip_ansi,
)
from .engine import config as cfgmod, pam as pammod, policy as polmod  # noqa: E402,F401

ROOT = os.geteuid() == 0


def adw_at_least(minor: int) -> bool:
    return Adw.get_major_version() > 1 or Adw.get_minor_version() >= minor

# Adw.ResponseAppearance exists only from libadwaita 1.2; the < 1.2 _Dialog takes plain strings.


RA = Adw.ResponseAppearance if hasattr(Adw, "ResponseAppearance") else SimpleNamespace(
    DEFAULT="default", SUGGESTED="suggested", DESTRUCTIVE="destructive")


class _Dialog(Gtk.Window):
    """Adw.MessageDialog's interface for libadwaita < 1.2 (heading, body, extra child, responses)."""

    def __init__(self, parent, heading, body):
        super().__init__(transient_for=parent, modal=True, title=heading, resizable=False)
        self._handlers = []
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

    def set_response_appearance(self, rid, appearance):
        c = self._buttons.get_first_child()
        while c is not None:
            if c._rid == rid:
                c.add_css_class("destructive-action" if appearance == RA.DESTRUCTIVE else "suggested-action")
            c = c.get_next_sibling()

    def set_default_response(self, rid):
        pass

    def set_response_enabled(self, rid, on):
        c = self._buttons.get_first_child()
        while c is not None:
            if c._rid == rid:
                c.set_sensitive(on)
            c = c.get_next_sibling()

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
        sc = Gtk.ScrolledWindow(); sc.set_child(self.view); sc.set_min_content_height(170); sc.set_vexpand(True)
        sc.add_css_class("fu-surface"); self.append(sc)

    def write(self, text: str) -> None:
        end = self.buf.get_end_iter(); self.buf.insert(end, text)
        self.view.scroll_to_mark(self.buf.get_insert(), 0.0, True, 0.0, 1.0)


class Runner:
    """Runs one `ward …` command at a time; streams output to the console; callback at the end.
    `capture=True` collects the output for the caller instead (JSON reads)."""

    def __init__(self, console: Console, on_start=None):
        self.console = console; self.proc = None; self.on_start = on_start

    @property
    def busy(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def run(self, argv, privileged=True, stdin_text=None, on_done=None, capture=False) -> bool:
        if self.busy:
            self.console.write("… a command is still running\n"); return False
        cmd = priv_cmd(argv) if privileged else ward_cmd(argv)
        self.console.write(f"$ {'sudo ' if privileged and not ROOT else ''}ward {' '.join(argv)}\n")
        appui._set_pill(self.console.state, "running", "accent")
        if self.on_start and not capture:
            self.on_start()
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT if not capture else subprocess.PIPE,
                                         text=True, env=child_env(), bufsize=1)
        except OSError as e:
            self.console.write(f"cannot start: {e}\n"); appui._set_pill(self.console.state, "error", "bad"); return False
        if stdin_text is not None:
            try:
                self.proc.stdin.write(stdin_text + "\n"); self.proc.stdin.close()
            except OSError:
                pass

        def reader(proc=self.proc):
            out = []
            for line in proc.stdout:
                if capture:
                    out.append(line)
                else:
                    GLib.idle_add(self.console.write, strip_ansi(line))
            err = proc.stderr.read() if capture else ""
            rc = proc.wait()

            def finish():
                if rc in (126, 127) and privileged and not ROOT:
                    self.console.write("administrator authorisation was cancelled or refused\n")
                elif capture and rc != 0:
                    self.console.write(strip_ansi("".join(out) + err))
                appui._set_pill(self.console.state, "done" if rc == 0 else f"exit {rc}", "ok" if rc == 0 else "warn")
                if on_done:
                    on_done(rc, "".join(out)) if capture else on_done(rc)
                return False
            GLib.idle_add(finish)
        threading.Thread(target=reader, daemon=True).start()
        return True


def level_icon(level: str) -> str:
    return {"ok": "emblem-ok-symbolic", "warn": "dialog-warning-symbolic", "fail": "dialog-error-symbolic"}.get(level, "dialog-information-symbolic")


VERDICT = {"good": ("Reliable", "ok"), "fair": ("Sometimes fails", "warn"), "poor": ("Unreliable", "bad"), "unknown": ("Not enough data", "")}


class App(Adw.Application):
    def __init__(self):
        super().__init__(application_id="org.ward.App")
        self.connect("activate", self.on_activate)
        self.report = None
        self.people = None                 # None = not loaded (needs admin), else appdata.people() records
        self._groups = {}

    # ---- window
    def on_activate(self, *_):
        appui._load_css()
        self.win = Adw.ApplicationWindow(application=self, title="ward")
        self.win.set_default_size(1040, 780)
        self.console = Console()
        self.runner = Runner(self.console, on_start=lambda: self.console_btn.set_active(True))
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        header = Adw.HeaderBar(); header.set_title_widget(Adw.WindowTitle(title="Ward", subtitle="face unlock"))
        refresh = Gtk.Button(icon_name="view-refresh-symbolic"); refresh.set_tooltip_text("Refresh"); refresh.connect("clicked", lambda *_: self.refresh(people=self.people is not None))
        header.pack_end(refresh)
        self.console_btn = Gtk.ToggleButton(icon_name="utilities-terminal-symbolic"); self.console_btn.set_tooltip_text("Show the command output")
        header.pack_end(self.console_btn)
        root.append(header)
        self.stack = Gtk.Stack(); self.stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
        side = Gtk.StackSidebar(); side.set_stack(self.stack); side.set_size_request(180, -1); self.side = side
        body = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        body.append(side)
        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL); content.set_hexpand(True)
        scroller = Gtk.ScrolledWindow(); scroller.set_child(self.stack); scroller.set_vexpand(True)
        content.append(scroller)
        rev = Gtk.Revealer(); rev.set_transition_type(Gtk.RevealerTransitionType.SLIDE_UP)
        self.console.set_margin_start(18); self.console.set_margin_end(18); self.console.set_margin_bottom(12)
        rev.set_child(self.console); rev.set_vexpand(False)       # the console's vexpand must not steal the page's height
        self.console_btn.bind_property("active", rev, "reveal-child", 0)
        content.append(rev)
        body.append(content); body.set_vexpand(True)
        root.append(body)
        self.toast = Adw.ToastOverlay(); self.toast.set_child(root)
        self.win.set_content(self.toast)
        self.pages = {}
        for key, title in (("overview", "Overview"), ("people", "People"), ("reliability", "Reliability"), ("settings", "Settings"), ("advanced", "Advanced")):
            page = Adw.PreferencesPage(); page.set_margin_top(6)
            self.pages[key] = page; self.stack.add_titled(page, key, title)
        self.win.present()
        self.refresh(people=ROOT)

    # ---- data
    def refresh(self, people=False):
        """Re-read status and log (unprivileged); with people=True also the face store."""
        def work():
            try:
                rep = health.collect(probe_cameras=False)
            except Exception as e:
                rep = {"sections": [health.Section("error", "Error", [health.Item("fail", str(e))])], "identities": [], "conditions": [],
                       "policy": "", "attempts": {}, "user": appdata.invoking_user(), "profile": "-", "calibrated": None, "root": ROOT}
            try:
                rep["rows"] = att.read(cfgmod.LOG_PATH, 500)
            except Exception:
                rep["rows"] = []
            try:
                from .cli import daemon_state
                rep["daemon"] = daemon_state()           # systemctl calls: off the GTK thread
            except Exception:
                rep["daemon"] = None
            try:
                rep["cfg"] = cfgmod.load()
            except Exception:
                rep["cfg"] = {"security": {}, "behaviour": {}, "recognition": {}}
            try:
                rep["pam"] = bool(pammod.status()["enabled"])
            except Exception:
                rep["pam"] = False
            GLib.idle_add(self.render, rep)
        threading.Thread(target=work, daemon=True).start()
        if people:
            self.load_people()

    def load_people(self):
        def done(rc, out):
            try:
                data = json.loads(out.strip().splitlines()[-1]) if rc == 0 else None
            except (ValueError, IndexError):
                data = None
            if not (isinstance(data, dict) and isinstance(data.get("people"), list)):
                data = None                        # an unexpected reply reads like no reply
            if data is not None:
                self.people = data["people"]; self.calibrated = data.get("calibrated", "")
            elif self.people is None and rc == 1:
                self.people = []                   # no store yet / store unreadable: show the empty state
            if self.report:
                self.render(self.report)
        self.runner.run(["models", "--json"], capture=True, on_done=done)

    # ---- widgets
    def _clear(self, page):
        for g in self._groups.get(page, []):
            page.remove(g)
        self._groups[page] = []

    def _group(self, page, title, desc=""):
        g = Adw.PreferencesGroup(title=GLib.markup_escape_text(title), description=GLib.markup_escape_text(desc))
        page.add(g); self._groups.setdefault(page, []).append(g); return g

    def _button(self, label, on_click, style="suggested-action", about=""):
        b = Gtk.Button(label=label); b.set_valign(Gtk.Align.CENTER)
        if about:
            b.update_property([Gtk.AccessibleProperty.LABEL], [f"{label}: {about}"])
        if style:
            b.add_css_class(style)
        b.connect("clicked", lambda *_: on_click())
        return b

    def _row(self, group, title, subtitle="", icon=None, button=None, on_click=None, suffix=None, style="flat"):
        r = Adw.ActionRow(title=GLib.markup_escape_text(title), subtitle=GLib.markup_escape_text(subtitle))
        r.set_subtitle_lines(3) if hasattr(r, "set_subtitle_lines") else None
        if icon:
            r.add_prefix(Gtk.Image.new_from_icon_name(icon))
        if suffix is not None:
            r.add_suffix(suffix)
        if button and on_click:
            b = self._button(button, on_click, style, about=title)
            r.add_suffix(b); r.set_activatable_widget(b)
        group.add(r); return r

    def run_cmd(self, argv, stdin_text=None, privileged=True, people=False, then_refresh=True):
        if argv and argv[0] in CAMERA_CMDS and "--text" not in argv:
            return self.camera_session(argv)
        self.runner.run(argv, privileged, stdin_text,
                        on_done=(lambda rc: self.refresh(people=people or self.people is not None)) if then_refresh else None)

    # ---- camera sessions (registration, calibration) inside this window
    def _camera_page(self):
        if getattr(self, "cam", None) is None:
            cw = appui.CalibrateWindow("ward", 640, 360)
            cw.root.remove(cw.body)
            cw.push_key = self._send_key                  # its buttons drive the privileged process
            cw.preview.accent = appui._rgba(self.win)
            page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
            page.set_halign(Gtk.Align.CENTER)
            self.cam_title = Gtk.Label(xalign=0.5); self.cam_title.add_css_class("title-2"); self.cam_title.set_margin_top(12)
            page.append(self.cam_title); page.append(cw.body)
            self.stack.add_named(page, "camera")
            self.cam = cw
        return self.cam

    def camera_session(self, argv):
        """Run `ward enroll|calibrate --stream-ui` with administrator rights and draw its
        window here: snapshots arrive on stdout, the buttons go back as key lines on stdin,
        messages land in the console. One application, start to finish."""
        if self.runner.busy or getattr(self, "cam_proc", None) is not None:
            self.console.write("… a command is still running\n"); return
        from . import display as dispmod
        cw = self._camera_page()
        who = next((argv[i + 1] for i, a in enumerate(argv) if a == "--identity" and i + 1 < len(argv)), "")
        cond = next((argv[i + 1] for i, a in enumerate(argv) if a == "--add" and i + 1 < len(argv)), "")
        self.cam_title.set_label("Calibrate" if argv[0] == "calibrate" else
                                 (f"Add “{cond}”" if cond else "Register") + (f" · {who}" if who else ""))
        cw._phase = None; cw._pending = False
        cw._apply({"phase": "intro", "manual_start": False, "intro_text": "Approve the administrator prompt to start the camera"})
        appui._set_pill(cw.pill_over, "Waiting for authorisation…", "")
        self._back_page = self.stack.get_visible_child_name() or "people"
        self.stack.set_visible_child_name("camera"); self.side.set_sensitive(False)
        self.console.write(f"$ {'sudo ' if not ROOT else ''}ward {' '.join(argv)}\n")
        appui._set_pill(self.console.state, "running", "accent")
        try:
            proc = subprocess.Popen(priv_cmd(argv + ["--stream-ui"]), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, env=child_env(), bufsize=1)
        except OSError as e:
            self.console.write(f"cannot start: {e}\n"); self._camera_done(1, False); return
        self.cam_proc = proc
        state = {"seen": False, "pending": False}

        def apply(snap):
            state["pending"] = False
            cw._apply(snap)
            return False

        def read_out():
            for line in proc.stdout:
                snap = dispmod.decode_snapshot(line)
                if not snap or snap.get("phase") == "closed":
                    continue
                state["seen"] = True
                if not state["pending"] or snap.get("phase") == "done":
                    state["pending"] = True
                    GLib.idle_add(apply, snap, priority=GLib.PRIORITY_HIGH_IDLE)

        def read_err():
            for line in proc.stderr:
                GLib.idle_add(self.console.write, strip_ansi(line))
            rc = proc.wait()
            GLib.idle_add(self._camera_done, rc, state["seen"])
        threading.Thread(target=read_out, daemon=True).start()
        threading.Thread(target=read_err, daemon=True).start()

    def _send_key(self, k):
        proc = getattr(self, "cam_proc", None)
        if proc is None:
            if k in ("esc", "enter"):
                self.stack.set_visible_child_name(getattr(self, "_back_page", "people"))
            return
        try:
            proc.stdin.write(k + "\n"); proc.stdin.flush()
        except (BrokenPipeError, OSError):
            proc.terminate()

    def _toast(self, text):
        """A transient banner over the whole window (survives page changes)."""
        try:
            self.toast.add_toast(Adw.Toast(title=text))
        except Exception:
            pass

    def _camera_done(self, rc, seen):
        self.cam_proc = None
        self.side.set_sensitive(True)
        if rc in (126, 127) and not ROOT:
            self.console.write("administrator authorisation was cancelled or refused\n")
            self._toast("Administrator authorisation was cancelled")
            self.console_btn.set_active(True)          # otherwise a polkit cancel left no visible trace
        appui._set_pill(self.console.state, "done" if rc == 0 else f"exit {rc}", "ok" if rc == 0 else "warn")
        then = getattr(self, "_cam_then", None)
        if then is not None:                      # a wizard step drove the camera: go back to it and report
            self._cam_then = None
            self.stack.set_visible_child_name(getattr(self, "_back_page", "setup"))
            self.refresh(people=True)
            then(rc)
            return False
        if rc == 0 or not seen:
            self.stack.set_visible_child_name(getattr(self, "_back_page", "people"))
        else:                                     # it failed with the window up: keep the card, offer the way back
            self.console_btn.set_active(True)
            self.cam.btn_next.set_label("Back"); self.cam.btn_cancel.set_visible(False)
        self.refresh(people=True)
        return False

    # ---- in-app setup wizard -------------------------------------------------
    def start_setup_wizard(self):
        if getattr(self, "cam_proc", None) is not None or self.runner.busy:
            self.console.write("… finish the current action first\n"); return
        steps = setup_steps(self.report)
        status = {s["key"]: "ok" for s in steps if s.get("satisfied")}   # already-on steps start satisfied
        self._wiz = {"steps": steps, "i": 0, "status": status}
        if getattr(self, "wiz_page", None) is None:
            self.wiz_page = Adw.PreferencesPage()
            self.stack.add_named(self.wiz_page, "setup")
            self._groups[self.wiz_page] = []
        self.stack.set_visible_child_name("setup")
        self._wiz_render()

    def _wiz_render(self):
        w = self._wiz
        p = self.wiz_page; self._clear(p)
        n, i = len(w["steps"]), w["i"]
        if i >= n:
            g = self._group(p, "Face unlock is set up", "You can change any of this later from the other pages.")
            self._row(g, "Test unlock", "run one attempt exactly like sudo would", "camera-web-symbolic",
                      "Test", self.test_unlock, style="suggested-action")
            self._row(g, "Done", "back to the overview", "emblem-ok-symbolic", "Finish",
                      lambda: (self.stack.set_visible_child_name("overview"), self.refresh(people=True)), style="flat")
            return
        step = w["steps"][i]
        status = w["status"].get(step["key"], "")
        g = self._group(p, f"Set up Ward — step {i + 1} of {n}", "Each step is reversible; your password always works.")
        icon = {"ok": "emblem-ok-symbolic", "failed": "dialog-error-symbolic",
                "skipped": "action-unavailable-symbolic"}.get(status, "")
        row = Adw.ActionRow(title=GLib.markup_escape_text(step["title"]),
                            subtitle=GLib.markup_escape_text(step["desc"]))
        row.set_subtitle_lines(4) if hasattr(row, "set_subtitle_lines") else None
        if icon:
            row.add_prefix(Gtk.Image.new_from_icon_name(icon))
        if status == "running":
            sp = Gtk.Spinner(); sp.start(); sp.set_valign(Gtk.Align.CENTER); row.add_suffix(sp)
        g.add(row)
        nav = self._group(p, "")
        done = status in ("ok", "skipped")
        last = i == n - 1
        if status != "running":
            if not done:
                nav.add(self._nav_row(step["run_label"], self._wiz_run, "suggested-action"))
                if step.get("optional"):
                    nav.add(self._nav_row("Skip this step", lambda: self._wiz_skip(), "flat"))
            else:
                nav.add(self._nav_row("Finish" if last else "Next step",
                                      lambda: self._wiz_advance(), "suggested-action"))
            if i > 0:
                nav.add(self._nav_row("Back", lambda: self._wiz_back(), "flat"))
            nav.add(self._nav_row("Leave setup", lambda: (self.stack.set_visible_child_name("overview"), self.refresh()), "flat"))

    def _nav_row(self, label, cb, style):
        r = Adw.ActionRow(); r.add_suffix(self._button(label, cb, style)); r.set_activatable(False); return r

    def _wiz_set(self, key, status):
        self._wiz["status"][key] = status
        self._wiz_render()

    def _wiz_run(self):
        step = self._wiz["steps"][self._wiz["i"]]
        key = step["key"]
        if step.get("keyring"):
            # ask for the password BEFORE marking running, so a cancelled dialog leaves the step idle
            def enable(pw):
                self._wiz_set(key, "running")
                self.runner.run(["keyring", "enable", "--password-stdin"], True, pw,
                                on_done=lambda rc: self._wiz_set(key, "ok" if rc == 0 else "failed"))
            self.ask_text("Unlock the keyring by face",
                          "Your login password is verified and stored sealed; it is released only after a face accept.",
                          "", enable, password=True)
            return
        self._wiz_set(key, "running")
        if step.get("camera"):
            self._cam_then = lambda rc: self._wiz_set(key, "ok" if rc == 0 else "failed")
            self._back_page = "setup"
            self.camera_session(step["camera"])
        else:
            def after(rc, then=step.get("then_argv")):
                if rc == 0 and then:
                    self.runner.run(then, on_done=lambda rc2: self._wiz_set(key, "ok" if rc2 == 0 else "failed"))
                else:
                    self._wiz_set(key, "ok" if rc == 0 else "failed")
            self.runner.run(step["argv"], on_done=after)

    def _wiz_skip(self):
        self._wiz_set(self._wiz["steps"][self._wiz["i"]]["key"], "skipped")

    def _wiz_advance(self):
        self._wiz["i"] += 1; self._wiz_render()

    def _wiz_back(self):
        self._wiz["i"] = max(0, self._wiz["i"] - 1); self._wiz_render()

    def confirm(self, heading, body, label, cb, destructive=True):
        dlg = message_dialog(self.win, heading, body)
        dlg.add_response("cancel", "Cancel"); dlg.add_response("ok", label)
        dlg.set_response_appearance("ok", RA.DESTRUCTIVE if destructive else RA.SUGGESTED)
        dlg.connect("response", lambda d, r: cb() if r == "ok" else None); dlg.present()

    def ask_text(self, title, body, placeholder, cb, password=False, validate=None):
        dlg = message_dialog(self.win, title, body)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        entry = Gtk.PasswordEntry() if password else Gtk.Entry()
        if not password:
            entry.set_placeholder_text(placeholder)
        entry.set_margin_top(8); box.append(entry)
        hint = Gtk.Label(xalign=0, wrap=True); hint.add_css_class("fu-dim"); box.append(hint)
        dlg.set_extra_child(box)
        dlg.add_response("cancel", "Cancel"); dlg.add_response("ok", "Continue"); dlg.set_response_appearance("ok", RA.SUGGESTED)

        def check(*_):
            t = entry.get_text().strip()
            why = validate(t) if (validate and t) else ""
            hint.set_label(why); dlg.set_response_enabled("ok", bool(t) and not why)
        entry.connect("changed", check); check()
        dlg.connect("response", lambda d, r: cb(entry.get_text().strip()) if r == "ok" and entry.get_text().strip() else None)
        dlg.present()

    def ask_condition(self, who_label, cb):
        dlg = message_dialog(self.win, f"Add a condition for {who_label}",
                             "Put it on first (glasses, headset…), then start: a short four-pose sweep. The calibrated gate is never loosened.")
        for c in CONDITIONS:
            dlg.add_response(c, c.capitalize())
        dlg.add_response("other", "Other…"); dlg.add_response("cancel", "Cancel")

        def on_resp(d, resp):
            if resp == "other":
                self.ask_text("Add a condition", "Name it", "e.g. mask", lambda c: cb(c.lower()),
                              validate=lambda t: "" if re.fullmatch(r"[a-z][a-z0-9-]{0,23}", t.lower()) else "one lowercase word")
            elif resp != "cancel":
                cb(resp)
        dlg.connect("response", on_resp); dlg.present()

    # ---- render
    def render(self, rep):
        self.report = rep
        try:
            for fn in (self._overview, self._people, self._reliability, self._settings, self._advanced):
                fn(rep)
        finally:
            if getattr(self, "selftest", False):
                n = sum(len(s.items) for s in rep["sections"])
                print(f"selftest: rendered {len(rep['sections'])} sections / {n} items, pages: {', '.join(self.pages)}", flush=True)
                GLib.timeout_add(300, lambda: (self.quit(), False)[1])
        return False

    def _n_people(self):
        return None if self.people is None else len(self.people)

    def _overview(self, rep):
        p = self.pages["overview"]; self._clear(p)
        rel = appdata.reliability(rep["rows"], self._n_people())
        on = rep.get("pam")
        if not on:                                   # not set up yet → lead straight into the guided wizard
            g0 = self._group(p, "Get started", "Set up face unlock in a few guided steps — right here in this window.")
            self._row(g0, "Set up Ward", "cameras → infrared → register your face → turn on", "system-run-symbolic",
                      "Start setup", lambda: self.start_setup_wizard(), style="suggested-action")
        g = self._group(p, "Face unlock", f"Login, sudo and system prompts for {rep['user']}")
        sw = Gtk.Switch(active=bool(on)); sw.set_valign(Gtk.Align.CENTER)

        def toggled(s, _p):
            if s.get_active():
                # never enable PAM with nobody enrolled — route through the wizard, which registers first
                if self.people is not None and not any(x["owner"] and x["models"] for x in self.people):
                    s.handler_block(hid); s.set_active(False); s.handler_unblock(hid)
                    self.start_setup_wizard(); return
                self.run_cmd(["pam", "enable"]); return
            s.handler_block(hid); s.set_active(True); s.handler_unblock(hid)   # only flips once confirmed
            self.confirm("Turn face unlock off?", "ward is removed from the login stack; passwords only. Nothing else changes.", "Turn off",
                         lambda: self.run_cmd(["pam", "disable"]))
        hid = sw.connect("notify::active", toggled)
        # keep the switch live while OFF only if the owner is already enrolled (a quick re-enable);
        # on first run / nobody enrolled the 'Start setup' card above is the single entry point
        if not on and not (self.people and any(x["owner"] and x["models"] for x in self.people)):
            sw.set_sensitive(False)
        self._row(g, "On" if on else "Off", "your password always works as well — a failed face check falls back to it", "system-lock-screen-symbolic", suffix=sw)
        text, cls = VERDICT[rel["verdict"]]
        self._row(g, text, describe_stats(rel["overall"]) + " · recent attempts", "emblem-ok-symbolic" if cls == "ok" else "dialog-warning-symbolic",
                  "Test unlock", self.test_unlock, style="flat")
        if self.people is not None:
            names = ", ".join(x["name"] for x in self.people) or "nobody yet"
            self._row(g, "People", names, "system-users-symbolic", "Manage", lambda: self.stack.set_visible_child_name("people"), style="flat")
        todo = [(s, it) for s in rep["sections"] for it in s.items if it.level in ("warn", "fail")]
        g = self._group(p, "To fix" if todo else "Checks", "" if todo else "Nothing needs attention.")
        for s, it in todo:
            self._row(g, it.text, s.title, level_icon(it.level), (it.fix_label or "Fix") if it.fix else None,
                      (lambda fix=it.fix: self.run_cmd(fix)) if it.fix else None, style="suggested-action")
        ex = Adw.ExpanderRow(title="All checks", subtitle=f"{sum(len(s.items) for s in rep['sections'])} items")
        for s in rep["sections"]:
            for it in s.items:
                r = Adw.ActionRow(title=GLib.markup_escape_text(it.text), subtitle=GLib.markup_escape_text(s.title))
                r.add_prefix(Gtk.Image.new_from_icon_name(level_icon(it.level))); ex.add_row(r)
        g.add(ex)

    def _people(self, rep):
        p = self.pages["people"]; self._clear(p)
        user = rep["user"]
        if self.people is None:
            g = self._group(p, "People who can unlock this login", "Face models are kept readable by root only.")
            self._row(g, "Show people", "asks for the administrator password", "changes-prevent-symbolic", "Show", self.load_people, style="suggested-action")
            return
        g = self._group(p, "People who can unlock this login",
                        "Any registered face unlocks this account. Register only people you would give your password to.")
        if not self.people:
            self._row(g, "Nobody registered yet", "one guided window in your usual light (it also works in the dark with an IR camera)",
                      "dialog-warning-symbolic", "Register my face", lambda: self.run_cmd(ENROL, people=True), style="suggested-action")
        for x in self.people:
            ident, name = x["identity"], x["name"]
            cams = " · ".join(f"{sp.upper()} {n}" for sp, n in sorted(x["models"].items())) or "no models"
            sub = [f"{cams} poses", f"registered {x['enrolled']}" if x["enrolled"] else "",
                   f"last unlock {x['last_seen'][:16].replace('T', ' ')}" if x["last_seen"] else "no unlock yet",
                   ("with " + ", ".join(x["conditions"])) if x["conditions"] else ""]
            ex = Adw.ExpanderRow(title=GLib.markup_escape_text(name + (" (account owner)" if x["owner"] else "")),
                                 subtitle=GLib.markup_escape_text(" · ".join(s for s in sub if s)))
            ex.add_prefix(Gtk.Image.new_from_icon_name("avatar-default-symbolic"))
            enrol = ENROL if x["owner"] else ["enroll", "--fresh", "--identity", ident]

            def sub_row(title, subtitle, label, cb, style="flat"):
                r = Adw.ActionRow(title=GLib.markup_escape_text(title), subtitle=GLib.markup_escape_text(subtitle))
                r.add_suffix(self._button(label, cb, style)); ex.add_row(r)
            sub_row("Register again", "replaces this person's face models with a fresh guided sweep", "Register again",
                    lambda e=enrol: self.run_cmd(e, people=True))
            for c in x["conditions"]:
                sub_row(f"Condition: {c}", "extra models taken while wearing it", "Remove",
                        lambda c=c, i=ident, n=name: self.confirm(f"Remove “{c}” for {n}?", "Those extra models are deleted; the rest stay.", "Remove",
                                                                  lambda: self.run_cmd(["models", "--remove-condition", c, "--identity", i], people=True)))
            sub_row("Add a condition", "glasses, headset… — a short sweep while wearing it", "Add…",
                    lambda i=ident, n=name: self.ask_condition(n, lambda c: self.run_cmd(["enroll", "--add", c] + ([] if not i else ["--identity", i]), people=True)))
            if x["learned"]:
                sub_row("Learned models", f"{x['learned']} added automatically after confident unlocks", "Forget",
                        lambda i=ident, n=name: self.confirm(f"Forget learned models for {n}?", "Only the models added automatically are deleted; registered ones stay.", "Forget",
                                                             lambda: self.run_cmd(["models", "--forget-learned", "--identity", i], people=True)))
            if not x["owner"]:
                sub_row("Remove this person", "they can no longer unlock this login", "Remove",
                        lambda i=ident, n=name: self.confirm(f"Remove {n}?", f"All of {n}'s face models are deleted. They can no longer unlock {user}'s account.", "Remove",
                                                     lambda: self.run_cmd(["models", "--remove-identity", i], people=True)), style="destructive-action")
            else:
                sub_row("Delete all my face models", "face unlock stops working until you register again", "Delete",
                        lambda: self.confirm("Delete your face models?", "Other people registered on this login are kept. Your password keeps working.", "Delete",
                                             lambda: self.run_cmd(["models", "--remove-identity", ""], people=True)), style="destructive-action")
            g.add(ex)
        if self.people:
            self._row(g, "Register another person", "a family member or a second look — same guided window, under their name",
                      "list-add-symbolic", "Add…",
                      lambda: self.ask_text("Register another person", f"They will be able to unlock {user}'s account. Sit them in front of the camera after Continue.",
                                            "Name", lambda n: self.run_cmd(["enroll", "--identity", n], people=True),
                                            validate=lambda t: appdata.valid_name(t, user) or ("already registered" if any(t == x["name"] for x in self.people) else "")),
                      style="suggested-action")
        g = self._group(p, "Thresholds")
        self._row(g, "Recalibrate", f"calibrated {getattr(self, 'calibrated', '') or 'at registration'} · eight guided passes re-measure your real distances",
                  "view-refresh-symbolic", "Calibrate", lambda: self.run_cmd(["calibrate"]))

    def _reliability(self, rep):
        p = self.pages["reliability"]; self._clear(p)
        rows = rep["rows"]
        rel = appdata.reliability(rows, self._n_people())
        text, cls = VERDICT[rel["verdict"]]
        g = self._group(p, "How well it works", "From the last 50 attempts in the log. Every attempt falls back to the password on failure.")
        pill = appui._pill(text, cls); pill.set_valign(Gtk.Align.CENTER)
        self._row(g, text, describe_stats(rel["overall"]), "emblem-ok-symbolic" if cls == "ok" else "dialog-warning-symbolic", suffix=pill)
        self._row(g, "In daylight", describe_stats(rel["day"]), "weather-clear-symbolic")
        self._row(g, "In the dark (IR only)", describe_stats(rel["night"]), "weather-clear-night-symbolic")
        for svc, s in (rel["services"].items() if len(rel["services"]) > 1 else ()):
            self._row(g, f"Via {svc}", describe_stats(s), "system-run-symbolic")
        if rel["who"] and (len(rel["who"]) > 1 or "" not in rel["who"]):
            for who, n in rel["who"].items():
                self._row(g, f"Unlocked as {who or rep['user']}", f"{n} accepts", "avatar-default-symbolic")
        if rel["reasons"]:
            g = self._group(p, "Why attempts failed")
            labels = {"no-match": "face not recognised", "liveness": "could not prove a live face", "spoof": "looked like a photo or screen",
                      "no-face": "no face seen", "dark": "too dark", "timeout": "took too long", "lockout": "locked out after spoof attempts"}
            tips = {"no-match": "register again in this light", "liveness": "move your head slightly during the attempt",
                    "no-face": "face the camera", "dark": "more light, or an IR camera", "timeout": "check the camera isn't busy"}
            for r, n in rel["reasons"].items():
                self._row(g, f"{n} × {labels.get(r, r)}", tips.get(r, ""), "dialog-error-symbolic")
        g = self._group(p, "What to do")
        for text_, argv in rel["advice"]:
            self._row(g, text_, "", "dialog-information-symbolic", "Do it" if argv else None,
                      (lambda a=argv: self.run_cmd(a, people=a[0] == "enroll")) if argv else None, style="suggested-action")
        self._row(g, "Test unlock now", "runs one attempt exactly like sudo would and shows every distance and vote", "camera-web-symbolic",
                  "Test", self.test_unlock, style="suggested-action")
        g = self._group(p, "Recent attempts")
        for r in reversed(rows[-25:]):
            what = "unlocked" + (f" as {r['who']}" if r.get("who") else "") if r["accepted"] else r.get("result", "?")
            extra = " · ".join(x for x in ("dark (IR only)" if r.get("dark") else "", f"{r['ms']} ms" if r["ms"] else "", r["service"]) if x)
            detail = " · ".join(x for x in (f"ir {r['ir']:.2f}" if r.get("ir") is not None else "",
                                             f"rgb {r['rgb']:.2f}" if r.get("rgb") is not None else "", " ".join(r["votes"])) if x)
            row = self._row(g, f"{r['time'].replace('T', ' ')}  {what}", extra, "emblem-ok-symbolic" if r["accepted"] else "dialog-error-symbolic")
            if detail:
                row.set_tooltip_text(detail)
        if not rows:
            self._row(g, "Nothing logged yet", "")

    def test_unlock(self):
        self.run_cmd(["auth", "-v"])

    def _settings(self, rep):
        p = self.pages["settings"]; self._clear(p)
        cfg = rep["cfg"]
        g = self._group(p, "Security", "Presets only ever tighten the gate measured at registration.")
        levels = list(polmod.LEVELS)
        dd = Gtk.DropDown.new_from_strings(levels); dd.set_valign(Gtk.Align.CENTER)
        if rep.get("policy") in levels:
            dd.set_selected(levels.index(rep["policy"]))
        dd.connect("notify::selected", lambda d, _p: self.run_cmd(["tune", "policy", levels[d.get_selected()]]))
        self._row(g, "Policy", polmod.describe(rep.get("policy") or "balanced"), "security-high-symbolic", suffix=dd)
        opts = ["", "convenience", "balanced", "strict", "off"]
        names = ["Same as overall", "Convenience", "Balanced", "Strict", "Password only"]
        for svc, key in (("sudo", "policy_sudo"), ("System prompts (polkit)", "policy_polkit"), ("Login screen", "policy_login"), ("Lock screen", "policy_unlock")):
            cur = cfg.get("security", {}).get(key, "") or ""
            d = Gtk.DropDown.new_from_strings(names); d.set_valign(Gtk.Align.CENTER)
            d.set_selected(opts.index(cur) if cur in opts else 0)
            d.connect("notify::selected", lambda d, _p, k=key: self.run_cmd(["tune", f"security.{k}", opts[d.get_selected()]]))
            self._row(g, svc, "", suffix=d)
        g = self._group(p, "Behaviour")
        b = cfg["behaviour"]

        def switch(title, sub, key, val):
            s = Gtk.Switch(active=bool(val)); s.set_valign(Gtk.Align.CENTER)
            s.connect("notify::active", lambda w, _p: self.run_cmd(["tune", key, "true" if w.get_active() else "false"]))
            self._row(g, title, sub, suffix=s)
        switch("First login uses the password", "keeps the keyring unlocking the classic way", "behaviour.skip_login", b.get("skip_login"))
        switch("Face unlock on text consoles", "Ctrl+Alt+F3 logins", "behaviour.tty_login", b.get("tty_login"))
        switch("Learn from unlocks", "add a model after a confident unlock in new conditions", "behaviour.learn", b.get("learn"))
        notes = ["accept", "all", "off"]
        d = Gtk.DropDown.new_from_strings(["After unlocks", "After every attempt", "Never"]); d.set_valign(Gtk.Align.CENTER)
        d.set_selected(notes.index(b.get("notify")) if b.get("notify") in notes else 0)
        d.connect("notify::selected", lambda d, _p: self.run_cmd(["tune", "behaviour.notify", notes[d.get_selected()]]))
        self._row(g, "Desktop notification", "", suffix=d)
        g = self._group(p, "Speed")
        ds = rep.get("daemon") or {"installed": False, "active": False}
        if ds["installed"]:
            sw = Gtk.Switch(active=bool(ds["active"])); sw.set_valign(Gtk.Align.CENTER)
            sw.connect("notify::active", lambda w, _p: self.run_cmd(["daemon", "enable" if w.get_active() else "disable"]))
            self._row(g, "Keep face unlock ready", "the face models stay loaded (~150 MB of memory) so an unlock starts about 1 s sooner", "power-profile-performance-symbolic", suffix=sw)
        else:
            self._row(g, "Keep face unlock ready", "not installed — run install.sh again", "power-profile-performance-symbolic")
        g = self._group(p, "Keyring", "After a face login, unlock GNOME Keyring / KWallet with your password sealed by the encrypted disk or TPM.")
        ks = next((it for s in rep["sections"] if s.key == "access" for it in s.items if "keyring" in it.text), None)
        known = ks is not None and "needs root" not in ks.text
        enabled = bool(ks) and ks.level == "ok"

        def disable():
            self.confirm("Disable keyring unlock?", "The sealed copy of your password is deleted.", "Disable", lambda: self.run_cmd(["keyring", "disable"]))
        if not known or not enabled:
            self._row(g, "Unlock the keyring by face", ks.text if ks else "", "dialog-password-symbolic", "Enable", self.keyring_enable)
        if not known or enabled:
            self._row(g, "Stop unlocking the keyring", "removes the sealed password", None if not known else "dialog-password-symbolic", "Disable",
                      disable, style="destructive-action" if known else "flat")

    def _advanced(self, rep):
        p = self.pages["advanced"]; self._clear(p)
        g = self._group(p, "Set up", "The whole chain in one go, or step by step. Each step is reversible.")
        self._row(g, "Run the full setup", "a guided wizard, right here in this window — cameras → IR emitter → anti-spoof → register → keyring → turn on", "system-run-symbolic", "Start",
                  lambda: self.start_setup_wizard(), style="suggested-action")
        self._row(g, "Detect cameras", PROFILE_TEXT.get(rep.get("profile"), "not detected yet"), None, "Detect", lambda: self.run_cmd(["setup", "--cameras-only"]))
        if rep.get("profile") in ("A", "B"):
            self._row(g, "IR emitter", "find the LED control and benchmark it (enables the shadow liveness test)", None, "Probe",
                      lambda: self.runner.run(["emitter", "probe"], on_done=lambda rc: self.run_cmd(["emitter", "toggle-bench"]) if rc == 0 else self.refresh()))
        cfg = rep["cfg"]
        rk = cfg["recognition"].get("recogniser", "dlib")
        self._row(g, "Recogniser", f"{rk} · switching requires registering everyone again", None,
                  "Use ArcFace" if rk == "dlib" else "Use dlib", lambda: self.confirm("Switch recogniser?", "Every face must be registered again afterwards.", "Switch",
                                                                                    lambda: self.run_cmd(["recogniser", "use", "arcface" if rk == "dlib" else "dlib"]), destructive=False))
        on = bool(cfg["liveness"].get("antispoof_cnn", True))
        self._row(g, "Anti-spoof CNN", ("on" if on else "off") + " · daylight liveness on the colour camera (bundled MiniFASNetV2; abstains when unsure)", None,
                  "Turn off" if on else "Turn on", lambda: self.run_cmd(["antispoof", "off" if on else "on"]))
        g = self._group(p, "Lab")
        self._row(g, "Open the lab", "live evaluation bench", "utilities-system-monitor-symbolic", "Open", lambda: self.run_cmd(["lab"], then_refresh=False))
        frames = os.path.join(os.environ.get("WARD_ETC", "/etc/ward"), "lab", "frames")
        self._row(g, "Record frames for replay", frames, "media-record-symbolic", "Record", lambda: self.run_cmd(["lab", "--record-frames", frames], then_refresh=False))
        g = self._group(p, "Maintenance")
        self._row(g, "Full diagnostic report", "every check, with what to do", "utilities-terminal-symbolic", "Run", lambda: self.run_cmd(["doctor"], then_refresh=False))
        self._row(g, "Show the config", "all settings and measured thresholds", "document-properties-symbolic", "Show", lambda: self.run_cmd(["tune"], then_refresh=False))

    def keyring_enable(self):
        self.ask_text("Unlock the keyring by face", "Your login password is verified and stored sealed (encrypted disk / TPM). It is released only after a face accept.", "",
                      lambda pw: self.run_cmd(["keyring", "enable", "--password-stdin"], stdin_text=pw), password=True)
