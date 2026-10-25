"""Security policy (DESIGN.md §6.8): presets, per-service overrides, lockout; identities and
conditions in the store; the engine honouring them on the replay harness."""
import os
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("WARD_ETC", tempfile.mkdtemp(prefix="ward-test-"))
os.environ.setdefault("WARD_LOG", os.path.join(os.environ["WARD_ETC"], "log"))

from ward.engine import config, policy, store


def _cfg(**sec):
    cfg = config.load()
    cfg["recognition"]["certainty"] = 0.40; cfg["recognition"]["rgb_certainty"] = 0.38
    cfg["security"].update(sec)
    return cfg


class Presets(unittest.TestCase):
    def test_levels_resolve_and_only_tighten(self):
        cfg = _cfg(policy="balanced")
        e = policy.effective(cfg, "sudo")
        self.assertEqual(e["level"], "balanced"); self.assertEqual(e["certainty"], 0.40); self.assertTrue(e["rgb_when_bright"])
        self.assertEqual(e["required_votes"], 2); self.assertTrue(e["learn"])
        self.assertFalse(e["whitened"])                                    # off by default (measured never to fire)
        on = _cfg(policy="balanced"); on["recognition"]["whitened"] = True
        self.assertTrue(policy.effective(on, "sudo")["whitened"])          # still available as an opt-in
        e = policy.effective(_cfg(policy="strict"), "gdm-password")
        self.assertEqual(e["certainty"], 0.35); self.assertTrue(e["require_rgb_match"]); self.assertFalse(e["whitened"])
        self.assertTrue(e["need_strong_voter"]); self.assertFalse(e["learn"])
        e = policy.effective(_cfg(policy="convenience"))
        self.assertEqual(e["required_votes"], 1); self.assertFalse(e["rgb_when_bright"])
        # the preset never loosens a measured gate: strict caps at 0.35 even when the gate is 0.45
        cfg = _cfg(policy="strict"); cfg["recognition"]["certainty"] = 0.45
        self.assertEqual(policy.effective(cfg)["certainty"], 0.35)

    def test_per_service_override_and_off(self):
        cfg = _cfg(policy="balanced", policy_sudo="strict", policy_login="off")
        self.assertEqual(policy.level_for(cfg, "sudo"), "strict")
        self.assertEqual(policy.level_for(cfg, "su-l"), "strict")
        self.assertEqual(policy.level_for(cfg, "login"), "off"); self.assertTrue(policy.effective(cfg, "login")["off"])
        self.assertEqual(policy.level_for(cfg, "gdm-password"), "balanced")
        self.assertEqual(policy.level_for(_cfg(policy="bogus")), "balanced")

    def test_apply_preset(self):
        cfg = _cfg()
        self.assertEqual(policy.apply_preset(cfg, "strict"), {"security.policy": ("balanced", "strict")})
        with self.assertRaises(ValueError):
            policy.apply_preset(cfg, "paranoid")
        for lvl in policy.LEVELS:
            self.assertTrue(policy.describe(lvl))


class Lockout(unittest.TestCase):
    def test_spoof_fails_pause_face_unlock(self):
        log = os.path.join(tempfile.mkdtemp(), "log")
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        lines = [f"{now} user=eve result=spoof:cross ir=0.20 frames=3 t=900ms"] * 3 + [f"{now} user=eve result=no-match ir=0.7 frames=9 t=2500ms"]
        open(log, "w").write("\n".join(lines) + "\n")
        cfg = _cfg(lockout_failures=3, lockout_minutes=15)
        locked, remaining = policy.locked_out(cfg, log, "eve")
        self.assertTrue(locked); self.assertGreater(remaining, 14 * 60)
        self.assertEqual(policy.locked_out(cfg, log, "bob"), (False, 0))               # per user
        self.assertEqual(policy.locked_out(_cfg(lockout_failures=0), log, "eve"), (False, 0))   # disabled
        open(log, "w").write("\n".join(lines[:2]) + "\n")
        self.assertFalse(policy.locked_out(cfg, log, "eve")[0])                       # two is not three
        old = "2000-01-01T00:00:00"
        open(log, "w").write("\n".join(l.replace(now, old) for l in lines) + "\n")
        self.assertFalse(policy.locked_out(cfg, log, "eve")[0])                       # expired


class Identities(unittest.TestCase):
    def test_store_keeps_identity_and_condition(self):
        rng = np.random.default_rng(3)
        me = rng.normal(size=128); me /= np.linalg.norm(me)
        alice = rng.normal(size=128); alice /= np.linalg.norm(alice)
        st = store.Store(user="home", path=store.path_for("home"))
        st.add("C/level", "ir", me, 0.9, None, samples=[me])
        st.add("C/level", "ir", me, 0.9, None, samples=[me], condition="glasses")
        st.add("C/level", "ir", alice, 0.9, None, samples=[alice], identity="Alice")
        st.save(); back = store.load("home")
        self.assertEqual(back.identities(), ["", "Alice"]); self.assertEqual(back.conditions(), ["glasses"])
        d, m = back.nearest(alice, "ir"); self.assertEqual(m.identity, "Alice"); self.assertLess(d, 0.01)
        d, m = back.nearest(me, "ir"); self.assertEqual(m.identity, "")
        # scoped removal never touches the others
        self.assertEqual(back.remove_label("C/level", identity="", condition="glasses"), 1)
        self.assertEqual({(m.identity, m.condition) for m in back.models}, {("", ""), ("Alice", "")})
        self.assertEqual(back.remove_identity("Alice"), 1); self.assertEqual(back.identities(), [""])

    def test_engine_logs_the_matched_identity(self):
        from tests.test_compat import frames_dir, synthetic_face, USER_VEC, OTHER_VEC, run_auth
        st = store.Store(user="fam", path=store.path_for("fam"))
        for sp in ("ir", "rgb"):
            st.add("C/level", sp, USER_VEC, 0.9, (0, 0), samples=[USER_VEC] * 3)
            st.add("C/level", sp, OTHER_VEC, 0.9, (0, 0), samples=[OTHER_VEC] * 3, identity="Alice")
        st.save()
        out = run_auth("fam", frames_dir(), who=OTHER_VEC)
        self.assertTrue(out.accepted, out.logline()); self.assertEqual(out.identity, "Alice"); self.assertIn("who=Alice", out.logline())
        out = run_auth("fam", frames_dir(), who=USER_VEC)
        self.assertTrue(out.accepted); self.assertEqual(out.identity, "")

    def test_engine_respects_policy_off_and_lockout(self):
        from tests.test_compat import frames_dir, run_auth, enrolled_store
        enrolled_store("pol")
        os.environ["PAM_SERVICE"] = "login"
        try:
            out = run_auth("pol", frames_dir(), {"security": {"policy_login": "off"}})
            self.assertEqual(out.reason, "policy")
        finally:
            os.environ.pop("PAM_SERVICE", None)
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(os.environ["WARD_LOG"], "a") as f:
            for _ in range(3):
                f.write(f"{now} user=pol result=spoof:motion ir=0.2 frames=4 t=900ms\n")
        out = run_auth("pol", frames_dir(), {"security": {"lockout_failures": 3}})
        self.assertEqual(out.reason, "lockout")
        out = run_auth("pol", frames_dir(), {"security": {"lockout_failures": 0}})
        self.assertTrue(out.accepted)


if __name__ == "__main__":
    unittest.main()
