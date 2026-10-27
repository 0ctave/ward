"""Keyring credential (roadmap K1): sealing through a fake systemd-creds, PAM-line repair, and the
native module exercised through libpam with fake helper/creds binaries (when gcc is available)."""
import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("WARD_ETC", tempfile.mkdtemp(prefix="ward-test-"))
os.environ.setdefault("WARD_LOG", os.path.join(os.environ["WARD_ETC"], "log"))

from ward.engine import keyring as kr

ROOT = Path(__file__).resolve().parents[1]


def fake_creds(d: Path) -> str:
    """A stand-in for systemd-creds: encrypt = reverse the bytes into the file, decrypt = reverse back."""
    p = d / "creds.sh"
    p.write_text("#!/bin/sh\n"
                 "case \"$1\" in\n"
                 "  encrypt) tr -d '\\n' | rev > \"$5\" ;;\n"
                 "  decrypt) rev < \"$3\"; echo ;;\n"
                 "esac\n")
    p.chmod(0o755)
    return str(p)


class Sealing(unittest.TestCase):
    def test_seal_and_unseal_roundtrip_with_fake_creds(self):
        d = Path(tempfile.mkdtemp()); creds = fake_creds(d)
        kr.CREDS_DIR = d / "keyring"
        path, kind = kr.seal("alice", "hunter2", force_host_key=True, creds_bin=creds)
        self.assertTrue(path.exists()); self.assertEqual(oct(path.stat().st_mode & 0o777), "0o600")
        self.assertIn(kind, ("tpm2", "host"))
        self.assertEqual(kr.unseal("alice", creds_bin=creds), "hunter2")
        self.assertTrue(kr.status("alice")["enrolled"])
        self.assertTrue(kr.remove("alice")); self.assertFalse(kr.remove("alice"))
        self.assertIsNone(kr.unseal("alice", creds_bin=creds))
        with self.assertRaises(ValueError):
            kr.cred_path("../etc")

    def test_key_choice_prefers_fast_host_key_on_fde(self):
        import ward.engine.keyring as k
        orig_tpm, orig_fde = k.has_tpm, k.root_is_encrypted
        try:
            # FDE present → host key (fast), even when a TPM exists, unless forced
            k.has_tpm = lambda: True; k.root_is_encrypted = lambda: True
            self.assertEqual(k.choose_key(), "host")
            self.assertEqual(k.choose_key(prefer_tpm=True), "tpm2")
            # no FDE, TPM present → TPM (host key would be readable from a stolen disk)
            k.root_is_encrypted = lambda: False
            self.assertEqual(k.choose_key(), "tpm2")
            self.assertEqual(k.choose_key(force_host_key=True), "host")
            # no TPM, no FDE → host (caller must have passed force_host_key; seal() re-checks)
            k.has_tpm = lambda: False
            self.assertEqual(k.choose_key(), "host")
        finally:
            k.has_tpm, k.root_is_encrypted = orig_tpm, orig_fde

    def test_verify_password_without_shadow_access_is_none(self):
        # unprivileged: no shadow → None (caller asks the user); never a false True
        self.assertIn(kr.verify_password("no-such-user-ward", "x"), (None, False))


class PamRepair(unittest.TestCase):
    def test_tagged_comments_are_restored(self):
        d = Path(tempfile.mkdtemp())
        (d / "gdm-password").write_text("auth\trequisite\tpam_nologin.so\n"
                                        "# auth    optional        pam_gnome_keyring.so  # ward-keyring\n"
                                        "@include common-auth\n"
                                        "session optional        pam_gnome_keyring.so auto_start\n")
        (d / "common-password").write_text("# password\toptional\tpam_gnome_keyring.so   # ward-keyring\n")
        (d / "gdm-password.ward-bak").write_text("auth    optional        pam_gnome_keyring.so\n")
        found = kr.disabled_keyring_lines(d)
        self.assertEqual(sorted(p.name for p, _ in found), ["common-password", "gdm-password"])
        self.assertEqual(sorted(Path(x).name for x in kr.repair_pam_lines(d, dry_run=True)), ["common-password", "gdm-password"])
        self.assertTrue(kr.disabled_keyring_lines(d))                        # dry run changed nothing
        kr.repair_pam_lines(d)
        self.assertEqual(kr.disabled_keyring_lines(d), [])
        text = (d / "gdm-password").read_text()
        self.assertIn("\nauth    optional        pam_gnome_keyring.so\n", text)
        self.assertNotIn("ward-keyring", text)
        self.assertEqual((d / "common-password").read_text(), "password\toptional\tpam_gnome_keyring.so\n")


class NativeModule(unittest.TestCase):
    """Build data/pam_ward.c and drive pam_sm_authenticate through the real libpam."""

    @classmethod
    def setUpClass(cls):
        if shutil.which("gcc") is None:
            raise unittest.SkipTest("gcc not available")
        try:
            ctypes.CDLL("libpam.so.0", mode=ctypes.RTLD_GLOBAL)
        except OSError:
            raise unittest.SkipTest("libpam.so.0 not available")
        cls.dir = Path(tempfile.mkdtemp())
        cls.so = cls.dir / "pam_ward.so"
        r = subprocess.run(["gcc", "-O2", "-Wall", "-fPIC", "-shared", "-o", str(cls.so), str(ROOT / "data" / "pam_ward.c")],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise unittest.SkipTest(f"module did not build: {r.stderr}")

    def _pam(self, helper_exit: int, cred: str | None, body: str | None = None, opts: str = "", consumer: bool = True, service: str = "gdm-password"):
        """A real pam_authenticate over a private config dir (pam_start_confdir, Linux-PAM ≥ 1.4):
        our module first, then pam_exec with expose_authtok as the token consumer — the same
        way pam_gnome_keyring receives it."""
        d = self.dir
        helper = d / "helper.sh"
        helper.write_text(body or f"#!/bin/sh\necho 'Looking for your face…'\necho \"svc=$PAM_SERVICE user=$PAM_USER type=$PAM_TYPE\"\nexit {helper_exit}\n"); helper.chmod(0o755)
        creds = d / "creds.sh"; creds.write_text("#!/bin/sh\ncat \"$3\"\n"); creds.chmod(0o755)
        kd = d / "keyring"; kd.mkdir(exist_ok=True)
        cp = kd / "tester.cred"
        if cred is None:
            if cp.exists(): cp.unlink()
        else:
            cp.write_text(cred + "\n")
        tokfile = d / "token"; tokfile.write_text("")
        cons = d / "consumer.sh"; cons.write_text(f"#!/bin/sh\ncat > {tokfile}\n"); cons.chmod(0o755)
        conf = d / "pam.d"; conf.mkdir(exist_ok=True)
        pam_exec = next((p for p in Path("/lib/x86_64-linux-gnu/security", ).glob("pam_exec.so")), None) or Path("pam_exec.so")
        (conf / service).write_text(
            f"auth\t[success=1 default=ignore]\t{self.so} helper={helper} creds_dir={kd} creds={creds}{opts}\n"
            f"auth\trequisite\tpam_deny.so\n"
            + (f"auth\toptional\t{pam_exec} quiet expose_authtok {cons}\n" if consumer else "auth\toptional\tpam_permit.so\n"))
        libpam = ctypes.CDLL("libpam.so.0", mode=ctypes.RTLD_GLOBAL)
        if not hasattr(libpam, "pam_start_confdir"):
            self.skipTest("pam_start_confdir needs Linux-PAM >= 1.4")

        class Msg(ctypes.Structure):
            _fields_ = [("msg_style", ctypes.c_int), ("msg", ctypes.c_char_p)]

        class Resp(ctypes.Structure):
            _fields_ = [("resp", ctypes.c_char_p), ("resp_retcode", ctypes.c_int)]
        CONV = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.POINTER(Msg)), ctypes.POINTER(ctypes.POINTER(Resp)), ctypes.c_void_p)

        class Conv(ctypes.Structure):
            _fields_ = [("conv", CONV), ("appdata_ptr", ctypes.c_void_p)]
        seen = []
        libc = ctypes.CDLL("libc.so.6")
        libc.calloc.restype = ctypes.c_void_p

        def conv(n, msgs, resp, data):
            for i in range(n):
                seen.append((msgs[i].contents.msg_style, msgs[i].contents.msg.decode()))
            resp[0] = ctypes.cast(libc.calloc(n, ctypes.sizeof(Resp)), ctypes.POINTER(Resp)); return 0
        cv = Conv(CONV(conv), None)
        pamh = ctypes.c_void_p()
        self.assertEqual(libpam.pam_start_confdir(service.encode(), b"tester", ctypes.byref(cv), str(conf).encode(), ctypes.byref(pamh)), 0)
        rc = libpam.pam_authenticate(pamh, 0)
        libpam.pam_end(pamh, 0)
        tok = tokfile.read_bytes().rstrip(b"\n") or None
        return rc, seen, tok

    def test_accept_forwards_messages_and_sets_the_token(self):
        rc, seen, tok = self._pam(0, "s3cret")
        self.assertEqual(rc, 0)
        self.assertEqual([m for _, m in seen][0], "Looking for your face…")
        self.assertIn("svc=gdm-password user=tester type=auth", [m for _, m in seen][1])   # env the helper's feedback needs
        self.assertTrue(all(style == 4 for style, _ in seen))                       # PAM_TEXT_INFO
        self.assertEqual(tok, b"s3cret")

    def test_sudo_does_not_unseal_the_keyring_token(self):
        # scope fix: sudo/su/polkit never unlock the keyring, so the cleartext login password is
        # never materialised there even when a credential is enrolled and the face is accepted
        rc, _, tok = self._pam(0, "s3cret", service="sudo")
        self.assertEqual(rc, 0); self.assertIsNone(tok)

    def test_reject_is_auth_err_without_token(self):
        rc, _, tok = self._pam(1, "s3cret")
        self.assertNotEqual(rc, 0); self.assertIsNone(tok)                       # pam_deny ends the stack

    def test_accept_without_credential_sets_no_token(self):
        rc, _, tok = self._pam(0, None)
        self.assertEqual(rc, 0); self.assertIsNone(tok)

    def test_silent_hung_helper_times_out_and_falls_through(self):
        # golden rule 3: a wedged camera must never hang the login — the helper prints NOTHING
        # (fgets alone would wait forever) and its child keeps the pipe open (needs the group kill)
        import time as _t
        t = _t.monotonic()
        rc, seen, tok = self._pam(0, "s3cret", body="#!/bin/sh\nsleep 30 &\nsleep 30\nexit 0\n", opts=" timeout=1")
        self.assertLess(_t.monotonic() - t, 4.0)
        self.assertNotEqual(rc, 0); self.assertIsNone(tok)

    def test_chatty_hung_helper_times_out(self):
        import time as _t
        t = _t.monotonic()
        rc, seen, _ = self._pam(0, None, body="#!/bin/sh\nwhile true; do echo hold; sleep 0.2; done\n", opts=" timeout=1")
        self.assertLess(_t.monotonic() - t, 4.0)
        self.assertNotEqual(rc, 0)
        self.assertTrue(seen and all(m == "hold" for _, m in seen))

    def test_helper_gets_a_scrubbed_environment(self):
        # PAM-1: through setuid sudo / polkit the caller's environment is the unprivileged user's;
        # the root helper must not inherit PYTHONHOME, HOME (user site-packages) or the like
        os.environ["PYTHONHOME"] = "/tmp/evil"; os.environ["WARD_CANARY"] = "x"
        try:
            rc, seen, _ = self._pam(0, None, body="#!/bin/sh\necho \"home=${PYTHONHOME:-unset} canary=${WARD_CANARY:-unset} user=$PAM_USER\"\nexit 0\n")
        finally:
            os.environ.pop("PYTHONHOME", None); os.environ.pop("WARD_CANARY", None)
        self.assertEqual(rc, 0)
        self.assertIn("home=unset canary=unset user=tester", [m for _, m in seen])

    def test_reject_does_not_wait_for_a_stuck_unseal(self):
        # PAM-2: a decrypt that ignores SIGTERM (wedged TPM) must not hold back the password module
        import time as _t
        d = self.dir
        slow = d / "slowcreds.sh"; slow.write_text("#!/bin/sh\ntrap '' TERM\nsleep 30\n"); slow.chmod(0o755)
        t = _t.monotonic()
        rc, _, tok = self._pam(1, "s3cret", opts=f" creds={slow}")
        self.assertLess(_t.monotonic() - t, 4.0)
        self.assertNotEqual(rc, 0); self.assertIsNone(tok)

    def test_accept_survives_an_ignored_sigchld(self):
        # PAM-5: a caller with SIGCHLD ignored must not turn an accept into a failure (ECHILD)
        import signal as _s
        old = _s.signal(_s.SIGCHLD, _s.SIG_IGN)
        try:
            # no pam_exec consumer here: Linux-PAM < 1.5.3's pam_exec itself fails under an ignored
            # SIGCHLD (PAM_PERM_DENIED on Debian 12 / Ubuntu 22.04, measured) — that is not our module
            rc, _, _ = self._pam(0, None, consumer=False)
        finally:
            _s.signal(_s.SIGCHLD, old)
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
