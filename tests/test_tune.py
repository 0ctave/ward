import os
import tempfile
import unittest
from types import SimpleNamespace

os.environ.setdefault("WARD_ETC", tempfile.mkdtemp(prefix="ward-test-"))   # bypasses need_root

from ward.cli.doctor import cmd_tune          # noqa: E402
from ward.engine import config as cfgmod       # noqa: E402


class Tune(unittest.TestCase):
    def tearDown(self):
        try:
            cfgmod.CONFIG_PATH.unlink()        # back to defaults; don't leak state to other suites
        except FileNotFoundError:
            pass

    def _run(self, *items):
        return cmd_tune(SimpleNamespace(args=list(items)))

    def test_single_legacy_form_still_works(self):
        self.assertEqual(self._run("liveness.cross_spectrum", "false"), 0)
        self.assertIs(cfgmod.load()["liveness"]["cross_spectrum"], False)

    def test_multi_pair_applied_and_saved_once(self):
        self.assertEqual(self._run("security.policy_sudo=off", "liveness.antispoof_cnn=false"), 0)
        cfg = cfgmod.load()
        self.assertEqual(cfg["security"]["policy_sudo"], "off")
        self.assertIs(cfg["liveness"]["antispoof_cnn"], False)

    def test_multi_pair_atomic_on_bad_key(self):
        self._run("liveness.antispoof_cnn=true")                       # known baseline
        rc = self._run("liveness.antispoof_cnn=false", "bogus.nonsense=1")
        self.assertEqual(rc, 1)
        self.assertIs(cfgmod.load()["liveness"]["antispoof_cnn"], True)  # the good pair was NOT saved

    def test_mixed_forms_rejected(self):
        self.assertEqual(self._run("liveness.antispoof_cnn=true", "security.policy_sudo"), 1)

    def test_empty_value_clears_to_default(self):
        self._run("security.policy_sudo=strict")
        self.assertEqual(self._run("security.policy_sudo="), 0)   # the app's "Same as overall"
        self.assertEqual(cfgmod.load()["security"]["policy_sudo"], "")

    def test_empty_shows_config(self):
        self.assertEqual(self._run(), 0)


if __name__ == "__main__":
    unittest.main()
