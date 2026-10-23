import unittest

from faceup import paths


class Prefix(unittest.TestCase):
    def test_follows_package_location(self):
        self.assertEqual(paths.prefix("/usr/lib/faceup", exists=lambda p: False), "/usr")
        self.assertEqual(paths.prefix("/usr/local/lib/faceup", exists=lambda p: True), "/usr/local")

    def test_checkout_uses_existing_install(self):
        self.assertEqual(paths.prefix("/home/x/faceup", exists=lambda p: p == "/usr/bin/faceup-auth"), "/usr")
        self.assertEqual(paths.prefix("/home/x/faceup", exists=lambda p: True), "/usr/local")
        self.assertEqual(paths.prefix("/home/x/faceup", exists=lambda p: False), "/usr/local")

    def test_unit_either_location(self):
        self.assertTrue(paths.unit_installed(exists=lambda p: p == "/usr/lib/systemd/system/faceup.service"))
        self.assertFalse(paths.unit_installed(exists=lambda p: False))


if __name__ == "__main__":
    unittest.main()


class StalePrefixLine(unittest.TestCase):
    def test_pam_exec_line_from_other_prefix_is_rewritten(self):
        from faceup.engine import pam
        other = "/usr" if pam.ENGINE.startswith("/usr/local/") else "/usr/local"
        stale = f"auth\tsufficient\t\t\tpam_exec.so quiet stdout seteuid {other}/bin/faceup-auth\t# faceup\n"
        out = pam.render_enabled(stale + "auth\t[success=1 default=ignore]\tpam_unix.so\n", native=False)
        self.assertIn(pam.ENGINE, out)
        self.assertNotIn(f"{other}/bin/faceup-auth", out)
