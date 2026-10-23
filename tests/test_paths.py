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
