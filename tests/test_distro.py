"""Install hints per distro family: nobody on Fedora or Arch is told to run apt."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from faceup import distro  # noqa: E402

UBUNTU = 'NAME="Ubuntu"\nID=ubuntu\nID_LIKE=debian\n'
FEDORA = 'NAME="Fedora Linux"\nID=fedora\n'
ARCH = 'NAME="Arch Linux"\nID=arch\n'
TUMBLEWEED = 'NAME="openSUSE Tumbleweed"\nID="opensuse-tumbleweed"\nID_LIKE="opensuse suse"\n'
MINT = 'ID=linuxmint\nID_LIKE="ubuntu debian"\n'


class Distro(unittest.TestCase):
    def test_family(self):
        self.assertEqual([distro.family(t) for t in (UBUNTU, FEDORA, ARCH, TUMBLEWEED, MINT, "ID=nixos\n")],
                         ["debian", "fedora", "arch", "suse", "debian", ""])

    def test_hints_use_the_right_tool(self):
        self.assertTrue(distro.hint("opencv", os_release=UBUNTU).startswith("sudo apt install python3-opencv"))
        self.assertTrue(distro.hint("opencv", os_release=FEDORA).startswith("sudo dnf install"))
        self.assertIn("pacman -S python-opencv", distro.hint("opencv", "gtk", os_release=ARCH))
        self.assertNotIn("apt", distro.hint("gtk", os_release=TUMBLEWEED))

    def test_unpackaged_pieces_get_a_note_not_a_bogus_package(self):
        h = distro.hint("dlib", "dlib-models", os_release=ARCH)
        self.assertIn("ArcFace", h); self.assertIn("dlib.net", h); self.assertNotIn("pacman", h)

    def test_unknown_distro(self):
        self.assertIn("package manager", distro.hint("opencv", os_release="ID=nixos\n"))


if __name__ == "__main__":
    unittest.main()
