"""The maintainer release toolchain (tools/changelog.py + tools/release.py).

Stdlib, no camera, no git, no disk writes for the mutation logic — every
``bump_*`` / ``prepend_*`` is a pure text transform and ``assemble`` is
exercised against a temp directory of fragments.
"""
from __future__ import annotations

import datetime
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import changelog  # noqa: E402
import release  # noqa: E402


def _write(d, name, text):
    with open(os.path.join(d, name), "w", encoding="utf-8") as fh:
        fh.write(text)


class Assemble(unittest.TestCase):
    def test_empty_dir_is_empty(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(changelog.assemble(d), "")

    def test_missing_dir_is_empty(self):
        self.assertEqual(changelog.assemble("/no/such/dir/at/all"), "")

    def test_readme_and_gitkeep_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            _write(d, "README.md", "# docs")
            _write(d, ".gitkeep", "")
            self.assertEqual(changelog.assemble(d), "")

    def test_groups_under_headings_in_order(self):
        with tempfile.TemporaryDirectory() as d:
            # Deliberately out of heading order on disk.
            _write(d, "x.fixed.md", "- a bug is gone")
            _write(d, "y.added.md", "- shiny new thing")
            _write(d, "z.security.md", "- a hole is closed")
            out = changelog.assemble(d)
            self.assertEqual(
                out,
                "### Added\n- shiny new thing\n\n"
                "### Fixed\n- a bug is gone\n\n"
                "### Security\n- a hole is closed",
            )

    def test_all_six_types_in_canonical_order(self):
        with tempfile.TemporaryDirectory() as d:
            for t in ("packaging", "docs", "security", "fixed", "changed", "added"):
                _write(d, f"frag.{t}.md", f"- {t} line")
            out = changelog.assemble(d)
            headings = [ln for ln in out.splitlines() if ln.startswith("### ")]
            self.assertEqual(
                headings,
                ["### Added", "### Changed", "### Fixed",
                 "### Security", "### Docs", "### Packaging"],
            )

    def test_normalises_bullets(self):
        with tempfile.TemporaryDirectory() as d:
            # a bare line, a "* " bullet, a "- " bullet, blank lines, and an internal hyphen
            _write(d, "frag.fixed.md", "no marker\n\n* star marker\n- dash marker\nkeep-the-hyphen\n")
            out = changelog.assemble(d)
            self.assertEqual(
                out,
                "### Fixed\n"
                "- no marker\n"
                "- star marker\n"
                "- dash marker\n"
                "- keep-the-hyphen",
            )

    def test_multiple_fragments_same_type_sorted_by_name(self):
        with tempfile.TemporaryDirectory() as d:
            _write(d, "b-second.fixed.md", "- second")
            _write(d, "a-first.fixed.md", "- first")
            out = changelog.assemble(d)
            self.assertEqual(out, "### Fixed\n- first\n- second")

    def test_unknown_type_raises(self):
        with tempfile.TemporaryDirectory() as d:
            _write(d, "oops.bugfix.md", "- typo type")
            with self.assertRaises(ValueError):
                changelog.assemble(d)

    def test_fragment_files_excludes_reserved(self):
        with tempfile.TemporaryDirectory() as d:
            _write(d, "README.md", "x")
            _write(d, ".gitkeep", "")
            _write(d, "real.added.md", "- yes")
            files = changelog.fragment_files(d)
            self.assertEqual([os.path.basename(f) for f in files], ["real.added.md"])


class Slug(unittest.TestCase):
    def test_slugify(self):
        self.assertEqual(changelog.slugify("Fix the Night Unlock!"), "fix-the-night-unlock")

    def test_slugify_fallback(self):
        self.assertEqual(changelog.slugify("!!!"), "entry")


class DateHelpers(unittest.TestCase):
    def test_rpm_date_weekday_is_correct(self):
        # 2026-10-28 is a Wednesday (matches the shipped spec stanza).
        self.assertEqual(release.rpm_date(datetime.date(2026, 10, 28)),
                         "Wed Oct 28 2026")

    def test_rpm_date_another_weekday(self):
        # 2026-10-25 is a Sunday.
        self.assertEqual(release.rpm_date(datetime.date(2026, 10, 25)),
                         "Sun Oct 25 2026")

    def test_deb_date_rfc2822(self):
        self.assertEqual(release.deb_date(datetime.date(2026, 10, 28)),
                         "Wed, 28 Oct 2026 12:00:00 +0000")


class BumpInit(unittest.TestCase):
    def test_bumps_version(self):
        src = '"""ward."""\n__version__ = "0.5.0"\n\n\ndef main():\n    pass\n'
        out = release.bump_init(src, "0.6.0")
        self.assertIn('__version__ = "0.6.0"', out)
        self.assertNotIn("0.5.0", out)

    def test_missing_raises(self):
        with self.assertRaises(ValueError):
            release.bump_init("no version here\n", "0.6.0")


class BumpPyproject(unittest.TestCase):
    def test_bumps_project_version_not_target_version(self):
        src = (
            '[project]\n'
            'name = "ward"\n'
            'version = "0.5.0"\n'
            '[tool.ruff]\n'
            'target-version = "py310"\n'
        )
        out = release.bump_pyproject(src, "0.6.0")
        self.assertIn('version = "0.6.0"', out)
        self.assertIn('target-version = "py310"', out)   # untouched
        self.assertNotIn('version = "0.5.0"', out)


class BumpPkgbuild(unittest.TestCase):
    def test_bumps_pkgver_keeps_pkgrel(self):
        src = "pkgname=ward\npkgver=0.5.0\npkgrel=1\n"
        out = release.bump_pkgbuild(src, "0.6.0")
        self.assertIn("pkgver=0.6.0", out)
        self.assertIn("pkgrel=1", out)


class BumpSpec(unittest.TestCase):
    def test_bumps_version_and_prepends_dated_stanza(self):
        src = (
            "Name:           ward\n"
            "Version:        0.5.0\n"
            "Release:        1%{?dist}\n"
            "%changelog\n"
            "* Wed Oct 28 2026 octave <0ctave@users.noreply.github.com> - 0.5.0-1\n"
            "- Release 0.5.0\n"
        )
        out = release.bump_spec(src, "0.6.0", datetime.date(2026, 11, 2))
        self.assertIn("Version:        0.6.0", out)
        # 2026-11-02 is a Monday — the new stanza must use the right weekday.
        self.assertIn(
            "* Mon Nov 02 2026 octave <0ctave@users.noreply.github.com> - 0.6.0-1",
            out,
        )
        self.assertIn("- see CHANGELOG.md", out)
        # New stanza sits directly under %changelog, above the old one.
        self.assertLess(out.index("0.6.0-1"), out.index("0.5.0-1"))
        # The %changelog header is not duplicated.
        self.assertEqual(out.count("%changelog"), 1)


class BumpDebian(unittest.TestCase):
    def test_prepends_native_stanza(self):
        src = (
            "ward (0.5.0) unstable; urgency=medium\n\n"
            "  * Release 0.5.0 (see CHANGELOG.md for the full list of changes).\n\n"
            " -- octave <0ctave@users.noreply.github.com>  Wed, 28 Oct 2026 12:00:00 +0000\n"
        )
        out = release.bump_debian_changelog(src, "0.6.0", datetime.date(2026, 11, 2))
        lines = out.splitlines()
        # native: version carries no Debian revision (-1)
        self.assertEqual(lines[0], "ward (0.6.0) unstable; urgency=medium")
        self.assertIn("  * Release 0.6.0 (see CHANGELOG.md).", out)
        self.assertIn(
            " -- octave <0ctave@users.noreply.github.com>  Mon, 02 Nov 2026 12:00:00 +0000",
            out,
        )
        # old stanza preserved below
        self.assertIn("ward (0.5.0) unstable; urgency=medium", out)
        self.assertLess(out.index("0.6.0"), out.index("0.5.0"))


class PrependChangelog(unittest.TestCase):
    def test_prepends_above_top_section(self):
        src = (
            "# Changelog\n\n"
            "## 0.5.0 — 2026-10-28\n\n"
            "### Fixed\n- old thing\n"
        )
        body = "### Added\n- new thing"
        out = release.prepend_changelog(src, "0.6.0", datetime.date(2026, 11, 2), body)
        self.assertIn("## 0.6.0 — 2026-11-02", out)
        self.assertIn("### Added\n- new thing", out)
        # new section comes before the old one, and old history is preserved verbatim
        self.assertLess(out.index("## 0.6.0"), out.index("## 0.5.0"))
        self.assertIn("## 0.5.0 — 2026-10-28", out)
        self.assertIn("### Fixed\n- old thing", out)
        # the title stays at the very top
        self.assertTrue(out.startswith("# Changelog\n"))

    def test_empty_body_section_has_no_subheadings(self):
        src = "# Changelog\n\n## 0.5.0 — 2026-10-28\n\n### Fixed\n- old\n"
        out = release.prepend_changelog(src, "0.6.0", datetime.date(2026, 11, 2), "")
        self.assertIn("## 0.6.0 — 2026-11-02", out)
        # nothing between the new header and the old section but a blank line
        between = out.split("## 0.6.0 — 2026-11-02\n", 1)[1].split("## 0.5.0", 1)[0]
        self.assertEqual(between.strip(), "")


class SemverValidation(unittest.TestCase):
    def test_accepts_semver(self):
        self.assertIsNotNone(release.SEMVER_RE.match("1.2.3"))

    def test_rejects_non_semver(self):
        for bad in ("1.2", "v1.2.3", "1.2.3-rc1", "1.2.3.4", "x"):
            self.assertIsNone(release.SEMVER_RE.match(bad), bad)


if __name__ == "__main__":
    unittest.main()
