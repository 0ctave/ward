#!/usr/bin/env python3
"""Cut a Ward release in one command — stdlib only.

MAINTAINER TOOL. Lives in tools/, is **not** part of the shipped ward package,
and is **not** a `ward` subcommand. It never imports the engine.

    python3 tools/release.py X.Y.Z [--dry-run] [--date YYYY-MM-DD] [--allow-empty]

What a release is (see RELEASING.md):
  1. assemble changelog.d/ into a new "## X.Y.Z — <date>" section, prepended
     above the current top section of CHANGELOG.md (old entries untouched);
  2. bump the version in the five authored files
     (ward/__init__.py, pyproject.toml, packaging/PKGBUILD, packaging/ward.spec
     — plus a dated %changelog stanza — and debian/changelog — plus a native
     stanza);
  3. delete the consumed fragments (keep README.md + .gitkeep);
  4. git add -A, commit "release: X.Y.Z", annotated tag vX.Y.Z;
  5. print the push + per-channel next steps.

build-deb.sh and tests/containers/run.sh DERIVE the version from
ward/__init__.py, so they need no edit.

The file-mutation logic is factored into pure ``bump_*`` / ``prepend_*``
functions (old text -> new text) so tests exercise them without touching disk
or git.
"""
from __future__ import annotations

import argparse
import datetime
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import changelog  # noqa: E402  (sibling module in tools/)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAINTAINER = "octave <0ctave@users.noreply.github.com>"
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")

# Locale-independent names (strftime honours $LC_TIME; a wrong weekday fails rpmbuild).
_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_MONTHS = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------------------
# Date formatting (pure)
# ---------------------------------------------------------------------------

def _parse_date(date_str: str | None) -> datetime.date:
    if date_str is None:
        return datetime.date.today()
    return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()


def iso_date(d: datetime.date) -> str:
    return d.isoformat()  # YYYY-MM-DD


def rpm_date(d: datetime.date) -> str:
    """RPM %changelog date, e.g. 'Wed Oct 28 2026' — weekday matched to the date."""
    return f"{_WEEKDAYS[d.weekday()]} {_MONTHS[d.month]} {d.day:02d} {d.year}"


def deb_date(d: datetime.date) -> str:
    """Debian changelog RFC2822 date, e.g. 'Wed, 28 Oct 2026 12:00:00 +0000'."""
    return f"{_WEEKDAYS[d.weekday()]}, {d.day:02d} {_MONTHS[d.month]} {d.year} 12:00:00 +0000"


# ---------------------------------------------------------------------------
# Pure text mutations — each takes the current file text and returns the new text.
# ---------------------------------------------------------------------------

def bump_init(text: str, version: str) -> str:
    """ward/__init__.py: __version__ = "X.Y.Z"."""
    new, n = re.subn(
        r'^__version__\s*=\s*"[^"]*"',
        f'__version__ = "{version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise ValueError("ward/__init__.py: no __version__ line found")
    return new


def bump_pyproject(text: str, version: str) -> str:
    """pyproject.toml: the [project] version = "X.Y.Z" (not target-version)."""
    new, n = re.subn(
        r'^version\s*=\s*"[^"]*"',
        f'version = "{version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise ValueError("pyproject.toml: no version line found")
    return new


def bump_pkgbuild(text: str, version: str) -> str:
    """packaging/PKGBUILD: pkgver=X.Y.Z (pkgrel stays 1)."""
    new, n = re.subn(
        r"^pkgver=.*$",
        f"pkgver={version}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise ValueError("PKGBUILD: no pkgver line found")
    return new


def bump_spec(text: str, version: str, d: datetime.date) -> str:
    """packaging/ward.spec: bump Version and prepend a dated %changelog stanza."""
    new, n = re.subn(
        r"^(Version:\s*).*$",
        lambda m: m.group(1) + version,
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise ValueError("ward.spec: no Version line found")

    stanza = (
        f"* {rpm_date(d)} {MAINTAINER} - {version}-1\n"
        "- see CHANGELOG.md\n"
    )
    new, n = re.subn(
        r"^%changelog\n",
        "%changelog\n" + stanza + "\n",
        new,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise ValueError("ward.spec: no %changelog section found")
    return new


def bump_debian_changelog(text: str, version: str, d: datetime.date) -> str:
    """debian/changelog: prepend a 3.0 (native) stanza (version carries no -1)."""
    stanza = (
        f"ward ({version}) unstable; urgency=medium\n\n"
        f"  * Release {version} (see CHANGELOG.md).\n\n"
        f" -- {MAINTAINER}  {deb_date(d)}\n"
    )
    return stanza + "\n" + text


def prepend_changelog(text: str, version: str, d: datetime.date, body: str) -> str:
    """CHANGELOG.md: insert '## X.Y.Z — <date>' + *body* above the top '## ' section."""
    section = f"## {version} — {iso_date(d)}\n"
    if body:
        section += "\n" + body + "\n"

    lines = text.splitlines(keepends=True)
    insert_at = None
    for i, line in enumerate(lines):
        if line.startswith("## "):
            insert_at = i
            break
    if insert_at is None:
        # No prior release section (e.g. only the '# Changelog' title) — append.
        sep = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
        return text + sep + section
    head = "".join(lines[:insert_at])
    tail = "".join(lines[insert_at:])
    return head + section + "\n" + tail


# ---------------------------------------------------------------------------
# A release plan: compute every change as (path, old_text, new_text) without
# writing anything, so --dry-run and the real run share one code path.
# ---------------------------------------------------------------------------

FILES = {
    "init": "ward/__init__.py",
    "pyproject": "pyproject.toml",
    "pkgbuild": "packaging/PKGBUILD",
    "spec": "packaging/ward.spec",
    "deb": "debian/changelog",
    "changelog": "CHANGELOG.md",
}


def _read(root: str, rel: str) -> str:
    with open(os.path.join(root, rel), encoding="utf-8") as fh:
        return fh.read()


def compute_plan(root: str, version: str, d: datetime.date, body: str) -> list[tuple[str, str, str]]:
    """Return [(rel_path, old_text, new_text), ...] for every file the release edits."""
    plan = []
    for rel, new in (
        (FILES["init"], bump_init(_read(root, FILES["init"]), version)),
        (FILES["pyproject"], bump_pyproject(_read(root, FILES["pyproject"]), version)),
        (FILES["pkgbuild"], bump_pkgbuild(_read(root, FILES["pkgbuild"]), version)),
        (FILES["spec"], bump_spec(_read(root, FILES["spec"]), version, d)),
        (FILES["deb"], bump_debian_changelog(_read(root, FILES["deb"]), version, d)),
        (FILES["changelog"], prepend_changelog(_read(root, FILES["changelog"]), version, d, body)),
    ):
        plan.append((rel, _read(root, rel), new))
    return plan


# ---------------------------------------------------------------------------
# git + orchestration (impure)
# ---------------------------------------------------------------------------

def _git(root: str, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", root, *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _tree_clean(root: str) -> bool:
    return _git(root, "status", "--porcelain").strip() == ""


def _tag_exists(root: str, tag: str) -> bool:
    out = subprocess.run(
        ["git", "-C", root, "tag", "--list", tag],
        capture_output=True, text=True,
    ).stdout
    return out.strip() != ""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="tools/release.py",
        description="Cut a Ward release: assemble the changelog, bump the five files, commit and tag.",
    )
    p.add_argument("version", help="the new version, X.Y.Z")
    p.add_argument("--date", help="override the release date (YYYY-MM-DD); default today")
    p.add_argument("--dry-run", action="store_true", help="print everything, write/commit/tag nothing")
    p.add_argument("--allow-empty", action="store_true", help="release even with no pending fragments")
    args = p.parse_args(argv)

    version = args.version
    if not SEMVER_RE.match(version):
        print(f"error: {version!r} is not a semver X.Y.Z", file=sys.stderr)
        return 2

    root = REPO_ROOT
    tag = f"v{version}"

    if not args.dry_run:
        if not _tree_clean(root):
            print("error: working tree is not clean; commit or stash first", file=sys.stderr)
            return 1
        if _tag_exists(root, tag):
            print(f"error: tag {tag} already exists", file=sys.stderr)
            return 1

    d = _parse_date(args.date)
    frag_dir = os.path.join(root, "changelog.d")
    body = changelog.assemble(frag_dir)
    fragments = changelog.fragment_files(frag_dir)

    if not fragments and not args.allow_empty:
        print(
            "error: no changelog fragments in changelog.d/ "
            "(add one with tools/changelog.py add, or pass --allow-empty)",
            file=sys.stderr,
        )
        return 1

    plan = compute_plan(root, version, d, body)
    section = f"## {version} — {iso_date(d)}\n" + ("\n" + body + "\n" if body else "")

    if args.dry_run:
        print(f"# DRY RUN — release {version} ({iso_date(d)})\n")
        print("## CHANGELOG.md section to be prepended:\n")
        print(section)
        print("## Files that would change:\n")
        for rel, _old, _new in plan:
            print(f"  - {rel}")
        if fragments:
            print("\n## Fragments that would be consumed:\n")
            for f in fragments:
                print(f"  - {os.path.relpath(f, root)}")
        print(f"\n## Would then: commit 'release: {version}' and tag {tag}")
        return 0

    for rel, _old, new in plan:
        with open(os.path.join(root, rel), "w", encoding="utf-8") as fh:
            fh.write(new)
    for f in fragments:
        os.remove(f)

    _git(root, "add", "-A")
    _git(root, "commit", "-m", f"release: {version}")
    _git(root, "tag", "-a", tag, "-m", f"Ward {version}")

    print(f"Released {version}: committed and tagged {tag}.\n")
    print("Next steps:")
    print("  git push && git push origin " + tag)
    print("    -> the release workflow (.github/workflows/release.yml) runs the tests,")
    print("       builds the .deb and publishes the GitHub Release.")
    print("  Then update the downstream channels per packaging/README.md:")
    print("    - AUR:  updpkgsums && makepkg --printsrcinfo > .SRCINFO (after the tag exists)")
    print("    - COPR: rebuild the SCM package at " + tag)
    print("    - PPA:  debuild -S -sa per Ubuntu series, then dput")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
