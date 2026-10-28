#!/usr/bin/env python3
"""Ward changelog fragments — a tiny, stdlib-only towncrier.

MAINTAINER TOOL. This lives in tools/ and is **not** part of the shipped ward
package: it is never imported by the engine and is not a `ward` subcommand.

Every pull request drops its own file under ``changelog.d/`` so parallel PRs
never touch the same lines of ``CHANGELOG.md`` (merge conflicts in a shared
changelog are the whole reason this exists). At release time
``tools/release.py`` assembles the fragments into a dated section.

Fragment file name: ``<slug>.<type>.md`` where <type> is one of
``added changed fixed security docs packaging``. The content is one or more
markdown bullet lines; a leading ``-`` / ``*`` is optional (it is normalised).

Usage:
    python3 tools/changelog.py add -t fixed -m "message" [--slug name]
    python3 tools/changelog.py preview
"""
from __future__ import annotations

import argparse
import os
import re
import sys

# type -> the "### Heading" it lands under, in the order sections are emitted.
TYPE_HEADINGS = {
    "added": "Added",
    "changed": "Changed",
    "fixed": "Fixed",
    "security": "Security",
    "docs": "Docs",
    "packaging": "Packaging",
}
TYPE_ORDER = list(TYPE_HEADINGS)

# Default fragments directory, resolved relative to the repo root (this file's parent's parent).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(REPO_ROOT, "changelog.d")

# Files that live in changelog.d but are never fragments.
RESERVED = {"README.md", ".gitkeep"}


# ---------------------------------------------------------------------------
# Pure helpers (no I/O except the one dir read in assemble) — the test imports
# these directly and release.py reuses assemble().
# ---------------------------------------------------------------------------

def slugify(message: str) -> str:
    """A short, filesystem-safe slug derived from a message."""
    words = re.findall(r"[a-z0-9]+", message.lower())
    slug = "-".join(words)[:50].strip("-")
    return slug or "entry"


def parse_fragment_type(filename: str) -> str | None:
    """The <type> of ``<slug>.<type>.md``, or None if it isn't a fragment name."""
    if not filename.endswith(".md") or filename in RESERVED:
        return None
    stem = filename[: -len(".md")]
    if "." not in stem:
        return None
    return stem.rsplit(".", 1)[1]


def normalise_bullets(text: str) -> list[str]:
    """Turn fragment text into ``- ...`` bullet lines (one per non-blank line)."""
    bullets = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = line.lstrip("-*").strip()
        if line:
            bullets.append("- " + line)
    return bullets


def assemble(fragments_dir: str) -> str:
    """Assemble every fragment in *fragments_dir* into a markdown section body.

    Fragments are grouped under their ``### Heading`` in TYPE_ORDER; within a
    group they are ordered by file name (stable, conflict-free). Returns the
    empty string for an empty directory. Reads the directory; no other I/O.
    """
    try:
        names = sorted(os.listdir(fragments_dir))
    except FileNotFoundError:
        return ""

    by_type: dict[str, list[str]] = {t: [] for t in TYPE_ORDER}
    for name in names:
        ftype = parse_fragment_type(name)
        if ftype is None:
            continue
        if ftype not in TYPE_HEADINGS:
            raise ValueError(
                f"{name}: unknown changelog type {ftype!r} "
                f"(expected one of {', '.join(TYPE_ORDER)})"
            )
        with open(os.path.join(fragments_dir, name), encoding="utf-8") as fh:
            by_type[ftype].extend(normalise_bullets(fh.read()))

    blocks = []
    for ftype in TYPE_ORDER:
        bullets = by_type[ftype]
        if bullets:
            blocks.append("### " + TYPE_HEADINGS[ftype] + "\n" + "\n".join(bullets))
    return "\n\n".join(blocks)


def fragment_files(fragments_dir: str) -> list[str]:
    """Absolute paths of the real fragment files (excludes README/.gitkeep)."""
    try:
        names = sorted(os.listdir(fragments_dir))
    except FileNotFoundError:
        return []
    return [
        os.path.join(fragments_dir, n)
        for n in names
        if parse_fragment_type(n) is not None
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_add(args: argparse.Namespace) -> int:
    if args.type not in TYPE_HEADINGS:
        print(
            f"error: unknown type {args.type!r}; expected one of {', '.join(TYPE_ORDER)}",
            file=sys.stderr,
        )
        return 2
    slug = args.slug or slugify(args.message)
    os.makedirs(args.dir, exist_ok=True)
    path = os.path.join(args.dir, f"{slug}.{args.type}.md")
    if os.path.exists(path):
        print(f"error: {path} already exists (pick a different --slug)", file=sys.stderr)
        return 1
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("- " + args.message.strip() + "\n")
    print(path)
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    body = assemble(args.dir)
    if body:
        print(body)
    else:
        print("(no pending changelog fragments)", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tools/changelog.py",
        description="Manage Ward's conflict-free changelog fragments (stdlib-only maintainer tool).",
    )
    p.add_argument(
        "--dir",
        default=DEFAULT_DIR,
        help="fragments directory (default: changelog.d/ at the repo root)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("add", help="create a new changelog fragment")
    a.add_argument("-t", "--type", required=True, choices=TYPE_ORDER, help="change type")
    a.add_argument("-m", "--message", required=True, help="the changelog bullet text")
    a.add_argument("--slug", help="file slug (default: derived from the message)")
    a.set_defaults(func=cmd_add)

    pv = sub.add_parser("preview", help="print the assembled pending section")
    pv.set_defaults(func=cmd_preview)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
