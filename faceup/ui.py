"""Tiny terminal UI helpers — no dependencies, degrades to plain text when piped."""
from __future__ import annotations

import os
import sys

_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _COLOR else s


def bold(s: str) -> str:   return _c("1", s)
def dim(s: str) -> str:    return _c("2", s)
def green(s: str) -> str:  return _c("32", s)
def yellow(s: str) -> str: return _c("33", s)
def red(s: str) -> str:    return _c("31", s)
def cyan(s: str) -> str:   return _c("36", s)


def banner(title: str, sub: str = "") -> None:
    print()
    print(bold(cyan("▌ " + title)))
    if sub:
        print(dim("  " + sub))


def section(title: str) -> None:
    print()
    print(bold(title))
    print(dim("─" * min(len(title) + 2, 60)))


def ok(msg: str) -> None:   print(f"  {green('✔')} {msg}")
def warn(msg: str) -> None: print(f"  {yellow('!')} {msg}")
def fail(msg: str) -> None: print(f"  {red('✘')} {msg}")
def info(msg: str) -> None: print(f"  {dim('·')} {msg}")
def tip(msg: str) -> None:  print(f"    {dim('→')} {msg}")


def confirm(question: str, default: bool = True) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    try:
        a = input(f"  {bold('?')} {question} {dim(hint)} ").strip().lower()
    except EOFError:
        return default
    if not a:
        return default
    return a.startswith("y")


def ask(question: str, default: str = "") -> str:
    hint = f" {dim('[' + default + ']')}" if default else ""
    try:
        a = input(f"  {bold('?')} {question}{hint} ").strip()
    except EOFError:
        return default
    return a or default


def choose(question: str, options: list[str], default: int = 0) -> int:
    print(f"  {bold('?')} {question}")
    for i, o in enumerate(options):
        marker = "●" if i == default else "○"
        print(f"      {marker} {i + 1}. {o}")
    while True:
        a = ask("Choice", str(default + 1))
        if a.isdigit() and 1 <= int(a) <= len(options):
            return int(a) - 1
        warn("Enter a number from the list.")


def pause(msg: str = "Press Enter when ready") -> None:
    try:
        input(f"  {dim('…')} {msg} ")
    except EOFError:
        pass
