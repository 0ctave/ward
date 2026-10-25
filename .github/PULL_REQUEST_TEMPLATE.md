<!-- Thanks for contributing to Ward! See CONTRIBUTING.md for the ground rules. -->

## What this changes

<!-- A short description, and the issue it closes (e.g. "Closes #12"). -->

## Checklist

- [ ] `python3 -m unittest discover -s tests` passes (no camera needed), and I added/updated a test.
- [ ] No pip/virtualenv dependency added (Ward uses the distro Python only).
- [ ] Fail-open preserved — every authentication failure still falls back to the password.
- [ ] `ui.py` and `appui.py` are not merged.
- [ ] If I touched the PAM module: `gcc -fsyntax-only data/pam_ward.c` is clean.
- [ ] `CHANGELOG.md` (Unreleased) updated; `docs/DESIGN.md` updated if a mechanism changed.
- [ ] No personal or biometric data (face models, logs, recorded frames, datasets) committed.

## How I tested it

<!-- Commands run, hardware if relevant, and the result. -->
