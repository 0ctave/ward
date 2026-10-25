# Contributing to Ward

Thanks for helping make face unlock on Linux reliable. By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md). The most valuable contributions are
**camera quirks from other laptops** and **portability fixes** — ward is developed on one machine
and every new module teaches it something.

## Ground rules (the project's golden rules)

1. **Distro Python only, zero pip dependencies.** ward runs under `/usr/bin/python3` with the
   distro's `python3-opencv`, `python3-dlib` and `dlib-models`. A virtualenv can't see them. Never
   add a `requirements.txt` or a pip install step.
2. **Tests stay green and need no camera:** `python3 -m unittest discover -s tests`. The suite runs
   the full pipeline on recorded/synthetic frames with a patched detector. Add a test with every
   change; extend `tests/test_compat.py` when you add a profile-dependent path.
3. **Fail open.** Every authentication failure must fall through to the password. See
   [SECURITY.md](SECURITY.md). A change that could lock a user out will not be merged.
4. Keep the CLI the single implementation: the GTK app and the PAM path are drivers over it.

## Getting set up

```bash
git clone https://github.com/0ctave/ward && cd ward
python3 -m unittest discover -s tests      # must pass before you change anything
sudo ./install.sh                          # to try it on your hardware
ward doctor                              # diagnoses cameras, emitter, engine, PAM
```

## Reporting a camera / hardware issue

Open an issue with the **full `ward doctor` output**, your distro and desktop, the laptop model,
and `lsusb` / `v4l2-ctl --list-devices`. If recognition or liveness misbehaves, attach a recording
made with `sudo ward lab --record-frames <dir>` (frames of **your own** face only — never commit
them; see below) so the problem can be replayed without your hardware.

## Pull requests

- Match the surrounding style; keep diffs focused. One concern per PR.
- Run the test suite and `gcc -fsyntax-only data/pam_ward.c` if you touch the PAM module.
- Update `docs/DESIGN.md` when you change a mechanism and `CHANGELOG.md` (Unreleased) always.
- New bundled models must be permissively licensed, pinned by sha256, and small; their license and
  notice go in `ward/models/`. Non-redistributable models (e.g. ArcFace) are downloaded, not
  bundled.

## Never commit personal or biometric data

Face embeddings, the model store, the attempt log, the keyring credential, recorded frames and any
research dataset are all git-ignored and must stay out of the repository and its history. If you
think you may have added one, stop and say so in the PR before pushing.

## Scope ideas

See [docs/ROADMAP.md](docs/ROADMAP.md). High-value areas: a calibrated FAR study with multiple
subjects, a second fast liveness voter, the emitter-differential imaging path, and greeter/distro
coverage beyond the tested matrix.

By contributing you agree your work is licensed under the project's [MIT license](LICENSE).
