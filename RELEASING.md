# Releasing Ward

Ward's changelog and version live in several files. Two small **stdlib-only**
maintainer tools under `tools/` keep them in sync so a changelog entry never
conflicts and a release is one command. These tools are not part of the shipped
`ward` package and are not `ward` subcommands — they run from a checkout.

## Add a changelog entry (every user-facing PR)

Don't edit `CHANGELOG.md` by hand. Drop a fragment instead:

```
python3 tools/changelog.py add -t fixed -m "night unlock no longer stalls on a slow TPM"
```

`-t` is the change type — one of `added changed fixed security docs packaging`
(each becomes a `###` heading). This writes `changelog.d/<slug>.<type>.md`.
Because every PR adds its **own** file, parallel PRs never collide on the
changelog. See [`changelog.d/README.md`](changelog.d/README.md) for the format.

Preview what the next release will say:

```
python3 tools/changelog.py preview
```

The `changelog` CI check (`.github/workflows/changelog.yml`) fails a PR that
adds no fragment unless it carries the **`skip-changelog`** label (use that for
pure refactors or CI tweaks with nothing for users to read).

## Cut a release (one command)

```
python3 tools/release.py 0.6.0
```

This, on a clean tree:

1. assembles `changelog.d/` into a new `## 0.6.0 — <today>` section and
   **prepends** it above the current top section of `CHANGELOG.md` (old
   entries are never rewritten);
2. bumps the version in all five authored files —
   `ward/__init__.py`, `pyproject.toml`, `packaging/PKGBUILD` (`pkgver`),
   `packaging/ward.spec` (`Version:` **and** a correctly-dated `%changelog`
   stanza) and `debian/changelog` (a `3.0 (native)` stanza, no `-1` revision);
3. deletes the consumed fragments (keeps `README.md` and `.gitkeep`);
4. `git add -A`, commits `release: 0.6.0`, and creates the annotated tag
   `v0.6.0`.

`build-deb.sh` and `tests/containers/run.sh` derive the version from
`ward/__init__.py`, so they need no edit.

Useful flags:

- `--dry-run` — print the assembled section and every file that would change,
  but write nothing, commit nothing, tag nothing. Run this first.
- `--date YYYY-MM-DD` — override the release date (default: today).
- `--allow-empty` — release even with no pending fragments.

### Push the tag

```
git push && git push origin v0.6.0
```

The pushed `v*` tag triggers [`.github/workflows/release.yml`](.github/workflows/release.yml):
it runs the test suite, builds the `.deb` with `packaging/build-deb.sh`, and
publishes the GitHub Release with checksums.

## Update the downstream channels

After the tag and GitHub Release exist, publish to AUR, Fedora COPR and the
Ubuntu PPA as described in [`packaging/README.md`](packaging/README.md)
(`updpkgsums` + `.SRCINFO` for AUR, the SCM rebuild for COPR, `debuild -S`
per Ubuntu series for the PPA).
