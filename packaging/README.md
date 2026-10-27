# Packaging & distribution

One-line install is the biggest adoption lever. The build inputs here are ready; the uploads need
**your** maintainer accounts (they can't be automated from CI). Bump the version in each file on a
release, then follow the per-distro steps.

| File | Target | Upload channel |
|---|---|---|
| `PKGBUILD` | Arch / derivatives | **AUR** |
| `ward.spec` | Fedora / RHEL / openSUSE | **Fedora COPR** (and/or OBS) |
| `build-deb.sh` → `.deb` | Debian / Ubuntu | **Launchpad PPA** (and the GitHub release the `release` workflow builds) |
| `stage.sh` | the shared file layout all of the above call | — |

All of them install the same layout under `/usr` and **never edit PAM** — the user runs `ward setup`
after install. Keep `pkgver`/`Version`/`pkgrel` in step with the git tag.

## AUR (Arch) — do this first; `yay -S ward` is the headline install

One-time: an [aur.archlinux.org](https://aur.archlinux.org) account with your SSH key added.

```bash
cd packaging
makepkg --printsrcinfo > .SRCINFO          # AUR requires this; regenerate whenever PKGBUILD changes
git clone ssh://aur@aur.archlinux.org/ward.git aur-ward
cp PKGBUILD .SRCINFO aur-ward/
cd aur-ward && git add PKGBUILD .SRCINFO && git commit -m 'Initial import: ward 0.4.0' && git push
```

The `source=(...)` in `PKGBUILD` should point at the release tarball (`…/archive/v$pkgver.tar.gz`)
with a real `sha256sums` once a tag exists — build it locally with `makepkg -si` before pushing.

## Fedora COPR

One-time: a [copr.fedorainfracloud.org](https://copr.fedorainfracloud.org) project.

```bash
# build a source RPM from a release tarball, then upload to COPR (web UI or copr-cli)
rpmbuild -bs packaging/ward.spec            # needs the tarball in ~/rpmbuild/SOURCES
copr-cli build <you>/ward ~/rpmbuild/SRPMS/ward-*.src.rpm
```

`python3-dlib` is packaged only on openSUSE, so on Fedora the spec should `Recommends` ArcFace or
leave the user to `ward recogniser install` — `ward doctor` now flags a missing dlib embedder.

## Ubuntu / Debian PPA

One-time: a [Launchpad](https://launchpad.net) account + GPG key. `build-deb.sh` produces a binary
`.deb`; for a PPA you upload a *source* package (`debuild -S`) and `dput ppa:<you>/ward …`. The
GitHub `release` workflow already builds and attaches a binary `.deb` on each `v*` tag.

## Flathub (later)

A Flatpak needs a manifest and sandbox holes for V4L2 (`--device=all`) and PAM, which is a larger
piece — track it as a separate effort once the native packages are live.

## On each release

1. Bump the version in `PKGBUILD` (`pkgver`), `ward.spec` (`Version`), and tag `vX.Y.Z`.
2. The `release` workflow builds the `.deb` and creates the GitHub release.
3. Regenerate `.SRCINFO`, update `sha256sums`, and push AUR; rebuild COPR; `dput` the PPA.
