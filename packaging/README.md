# Packaging ward

Everything here produces the **same install layout** through one script,
`packaging/stage.sh DESTDIR PREFIX UNITDIR`. It copies `ward/`, writes the
`/usr/bin/ward` and `/usr/bin/ward-auth` launchers (`python3 -I`), compiles
`pam_ward.so` with gcc into `PREFIX/lib/security`, and installs the desktop
entry, polkit action, icons and the systemd unit — all under `DESTDIR`,
touching **nothing** outside it: no PAM edits, no `systemctl`, no `/etc/ward`.

Those system-side steps happen *after* install, run by the user:

```
ward doctor        # diagnose first
sudo ward setup    # wires PAM (fail-open) and the rest — nothing in /etc/pam.d is touched at install time
```

**Golden constraints (every channel honours these):**

- **Zero pip.** Runtime deps are distro packages only. There is no
  `requirements.txt` and no Python build step.
- **`python3-dlib` is optional,** never a hard dependency. It is packaged only
  on openSUSE; everywhere else it is a `Recommends` / `optdepend`, and ArcFace
  works without it. So it is never in `Depends`/`depends`/`Requires`.
- **Packages never edit PAM.** The post-install only creates `/etc/ward/models`
  (0700) and `/var/log/ward.log`, and enables `ward.service` on first install.

Version lives in `ward/__init__.py` (`__version__`), `pyproject.toml`,
`packaging/PKGBUILD` (`pkgver`), `packaging/ward.spec` (`Version`) and
`debian/changelog`. `build-deb.sh` reads the version straight from
`ward/__init__.py`, so it needs no edit on a bump. Current release: **0.5.0**.

---

## Debian / Ubuntu

There are three Debian/Ubuntu channels. Pick by what you need.

### 1. Prebuilt `.deb` from a GitHub release — the simplest path

Each tagged release attaches a ready-to-install `.deb` (built by
`build-deb.sh`). For most users this is all they need:

```
sudo apt install ./ward_0.5.0-1_amd64.deb
```

`apt install ./file.deb` pulls the `Depends`/`Recommends` from your archive.

### 2. Build the binary `.deb` locally — offline, no debhelper

```
packaging/build-deb.sh            # writes ./dist/ward_<ver>-1_<arch>.deb
```

This uses only `dpkg-deb` + `gcc`, derives the version from
`ward/__init__.py`, and reuses the maintainer scripts in `packaging/debian/`.
It is self-contained and does **not** use the `debian/` directory below.

### 3. Source package for `debuild` / a Launchpad PPA

The repo root carries a proper debhelper `debian/` directory for
`dpkg-buildpackage` / `debuild` (Launchpad builds from **source** and rejects
binary uploads). It reuses `packaging/stage.sh` and single-sources the
maintainer scripts from `packaging/debian/` (the `debian/{postinst,prerm,postrm}`
entries are symlinks), so there is no duplicated logic and channel 2 keeps
working untouched.

`debian/source/format` is **`3.0 (native)`** — ward is its own upstream
(code and packaging in one repo), so there is no separate `.orig` tarball and
the version carries no Debian revision (`0.5.0`, not `0.5.0-1`). The rationale
and the quilt alternative are in `debian/README.source`. `debian/rules`
overrides every `dh_auto_*` step to a no-op (there is no Python build — defeats
pybuild autodetection from `pyproject.toml`) and runs `stage.sh` in
`override_dh_auto_install`; `dh_installsystemd` is suppressed so the shared
cross-distro enable logic in the maintainer scripts stays the only copy.

```
# local sanity binary build
dpkg-buildpackage -b -us -uc          # ../ward_0.5.0_amd64.deb

# source build for a PPA (signed), then lint and upload
debuild -S -sa -k<your-gpg-key-email>
lintian -i ../ward_0.5.0_source.changes
dput ppa:0ctave/ward ../ward_0.5.0_source.changes
```

Needs `devscripts debhelper dput dpkg-dev gnupg`. For a **multi-series PPA**,
Launchpad refuses a repeated or lower version, so upload one source per series
with an ascending `~` suffix and the series codename as the distribution — edit
`debian/changelog` with `dch`, e.g. `ward (0.5.0~ppa1~noble1) noble; …` and
`ward (0.5.0~ppa1~jammy1) jammy; …` (`~` sorts before the plain `0.5.0`, so the
real release later supersedes them). The distribution field for a PPA must be
an Ubuntu series codename (`noble`, `jammy`, …), never `unstable`.

---

## Arch Linux (AUR)

`packaging/PKGBUILD` builds from the GitHub **release tarball** for tag
`v$pkgver`, and `packaging/.SRCINFO` mirrors it. The AUR git repo needs three
files committed together: `PKGBUILD`, `.SRCINFO` and `ward.install` (the
`install=` hook is not embedded in `.SRCINFO`).

Publishing / updating:

```
# after the v0.5.0 tag + GitHub release exist:
updpkgsums                               # fills the real sha256 (needs pacman-contrib)
makepkg --printsrcinfo > .SRCINFO        # regenerate from the PKGBUILD
makepkg -si                              # test-build in a clean chroot
git add PKGBUILD .SRCINFO ward.install && git commit && git push   # to the AUR remote
```

`sha256sums` ships as `SKIP` until the tag exists; replace it with the real
digest via `updpkgsums` before publishing. `arch` stays `x86_64 aarch64` (not
`any`) because `pam_ward.so` is compiled. `python-dlib` is an `optdepend`.

For a quick local build straight from a checkout (no tag needed), the
`package()` step is offline — point `source=` at a local tarball of the tree.

---

## Fedora / openSUSE / RHEL-likes (RPM, COPR)

`packaging/ward.spec` is the RPM (verified on Fedora by
`tests/containers/run.sh`). Offline local build:

```
git archive --prefix=ward-0.5.0/ -o ~/rpmbuild/SOURCES/ward-0.5.0.tar.gz HEAD
rpmbuild -ba packaging/ward.spec
```

### COPR — build from git (recommended, auto-rebuild on push)

In the COPR web UI (or `copr-cli`), add an **SCM** package:

```
copr-cli create ward --chroot fedora-rawhide-x86_64 --chroot fedora-41-x86_64
copr-cli add-package-scm ward --name ward \
    --clone-url https://github.com/0ctave/ward \
    --commit v0.5.0 --spec packaging/ward.spec --type git --method rpkg
copr-cli build-package ward --name ward
```

- Clone URL `https://github.com/0ctave/ward`, Spec `packaging/ward.spec`,
  Committish `v0.5.0` (or a branch), SRPM method **rpkg** (the default — the
  spec uses `%autosetup` and rpkg generates the `%{name}-%{version}.tar.gz`
  from the checkout, so no release-tarball upload is needed).
- Enable the webhook rebuild and add COPR's webhook URL to the GitHub repo so
  every push/tag rebuilds.
- `python3-dlib` is packaged only on openSUSE, so on Fedora chroots it stays an
  unsatisfied `Recommends` (not an error) — consistent with the no-hard-dlib rule.

### COPR — upload a prebuilt SRPM (one-off)

```
rpmbuild -bs packaging/ward.spec
copr-cli create ward --chroot fedora-rawhide-x86_64
copr-cli build ward ~/rpmbuild/SRPMS/ward-0.5.0-1.*.src.rpm
```

---

## Not a target: Flathub / Flatpak

Ward **cannot** ship as a sandboxed Flatpak, so Flathub is not a packaging
target. Ward is a system authentication component, not a sandboxed app:

- It installs a **PAM module** (`pam_ward.so`) into the host's security stack
  and is invoked by `login`/`sudo`/polkit — PAM runs host-side, outside any
  Flatpak sandbox, and a Flatpak cannot install into `/usr/lib/security` or
  edit the host PAM configuration.
- It runs a **system service** (`ward.service`) with raw access to the V4L2
  camera nodes and the UVC IR emitter (XU ioctls), and talks to the host
  keyring and `/etc/shadow` — exactly the host integration the Flatpak sandbox
  exists to prevent.

A tool whose whole job is to authenticate the host login belongs in the host's
own package manager. Use the `.deb`/RPM/AUR channels above.

---

## Release checklist for 0.5.0

1. Bump the version (see list above); `build-deb.sh` needs no edit.
2. `git tag v0.5.0 && git push --tags`; create the GitHub release and attach the
   `.deb` from `build-deb.sh` (and optionally an RPM).
3. **AUR:** drop `PKGBUILD` + `ward.install` into the AUR repo, run
   `updpkgsums && makepkg --printsrcinfo > .SRCINFO`, test `makepkg -si`, commit
   `PKGBUILD .SRCINFO ward.install`.
4. **COPR:** create the project and add the SCM package pointing at
   `packaging/ward.spec` @ `v0.5.0` (Route 1 above).
5. **PPA:** `debuild -S -sa` per Ubuntu series (ascending `~` suffixes) and
   `dput` each.
6. Verify `build-deb.sh` still produces a working `.deb` and
   `tests/containers/run.sh` still passes the RPM on Fedora.
