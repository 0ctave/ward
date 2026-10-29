# Installing Ward

[← back to the README](../README.md)

Ward runs on your **distro's own Python** — the one that already has OpenCV (and, on some distros,
dlib). It never uses pip or a virtualenv (a venv can't see the system OpenCV/dlib). Install it from
your distro's package repository (below) or straight from source; either way, `ward doctor` tells you
exactly what, if anything, is missing.

- [Prerequisites](#prerequisites)
- [Install from a package repository](#install-from-a-package-repository)
- [Install from source](#install-from-source)
- [Pick a recogniser](#pick-a-recogniser)
- [Set it up](#set-it-up)
- [Building the packages yourself](#building-the-packages-yourself)
- [Uninstall](#uninstall)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- **A Linux laptop or a webcam.** Ward is at its best with a **Windows Hello infrared camera** (most
  2019-and-later business/premium laptops) — infrared is what lets it work in the dark and reject
  phone-screen photos. A plain colour webcam also works (recognition + the anti-spoof CNN), just
  without infrared liveness and night unlock.
- **Python 3.10–3.13 with OpenCV**, from your distro. No pip, no virtualenv.

## Install from a package repository

The quickest path on a supported distro: the package pulls in its dependencies, and your normal
package manager keeps Ward up to date.

**Ubuntu 24.04 "Noble"** (and derivatives — Mint 22, Pop!\_OS 24.04, Zorin 18):

```bash
sudo add-apt-repository ppa:0ctave/ward
sudo apt update
sudo apt install ward
```

**Fedora** (Copr):

```bash
sudo dnf copr enable 0ctave/ward
sudo dnf install ward
```

**Any Debian/Ubuntu — the `.deb` directly** (grab the latest from
[Releases](https://github.com/0ctave/ward/releases/latest)):

```bash
sudo apt install ./ward_*.deb
```

**Arch (AUR):** *coming soon* — until the AUR package is published, use the
[from-source install](#install-from-source); it's one command.

> [!NOTE]
> **Recogniser.** Fedora's package pulls in **dlib** automatically. On **Ubuntu/Debian**, dlib isn't
> in the archive, so after installing run
> `sudo ward recogniser install && sudo ward recogniser use arcface` to switch to the bundled ArcFace
> model (no dlib needed), or install dlib yourself. `ward doctor` tells you which you have. See
> [Pick a recogniser](#pick-a-recogniser).

These packages **never edit PAM** — finish with the app's wizard or `sudo ward setup`, both under
[Set it up](#set-it-up).

## Install from source

Works on **any** distro. First the dependencies — **OpenCV**, the **GTK 4 / libadwaita** bindings
(for the app; it falls back to text without them), **gcc** (to build the small PAM module), and a
**face recogniser** (dlib, or Ward's downloaded ArcFace model):

| Distro | Command |
|---|---|
| Debian / Ubuntu / Mint / Pop!_OS / Zorin | `sudo apt install python3-opencv python3-gi python3-gi-cairo gir1.2-adw-1 gcc libc6-dev` |
| Fedora / RHEL | `sudo dnf install python3-opencv python3-gobject libadwaita gcc` |
| Arch / Manjaro | `sudo pacman -S python-opencv python-gobject python-cairo libadwaita gcc` |
| openSUSE Tumbleweed | `sudo zypper install python3-opencv python3-gobject python3-gobject-cairo typelib-1_0-Adw-1 gcc python3-dlib` |

Then install Ward itself:

```bash
git clone https://github.com/0ctave/ward && cd ward
sudo ./install.sh          # → /usr/local/lib/ward + /usr/local/bin/{ward,ward-auth}
ward doctor                # run this first — it diagnoses cameras, emitter, recogniser, PAM
```

`install.sh` copies the package, builds the native PAM module (if `gcc` is present), installs the app
launcher, and enables the resident engine. It **never edits your PAM config** — setup does that, and
asks first.

## Pick a recogniser

- **dlib** (default). Packaged as `python3-dlib` on **Fedora** and **openSUSE Tumbleweed** (the Fedora
  package and the Tumbleweed deps above pull it in). On **Ubuntu/Debian** it comes from Howdy's PPA,
  on **Arch** from the AUR (`python-dlib`); elsewhere build it from source. You also need the model
  files in `/usr/share/dlib` (package `dlib-models`, or download
  `shape_predictor_5_face_landmarks.dat`, `shape_predictor_68_face_landmarks.dat` and
  `dlib_face_recognition_resnet_model_v1.dat` from [dlib.net/files](http://dlib.net/files/)).
- **ArcFace** (no dlib). The simplest path when dlib isn't packaged for you (e.g. Ubuntu/Debian):
  `sudo ward recogniser install && sudo ward recogniser use arcface`. Ward then uses its bundled
  YuNet detector and a downloaded ArcFace model, both through OpenCV — no dlib at all.

> [!TIP]
> Not sure what you have? Run `ward doctor` — it checks every dependency and prints the exact command
> for anything missing.

## Set it up

**Desktop — the app (recommended):** open **Ward** from your applications (or just run `ward`). On
first run it leads with **Set up Ward → Start setup**, a guided wizard that does everything in one
window: cameras → infrared emitter → register your face → keyring → turn on. See the
[Usage guide](usage.md#the-app).

**Terminal / headless:**

```bash
sudo ward setup            # the same guided chain, in the terminal
```

On a machine with no display (SSH), the registration step automatically falls back to a **text pose
flow** ("look straight… look down… turn left…"). The camera has to be on *that* machine. See
[Usage → Headless](usage.md#headless--ssh).

Either way it finishes by testing a real unlock (`sudo -k && sudo true`). Howdy is not required and
coexists — if it's installed, see **[Migrating from Howdy](migrating-from-howdy.md)**, which Ward can
retire for you during setup.

## Building the packages yourself

Everything above installs a prebuilt package; this is for maintainers who want to **build** one from a
checkout. Built offline from this repo, with no network. The packages install the same files under
`/usr` (`/usr/bin/ward{,-auth}`, `/usr/lib/ward`, `/usr/lib/security/pam_ward.so`, the systemd unit);
they compile `pam_ward.so` at build time, so the target needs no `gcc`.

```bash
packaging/build-deb.sh                     # Debian/Ubuntu → dist/ward_*.deb
sudo apt install ./dist/ward_*.deb

ver=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' ward/__init__.py)
git archive --prefix=ward-$ver/ -o ~/rpmbuild/SOURCES/ward-$ver.tar.gz HEAD
rpmbuild -ba packaging/ward.spec           # Fedora/openSUSE
(cd packaging && makepkg -si)              # Arch
```

Packages **never edit PAM** — run `sudo ward setup` (or `sudo ward pam enable`) afterwards. Removal
is refused while `/etc/pam.d` still references Ward (run `sudo ward pam disable` first; pacman can't
refuse, so it warns). Purging the `.deb` deletes `/etc/ward` (models, config, keyring credential).
Pick **one** install method — a from-source unit in `/etc/systemd/system` overrides the packaged one.

See [`packaging/README.md`](../packaging/README.md) for publishing to the AUR, Fedora Copr and the
Debian/Ubuntu PPA, and [`RELEASING.md`](../RELEASING.md) for cutting a release.

## Uninstall

```bash
sudo ward pam disable
sudo ./install.sh --uninstall      # keeps /etc/ward (your models/config); rm -rf it to remove those too
```

Installed from a package repository instead? Remove it with your package manager
(`sudo apt remove ward`, `sudo dnf remove ward`, `sudo pacman -R ward`) — run `sudo ward pam disable`
first.

## Troubleshooting

`ward doctor` is the first stop for anything — it diagnoses every part and prints the fix command.
The usual ones:

| Symptom (or what `doctor` says) | Fix |
|---|---|
| `dlib missing`, or recognition errors on first use | install dlib (above), or `sudo ward recogniser use arcface` |
| Works in daylight, fails in the dark | the IR emitter isn't firing: `sudo ward emitter probe` |
| `no IR camera` but the laptop has one | it may stream as YUYV — set `cameras.ir` and `cameras.force_roles true` (doctor shows the id) |
| Face unlock never triggers | `ward pam status`; turn it on with `sudo ward pam enable` (or the app's switch). Then `sudo -k; sudo whoami` and check `ward log -n 2` — a fresh line means Ward ran. |
| Keyring stays locked after a face login | `sudo ward keyring enable` (and `sudo ward keyring repair` if an old config disabled it) |
| The app windows are plain text | GTK 4 + libadwaita bindings are missing — install them (above) |
| Unlock feels slow | `sudo ward daemon enable` (keeps the models loaded), and check you're not on the power-saver CPU profile |

Still stuck? Open a [camera / hardware issue](https://github.com/0ctave/ward/issues/new/choose) with
your full `ward doctor` output.
