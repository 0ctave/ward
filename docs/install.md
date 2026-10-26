# Installing Ward

[← back to the README](../README.md)

Ward installs with your **distro's own Python** — the one that already has OpenCV and dlib. It never
uses pip or a virtualenv (a venv can't see the system OpenCV/dlib). The whole thing is one
`install.sh`, and `ward doctor` will tell you exactly what, if anything, is missing.

- [Prerequisites](#prerequisites)
- [1. Install dependencies](#1-install-dependencies)
- [2. Install Ward](#2-install-ward)
- [3. Set it up](#3-set-it-up)
- [Distro packages](#distro-packages)
- [Uninstall](#uninstall)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- **A Linux laptop or a webcam.** Ward is at its best with a **Windows Hello infrared camera** (most
  2019-and-later business/premium laptops) — infrared is what lets it work in the dark and reject
  phone-screen photos. A plain colour webcam also works (recognition + the anti-spoof CNN), just
  without infrared liveness and night unlock.
- **Python 3.10–3.13 with OpenCV**, from your distro. No pip, no virtualenv.

## 1. Install dependencies

Ward needs **OpenCV**, the **GTK 4 / libadwaita** bindings (for the app; it falls back to text
without them), **gcc** (to build the small PAM module), and a **face recogniser** — either dlib (the
default) or Ward's downloaded ArcFace model (no dlib needed).

| Distro | Command |
|---|---|
| Debian / Ubuntu / Mint / Pop!_OS / Zorin | `sudo apt install python3-opencv python3-gi python3-gi-cairo gir1.2-adw-1 gcc libc6-dev` |
| Fedora / RHEL | `sudo dnf install python3-opencv python3-gobject libadwaita gcc` |
| Arch / Manjaro | `sudo pacman -S python-opencv python-gobject python-cairo libadwaita gcc` |
| openSUSE Tumbleweed | `sudo zypper install python3-opencv python3-gobject python3-gobject-cairo typelib-1_0-Adw-1 gcc python3-dlib` |

### Pick a recogniser

- **dlib** (default). Packaged as `python3-dlib` **only on openSUSE Tumbleweed** (included above). On
  **Ubuntu/Debian** it comes from Howdy's PPA, on **Arch** from the AUR (`python-dlib`); elsewhere
  build it from source. You also need the model files in `/usr/share/dlib` (package `dlib-models`, or
  download `shape_predictor_5_face_landmarks.dat`, `shape_predictor_68_face_landmarks.dat` and
  `dlib_face_recognition_resnet_model_v1.dat` from [dlib.net/files](http://dlib.net/files/)).
- **ArcFace** (no dlib). The simplest path when dlib isn't packaged for you: after installing Ward,
  run `sudo ward recogniser install && sudo ward recogniser use arcface`. Ward then uses its bundled
  YuNet detector and a downloaded ArcFace model, both through OpenCV — no dlib at all.

> [!TIP]
> Not sure what you have? Install Ward and run `ward doctor` — it checks every dependency and prints
> the exact command for anything missing.

## 2. Install Ward

```bash
git clone https://github.com/0ctave/ward && cd ward
sudo ./install.sh          # → /usr/local/lib/ward + /usr/local/bin/{ward,ward-auth}
ward doctor                # run this first — it diagnoses cameras, emitter, recogniser, PAM
```

`install.sh` copies the package, builds the native PAM module (if `gcc` is present), installs the app
launcher, and enables the resident engine. It **never edits your PAM config** — setup does that, and
asks first.

## 3. Set it up

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
coexists.

## Distro packages

Built offline from this repo, with no network. The packages install the same files under `/usr`
(`/usr/bin/ward{,-auth}`, `/usr/lib/ward`, `/usr/lib/security/pam_ward.so`, the systemd unit); they
compile `pam_ward.so` at build time, so the target needs no `gcc`.

```bash
packaging/build-deb.sh                     # Debian/Ubuntu → dist/ward_*.deb
sudo apt install ./dist/ward_*.deb

git archive --prefix=ward-0.4.0/ -o ~/rpmbuild/SOURCES/ward-0.4.0.tar.gz HEAD
rpmbuild -ba packaging/ward.spec           # Fedora/openSUSE
(cd packaging && makepkg -si)              # Arch
```

Packages **never edit PAM** — run `sudo ward setup` (or `sudo ward pam enable`) afterwards. Removal
is refused while `/etc/pam.d` still references Ward (run `sudo ward pam disable` first; pacman can't
refuse, so it warns). Purging the `.deb` deletes `/etc/ward` (models, config, keyring credential).
Pick **one** install method — a from-source unit in `/etc/systemd/system` overrides the packaged one.

## Uninstall

```bash
sudo ward pam disable
sudo ./install.sh --uninstall      # keeps /etc/ward (your models/config); rm -rf it to remove those too
```

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
