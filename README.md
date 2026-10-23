# faceup

**Make face unlock on Linux actually work.**

[Howdy](https://github.com/boltgolt/howdy) gives Linux Windows-Hello-style face
authentication for login, `sudo` and polkit. Out of the box it is unreliable for
reasons that have nothing to do with face recognition: it picks the wrong camera,
the infrared emitter never switches on, one enrolled pose means one working head
angle, and the defaults are tuned for a 2018 webcam. `faceup` fixes the setup, not
the algorithm — and it explains what it does so you can go further yourself.

```
$ faceup doctor
▌ faceup 0.3.0 · doctor
Cameras
  · /dev/video0  RGB  1920x1080  [MJPG/YUYV]  Integrated Camera
  ✔ /dev/video2  IR (greyscale)  640x360  [GREY]  Integrated Camera
  ✔ config: ir = /dev/video2, rgb = /dev/video0
  ! ir lighting now: mean 40/255 → dim
    → dark IR = the emitter is probably not firing
IR emitter (linux-enable-ir-emitter)
  ✘ installed but NO device configured — the tool has nothing to switch on
    → sudo faceup emitter configure
Engine
  ✔ `python3` on PATH (/usr/bin/python3) can import dlib + cv2
  ✔ ir: 8 model(s) for octave: straight, keyboard, tilt-left, …
PAM
  ✔ faceup-auth is enabled in /etc/pam.d/common-auth
```

## What v0.2 does

faceup no longer wraps Howdy: it ships its **own authentication engine** (`faceup-auth`,
called by PAM through the stock `pam_exec.so`) built on the same proven pieces — dlib's
detector and ResNet face embedding, OpenCV, V4L2 — with what Howdy lacks:

- **IR primary, RGB secondary.** Both cameras are read in lock-step (20 pairs/s on
  the reference laptop).
- **Multi-frame fusion.** Embeddings of the best frames are averaged before matching,
  which removes the per-frame noise behind random rejects.
- **Free-move enrollment.** A native Wayland (GTK 4) camera window tracks your head angle (yaw/pitch from facial
  landmarks) and fills a 5×3 pose grid as you move — no scripted poses. Adjacent cells are
  interpolated at match time so the angles in between are covered too. Enrollment ends with a
  structure-from-motion solve that builds *your* 3D face model from the recorded landmarks, so
  head pose is then measured against your own geometry, not a generic head.
- **Liveness voters** with PASS / FAIL / ABSTAIN semantics — two PASSes and zero FAILs
  needed, and abstaining never counts against you:
  - *cross-spectrum consistency* — the face must appear in both cameras at the same place
    (screens emit no IR; IR-tuned prints look wrong in colour)
  - *non-rigid motion* — a head moves like a 3D object, a photo or screen like a plane
  - *anti-spoof CNN* on the RGB crop (MiniFASNetV2, ONNX via OpenCV DNN, bundled — Apache-2.0,
    1.7 MB; passes real faces, fails only a confident repeated spoof, otherwise abstains)
  - *emitter-modulation shadow test* — metric implemented, enabled once the LED toggle is
    proven fast enough on the hardware (`shadow_test = "auto"`)
- **Stable camera identification.** Cameras are stored as USB/PCI ids (vendor:product:interface,
  serial or port when needed), not `/dev/videoN` numbers, which change when a module
  re-enumerates. IR is recognised by stream format or device name; `doctor` explains cameras
  it can't use (permissions, libcamera-only sensors).
- **Brightness management.** The face region is metered (not the frame); cameras with manual
  exposure are steered into range, cameras without are normalised in software (CLAHE/gamma);
  the engine picks whichever spectrum actually sees the face in the current light.
- **Hardware profile.** `doctor` reports CPU/AVX2, OpenCL, camera exposure controls, power
  profile; the engine adapts thread count and can run DNN models on OpenCL when present.
- **YuNet face detection.** OpenCV's detector (bundled, Apache-2.0): 4 ms per frame on a laptop
  CPU and robust to turned/tilted heads — dlib's HOG is frontal-only, its CNN 700 ms.
- **On-screen feedback.** The lock screen and login screen (GDM, SDDM, KDE), the polkit
  dialogs and `sudo` show what the camera is doing: "Looking for your face…", a hint when
  only liveness is holding ("move your head a little"), and the result. In a terminal the
  status line animates in place; a desktop notification after an unlock says which prompt
  it was and how close the match was (`behaviour.notify = accept | all | off`).
- **Fail open.** Every failure — dark camera, busy camera, spoof suspected, timeout —
  ends in the normal password prompt. One log line per attempt says why.

Full rationale and thresholds: [docs/DESIGN.md](docs/DESIGN.md).

## Two ways in: the app, or the command line

**The app** (`faceup` in your launcher, or `sudo faceup app`): one window with six pages —
**Status** (every health check with a Fix button), **Setup** (the seven steps), **Faces**
(you, other people, glasses/headset/night conditions), **Attempts** (the log), **Lab** (the
diagnostic bench) and **Settings** (policy, behaviour, recogniser). The app is a driver over
the command line: every button runs the matching `faceup …` command and streams its output
into the console pane, so the two can never disagree, and everything below works without it.

## The chain, end to end

```bash
sudo ./install.sh          # distro Python only; builds the native PAM module when gcc exists
sudo faceup setup          # cameras → emitter (measured) → recogniser → enrol window → keyring → PAM → policy
sudo -k && sudo true       # unlocks by face; faceup log shows every attempt
```

- **Enrol** is one guided window: a ring frames your face, an arrow points the way, the ring
  fills as you hold each pose — and the same sweep sets the recognition gate from held-out
  frames of your own face (honest thresholds, no constants).
- **Several people, one login**: `sudo faceup enroll --identity "Alice"` registers another
  face; any registered face unlocks, and the log says who (`who=Alice`).
- **Glasses, headset, night**: `sudo faceup enroll --add glasses` is a short four-pose sweep
  that adds the condition to your models without touching the rest. `faceup models` lists
  identities and conditions; `--remove-identity` / `--remove-condition` take them out.
- **Keyring**: `sudo faceup keyring enable` stores your login password sealed (host key on an
  encrypted disk, TPM2 optional) so a face login unlocks GNOME Keyring / KWallet.
- **Policy**: `sudo faceup tune policy convenience|balanced|strict`; `balanced` (default)
  makes the colour camera agree whenever it sees you well; `strict` tightens the gate to
  0.35, requires a shadow-test or anti-spoof PASS and turns learning off. Per service:
  `security.policy_sudo`, `policy_polkit`, `policy_login`, `policy_unlock` (`off` disables
  face unlock there). Three liveness failures in 15 minutes pause face unlock for 15 minutes;
  the password always works.

### Runs on

Any Linux with `python3-opencv` (≥ 4.6) and `python3-dlib` (`dlib-models`) — Debian, Ubuntu,
Fedora, Arch (AUR). Python 3.10 to 3.13. Profiles: **A** IR + RGB (Windows-Hello modules),
**B** IR only, **C** RGB only (recognition alone unless the anti-spoof model is present; setup
asks). Any frame size. Without a TPM the keyring credential uses the host key on an encrypted
disk; without gcc the PAM line falls back to `pam_exec` (no keyring token); without
GTK 4 + libadwaita the windows fall back to text flows. `tests/test_compat.py` runs the whole
pipeline on recorded frames across these profiles, in CI, without a camera.

## Commands

| Command | Root | What it does |
|---|---|---|
| `faceup app` | sudo* | The GTK application (*status and log without root). |
| `faceup doctor` | no | Diagnoses cameras (IR vs RGB via raw V4L2 ioctls), live lighting, emitter state, engine, enrolled models, PAM. Exit 1 if anything needs fixing. |
| `faceup setup` | yes | Guided: cameras → emitter → detector benchmark → anti-spoof model → enrollment → PAM. |
| `faceup enroll [--identity NAME] [--add CONDITION]` | yes | One guided window that enrols AND sets the gate in a single sweep. `--identity` registers another person on this login; `--add glasses` (headset, night…) is a short sweep on top. `--classic` old free-move map; `--guided` text flow. |
| `faceup auth -v` | yes | Runs one authentication and shows every step (what PAM calls without `-v`). |
| `faceup tune [section.key VALUE \| policy LEVEL]` | yes | Show or change the config with validation; `tune policy convenience\|balanced\|strict` sets the security preset. |
| `faceup pam enable\|disable\|status` | yes | The PAM line, with backup/restore; offers to retire `pam_howdy`. |
| `faceup lab` | yes | Live evaluation window: raw and whitened distances, liveness votes and the decision under three unlock policies side by side; record subjects (you, a friend, a phone photo) to CSV and get accept rates per subject and policy. `--text` for the terminal. |
| `faceup recogniser status\|install [mbf\|r50]\|use dlib\|arcface` | yes* | The embedding network: dlib (distro) or ArcFace via OpenCV DNN, downloaded pinned (*status needs no root). |
| `faceup calibrate [session\|refresh\|pose]` | yes | `session` (default, also run at the end of enrollment): eight guided passes measure your real distances and set the gate and the whitening threshold from them; `refresh` re-derives from recent unlocks; `pose` rebuilds the personal 3D face model. |
| `faceup models [--remove LABEL \| --remove-identity NAME \| --remove-condition C \| --forget-learned]` | yes | List identities, conditions and poses; remove any of them. |
| `faceup log [--summary\|--follow\|--json]` | no* | Recent attempts, one coloured line each; `--summary` = accept rate, median/p90 time, reasons, per service (*`adm` group). |
| `faceup bench` | no | Times the detectors on a live frame. |
| `faceup emitter probe\|toggle-bench\|check` | yes | `probe` finds the IR LED control by measurement (any lighting) and `toggle-bench` measures its switch to enable the shadow-test voter; `check` shows IR/RGB side by side. `install`/`configure` via linux-enable-ir-emitter. |
| `faceup keyring status\|enable\|disable\|repair` | yes* | Sealed login credential so a face login unlocks GNOME Keyring / KWallet (native PAM module); `repair` restores keyring PAM lines an earlier experiment disabled (*status needs no root). |

## Install

`faceup` deliberately uses the **distro Python** — the one that has `python3-dlib`
and `python3-opencv`. A virtualenv would not see them.

```bash
git clone https://github.com/<you>/faceup && cd faceup
sudo ./install.sh          # /usr/local/lib/faceup + /usr/local/bin/{faceup,faceup-auth}
faceup doctor
sudo faceup setup          # ends with: sudo -k && sudo true  → unlocks by face
```

Requirements: Linux, `python3-opencv`, `python3-dlib`, `dlib-models` (Ubuntu 24.04:
all in the archive). Howdy is no longer required, but coexists.
Uninstall: `sudo faceup pam disable && sudo ./install.sh --uninstall`.

### Distro packages (built offline from this repo)

`install.sh` stays the from-source path (`/usr/local`). The packages install the same files under
`/usr` (`/usr/bin/faceup{,-auth}`, `/usr/lib/faceup`, `/usr/lib/security/pam_faceup.so`, the unit in
`/usr/lib/systemd/system`); both share `packaging/stage.sh`. `pam_faceup.so` is compiled at package
build time, so the target needs no gcc.

```bash
packaging/build-deb.sh                     # Debian/Ubuntu → dist/faceup_0.3.0-1_<arch>.deb (dpkg-deb only)
sudo apt install ./dist/faceup_*.deb
git archive --prefix=faceup-0.3.0/ -o ~/rpmbuild/SOURCES/faceup-0.3.0.tar.gz HEAD
rpmbuild -ba packaging/faceup.spec         # Fedora/openSUSE
(cd packaging && makepkg -si)              # Arch
```

Installing enables the resident engine on first install (like `install.sh`) and **never edits
PAM**: run `sudo faceup setup` (or `sudo faceup pam enable`) afterwards. Removing the .deb/.rpm is
refused while `/etc/pam.d` still references faceup (`sudo faceup pam disable` first); pacman cannot
refuse, so it warns. Purging the .deb deletes `/etc/faceup` (models, config, keyring credential).
Pick one install method: a from-source unit in `/etc/systemd/system` overrides the packaged one.

Tests (no camera needed): `python3 -m unittest discover tests`.

---

## The guide: why face unlock fails on Linux, and how to make it reliable

Everything below is what `faceup` automates. Read it if you want to understand
or push further.

### 1. Use the infrared camera, not the RGB one

Most laptops with Windows Hello have **two** cameras behind the same USB device:
a colour one (`/dev/video0`, MJPG/YUYV, 720p–1080p) and an infrared one
(`/dev/video2`, 8-bit `GREY`, typically 640x360 or 640x480). They often carry
the *same* name ("Integrated RGB Camera"), so the only reliable tell is the pixel
format. `faceup doctor` lists them.

IR is the right choice because:
- **Lighting stops mattering.** The emitter floods your face with invisible
  light; the sensor sees the same picture at noon and at midnight.
- **Screens are invisible to it.** A photo on a phone barely registers in IR, so
  the cheapest spoof against RGB face unlock doesn't work.
- Howdy shipped `device_path = none` on some distros: nothing works until it's set.

With an IR camera set `dark_threshold` to ~90: IR frames are naturally dim and
the default rejects them as "too dark".

### 2. Make the emitter actually fire

This is the #1 cause of "works sometimes". On Windows the driver turns the IR
LEDs on; on Linux nothing does, unless the camera firmware does it by itself.
Result: in a bright room the IR sensor sees you by ambient light and Howdy works;
in a dim room it sees nothing and falls back to the password.

Test: lights off, `sudo howdy test`. If your face isn't clearly visible, the
emitter is off. Fix: `linux-enable-ir-emitter`, which brute-forces the camera's
vendor-specific UVC controls until the LEDs flash, then re-applies the setting
at **boot and after suspend** via a systemd unit (USB cameras forget it on
resume). Option order matters on v6: `linux-enable-ir-emitter --device
/dev/video2 --width 640 --height 360 configure`. Run it in a bright room, close
to the camera, and answer Y when the picture flashes.

### 3. Enroll many poses, not one face

Howdy's recogniser is dlib's ResNet face embedding (2017). It's decent, but it's
**pose-sensitive**: the embedding of you looking at the keyboard is far from the
embedding of you looking at the camera. Howdy compares a live frame against
*every* stored model and accepts the best match, so the fix is simple: **one
model per pose you actually use**. Windows Hello does exactly this during its
"keep moving your head" enrollment.

The poses that matter at a laptop, in order:
1. straight at the camera
2. **looking down at the keyboard** — where your face is when a `sudo` prompt appears
3. head tilted left / right (~15°)
4. head turned left / right (~20°)
5. leaning back, leaning in (distance changes apparent face size)
6. optional: with/without glasses, with a headset, lights off

Re-enroll from scratch after switching from RGB to IR — models captured on a
colour camera match poorly on IR frames.

### 3b. Let the machine set the gate

Every camera, room and face gives different distances, so a fixed threshold is wrong
somewhere. faceup measures instead: at the end of enrollment (or with `sudo faceup calibrate`)
a window guides eight short passes with head-movement cues (a dot to steer into a target,
rings to lean in or back, a tilting head), each pass verified from the pose estimate, and the
gate and whitening threshold are derived from your real distances with a hard ceiling. From then on every
unlock feeds a rolling window, and the whitening threshold follows your actual conditions
within bounds — a first night-time unlock goes through the whitened path or learning, and
the store absorbs it. `faceup doctor` shows when and from how many measurements the
thresholds were set, and says when your recent unlocks disagree with them.

### 4. Tune tolerance to the number of poses

`certainty` is the maximum embedding distance accepted (Howdy multiplies by 10:
3.5 ≈ 0.35). Default 3.5 is strict. With 6–10 enrolled poses, 4.0–4.5 is a good
balance: your own off-angle frames get accepted while strangers stay far above
it. Above 5 you're trading away real security.

`timeout` is how long Howdy keeps trying before the password prompt. 4 s default;
5–6 s gives it more frames to catch a clean one without a painful wait when it
genuinely can't see you.

### 5. Detector: HOG vs CNN

Two stages happen on every attempt: *detect* a face in the frame, then *recognise*
it. Howdy's default detector is dlib's HOG, which is fast (~50 ms) but **frontal
only** — a tilted or turned head is simply not found, so recognition never runs.
Howdy also ships dlib's CNN detector (`use_cnn = true`), which finds faces at
angles, but the distro dlib is built without AVX so it costs 0.4–1.7 s per frame
depending on `max_height`. `faceup bench` measures both on your CPU and
recommends the largest `max_height` that keeps an attempt under ~0.9 s. Try it
after multi-pose enrollment, not instead of it.

### 6. Things that silently break Howdy

- **Python shadowing.** `pam_howdy.so` executes plain `python3` from the caller's
  PATH. A global pyenv/mise/conda Python without dlib turns every unlock into
  `ModuleNotFoundError: No module named 'dlib'`. `faceup doctor` checks this.
- **PAM not enabled.** The package installs the module; `pam-auth-update` must
  reference it.
- **Keyring after face login.** GNOME's keyring is encrypted with your *password*;
  a face login can't unlock it, so expect one keyring prompt. Not a bug.
- **Broken upstream PPA on Ubuntu 24.04.** Use `ppa:ubuntuhandbook1/howdy`
  (Slimbook's maintained fork) until upstream Howdy 3 lands.

### Windows Hello parity

Windows Hello on the same camera does three things faceup did not: it recognises with a
network suited to infrared faces, it lights the face with the emitter and checks the
reflection as liveness, and it is specified against a false-accept rate of 1 in 100 000.
Where faceup stands:

| Hello has | faceup | Status |
|---|---|---|
| IR-capable recogniser with tight identity clusters | ArcFace (insightface, run by the distro OpenCV): `sudo faceup recogniser install && sudo faceup recogniser use arcface && sudo faceup enroll --fresh` | opt-in; the impostor calibration needs a second person in `faceup lab` |
| FAR 1 : 100 000 | measurable in the lab, not yet claimed | roadmap §2 |
| Both cameras must agree | `recognition.require_rgb_match = true` | available now |
| Keyring unlocked after a face login | native `pam_faceup.so` + TPM2-sealed credential (`faceup keyring enable`) | available now |
| Emitter-lit liveness | `faceup emitter probe` finds the LED control by measurement, `toggle-bench` activates the shadow voter | available now (decides in dim light) |

Identical twins defeat every 2D face system, Hello included. A strong look-alike is kept out
by the gate: keep `certainty` at 0.40, turn on `require_rgb_match`, and measure the sibling in
the lab before trusting the device in a shared household.

### Keyring

GNOME Keyring and KWallet unlock the login keyring with the password PAM received, and a
face accept has none — so without help a face **login** leaves the keyring locked until first
use (screen unlock, `sudo` and polkit happen inside a running session and are unaffected).
faceup closes the gap with a native PAM module and a sealed credential:

```bash
sudo faceup keyring enable      # asks the login password once, verifies it, seals it with the TPM2
sudo faceup pam enable          # switches the PAM line to pam_faceup.so (built by install.sh)
```

After a face accept the module unseals the credential (TPM2-bound, or host key on an encrypted
disk) and hands it to PAM exactly as if you had typed it; the keyring opens. `faceup keyring
status` shows the state; `disable` deletes the credential. If you prefer to keep the first
login on the password instead: `sudo faceup tune behaviour.skip_login true`.

### Honest limits

Even a well-tuned Howdy on IR is a **2D** system: no depth sensor, no liveness
detection. It defeats phone-screen photos and darkness; it does not defeat a
high-quality printed IR photograph. Treat it as strong convenience with a
password fallback — the same class as Windows Hello on the same hardware, not
the same class as Face ID. If you need something deterministic, a FIDO2 key
with `pam-u2f` is the reliable open-source option.

## Contributing

Issues and PRs welcome — especially camera quirks from other laptops (send the
`faceup doctor` output). Ideas on the list: a GTK4/libadwaita front-end, a
`faceup snapshot` command that saves failed frames for tuning, RGB liveness
heuristics, and packaging (deb / AUR / Flatpak for the GUI).

MIT © 2026
