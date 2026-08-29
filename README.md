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
▌ faceup 0.2.0 · doctor
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
  - *anti-spoof CNN* on the RGB crop (MiniFASNet, ONNX via OpenCV DNN; needs the model
    file — not bundled yet, the voter is disabled until it's present)
  - *emitter-modulation shadow test* — metric implemented, enabled once the LED toggle is
    proven fast enough on the hardware (`shadow_test = "auto"`)
- **Stable camera identification.** Cameras are stored as USB ids (vendor:product:interface),
  not `/dev/videoN` numbers, which change when the module re-enumerates.
- **YuNet face detection.** OpenCV's detector (bundled, Apache-2.0): 4 ms per frame on a laptop
  CPU and robust to turned/tilted heads — dlib's HOG is frontal-only, its CNN 700 ms.
- **Fail open.** Every failure — dark camera, busy camera, spoof suspected, timeout —
  ends in the normal password prompt. One log line per attempt says why.

Full rationale and thresholds: [docs/DESIGN.md](docs/DESIGN.md).

## Commands

| Command | Root | What it does |
|---|---|---|
| `faceup doctor` | no | Diagnoses cameras (IR vs RGB via raw V4L2 ioctls), live lighting, emitter state, engine, enrolled models, PAM. Exit 1 if anything needs fixing. |
| `faceup setup` | yes | Guided: cameras → emitter → detector benchmark → anti-spoof model → enrollment → PAM. |
| `faceup enroll` | yes | Free-move enrollment in a camera window (`--guided` for the scripted text flow), with calibration, wrong-person check and a 3-attempt self-check. |
| `faceup auth -v` | yes | Runs one authentication and shows every step (what PAM calls without `-v`). |
| `faceup tune [section.key VALUE]` | yes | Show or change `/etc/faceup/config.toml` with validation. |
| `faceup pam enable\|disable\|status` | yes | The PAM line, with backup/restore; offers to retire `pam_howdy`. |
| `faceup calibrate` | yes | Rebuild the personal 3D face model from the last enrollment's landmark tracks. |
| `faceup models [--remove LABEL]` | yes | List or remove enrolled poses. |
| `faceup log` | no* | Recent attempts, one line each (*`adm` group). |
| `faceup bench` | no | Times the detectors on a live frame. |
| `faceup emitter check` | yes | Lights off → IR/RGB side by side → tells you whether the emitter fires. `install`/`configure` via linux-enable-ir-emitter. |

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
