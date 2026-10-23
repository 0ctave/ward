# Changelog

## Unreleased — 2026-10-22

Enrol once in normal light, unlock in any light; one coherent enrolment window; safer PAM; other distros.

### Security audit
- **PAM helper environment (critical):** `pam_faceup.so` now `clearenv()`s both root children (the faceup-auth helper and the systemd-creds unseal) and sets only PATH, LANG and PAM_*; through setuid sudo / the polkit agent the inherited environment was the unprivileged caller's (PYTHONHOME, HOME → user site-packages = root code before the password). Both launchers run `python3 -I` with the package path set inside, ignoring every PYTHON* variable and the user site (`faceup` no longer appends the caller's PYTHONPATH).
- **Fail open in the PAM module:** the unseal abort kills and reaps with a bound (a decrypt wedged in the TPM no longer holds back the password module); an unreaped or buffer-filling unseal is never read as a good secret; SIGCHLD is defaulted around the helper (pam_exec-style) and a give-up kills the whole group.
- **Resident engine:** a queued request waits ≤ 1 s for the cameras then answers `busy` (no phantom attempts for clients that left, no handler pile-up); an attempt far past its hard timeout makes the daemon exit for a systemd restart; writes time out; the user name is validated before the cameras. The client falls back to the in-process engine on any daemon error before the attempt starts.
- **Liveness:** a FAIL from cross-spectrum, shadow or the CNN is sticky — the attempt ends as `spoof:*` (counted by the lockout) instead of waiting for a frame that slips through; motion stays soft. The lockout reads a 1 MB time-filtered tail instead of 400 shared lines. Online learning needs a cross/shadow/CNN PASS (never motion alone) and `learn_max_dist` is clamped to gate − 0.10.
- **Privileges:** the app never runs a user-writable checkout as root through pkexec (installed launcher, or opt in with `FACEUP_DEV_PKEXEC=1`); the polkit action is `auth_admin` (not remembered — another process of the session could reuse a kept grant for any subcommand).
- **Robustness:** a late camera open after `close()` or a failed attempt start (open timeout, model load error) is released by a non-blocking cancel; the log readers survive invalid UTF-8; the People page ignores an unexpected `models --json` reply.

### Cleanup
- Removed dead code (unused pose/sfm helpers, howdy `set_many`/`pam_enabled`, `ambient.is_dark`,
  `appui.cue_angle`, unused constants and an import-time gamma LUT in gui).
- App: turning face unlock off from the Overview switch now confirms (the duplicate Advanced row
  is gone); "Forget learned models" acts on that person only (`models --forget-learned --identity`)
  and confirms; dialogs name people by display name; keyring shows only the action that applies;
  policy dropdowns use words ("Same as overall", "Password only"); attempt distances and votes moved
  to a tooltip; failure reasons show a tip instead of a code; one Lab entry; plain subtitles
  instead of CLI jargon; suffix buttons carry an accessible label and rows activate them; config
  is read off the GTK thread. "Register" is the user-facing word in the app and CLI messages.
- Size/speed: `urllib`/`tempfile` imported only by the emitter download; install.sh byte-compiles
  the installed package (no compile on the first PAM attempt) and builds `pam_faceup.so` with
  `-Os -s` + section GC (21 KB → 19 KB, `pam_sm_*` still exported).

### Faster start (LEDs on ~2 s sooner)
- The ambient-light read (≈2 s on a sleeping sensor hub) no longer blocks the camera open: IR opens
  at once, the light answer only decides RGB (no answer in 2.5 s → RGB opens, as before).
- IR opens first and alone (the module's driver serialises opens; a racing RGB delayed IR ~0.4 s),
  and the dlib models load only once IR streams (their load held the GIL and stalled the open).
  Log `ph=` gains `ir:` = ms until the IR stream is up.
- Recognition starts on IR as soon as it streams; RGB joins on a later frame once open (and a
  failed RGB open no longer fails the attempt — IR carries it).

### Resident engine (unlock ~1 s sooner)
- `faceup.service` (faceup/daemon.py) keeps Python, OpenCV, dlib and the models loaded and warmed
  (first YuNet inference done at start). `faceup-auth` is now a stdlib-only client
  (faceup/client.py, ~30 ms to start) that asks it over a root-only socket and relays feedback; with
  no daemon it runs the in-process engine as before. On by default from install.sh;
  `faceup daemon status|enable|disable`, a doctor item and a Settings switch ("Keep face unlock ready").

### Smoother unlock
- The native PAM module waits for the keyring decrypt at most 1 s after the face is accepted
  (was up to 5 s from the start): a slow or failing TPM unseal no longer stalls an unlock.
- doctor/app warn when the keyring password is TPM-sealed on an encrypted disk and offer to
  re-seal with the instant host key; every attempt now logs its caller (`via gdm-password`, …).

### Manager app (`faceup app`)
- Runs as the logged-in user; every change goes through `pkexec` (polkit `org.faceup.app`,
  remembered for a few minutes) — no GTK as root. The launcher no longer wraps the app in pkexec.
- Pages: Overview (on/off switch, reliability verdict, test unlock, things to fix), People (each
  identity with its cameras, conditions, last unlock; register, re-enrol, add/remove a condition,
  remove a person or your own models — destructive steps confirm first; names are validated),
  Reliability (last 50 attempts by light, service and person, failure reasons in words, next steps
  with a one-click action), Settings (policy and per-service dropdowns, behaviour switches,
  keyring), Advanced (setup steps, recogniser, lab, doctor, turn off everywhere).
- One window: registering, adding a condition and calibrating run INSIDE the manager — the
  privileged `faceup enroll|calibrate --stream-ui` sends its window state as JSON lines (frame as
  JPEG) and takes the buttons back on stdin (`display.StreamDisplay`); messages go to the console.
- `faceup models --json` (people on this login, for the app), `--remove-condition C --identity N`;
  commands run through pkexec resolve the real user from `PKEXEC_UID`. `faceup/manage.py` holds the
  app's data layer (tested without GTK).

### Enrolment window, second pass
- Headless test harness for the whole window (`tests/enroll_harness.py`): virtual clock, scripted
  user, fake camera/sampler — every pass, key, timeout and failure is a unit test.
- Honest outcomes: "Done" only when at least half the poses produced a model; a lost camera, a
  crashed worker or a thin sweep ends with "Nothing saved" and the reason, and nothing is written.
- Guidance: the arrow points from the current pose to the target (corrects overshoot and drift),
  "Skip pass" button, current-pose dot, "Look up" label, ring halo, glyphs kept out of the chip,
  "Come a little closer" / "Camera not responding" pills; the pass clock runs only while a usable
  face is in view; the glasses sweep waits for Start; up/down no longer accept a level head.
- Models: 6 frames per pose spread over the pass; no synthetic lighting variants (4× sample
  rate); a second face leaning in is dropped; worker crashes are survived once.
- The whitened acceptance path is **off by default** (`recognition.whitened = false`): measured
  never to fire (live 28–37 vs threshold ~15) and it rescued no night frame. Opt-in remains.

### Night unlock (root cause found)
- On a Windows-Hello module (RGB + IR on one USB device), **opening the RGB camera in a dark room
  kills the IR stream**: flat frames, LED off, and closing RGB does not revive it. Every night
  no-match in the log had both cameras open. Now the laptop's ambient light sensor picks IR only
  below 5 lux (`engine/ambient.py`), and a watchdog drops RGB and reopens IR if the IR stream
  goes flat while RGB is dark too. Measured: IR-only dark frames sit 0.31 from daylight-enrolled
  models — no separate night enrolment needed. Attempts log `ir-only (dark)`.

### Enrolment
- Frames count for a pose only within 12° of its target; the last third in time is held out for
  the gate; dark frames enrol nothing; each camera ranked by its own quality; a pose completes on
  usable samples (not camera frames), the light pass stays open 4 s, in-flight samples are kept.
- The window path asks nothing in the terminal; IR↔RGB alignment happens during the intro; the
  done card lists missed poses and closes itself; closing it keeps the enrolment; an empty sweep
  is never saved. Launched without a terminal or display, enrol fails clearly instead of
  answering its own prompts.
- **Glasses in the same window**: "I also wear glasses" on the done card runs a short sweep tagged
  `condition=glasses`; the calibrated gate is untouched.

### Engine
- Online learning measures novelty against enrolled models only (no chaining), needs ≥ 2 frames,
  caps learned models per camera; whitening refits move the threshold ±20 % and keep the
  session floor.
- Early reject on the median of the last frames (the old all-frames rule never fired); per-phase
  timings `ph=models,cams,face` in every log line.

### Fail-open and portability
- `pam_faceup.so` enforces its timeout with `poll()` and kills the helper's process group (a silent
  hung helper used to block the login); `faceup-auth` has a kernel `SIGALRM` backstop.
- PAM: `common-auth` or `system-auth`, symlink-preserving atomic writes, authselect-managed stacks
  refused with instructions. `install.sh`: distro-aware hints (apt/dnf/pacman/zypper via
  `faceup/distro.py`), uninstall checks every PAM file, SELinux relabel.
- libadwaita < 1.3 fallbacks, X11 under sudo (XAUTHORITY), any Wayland socket, YuNet → HOG
  fallback, `FACEUP_DLIB_DIR` and more model paths, 16-bit IR frames, libcrypt on minimal/musl.

## 0.3.0 — 2026-09-09

The "it just opens, and it's yours" release: one guided setup, honest thresholds, a second
liveness voter, keyring unlock, and a compatibility matrix so it runs on other machines.

### The app
- `faceup app`: one GTK 4 / libadwaita window — Status with a Fix button per check, Setup (seven steps), Faces (people and conditions), Attempts, Lab, Settings. A driver over the CLI: every button runs the matching command and shows its output; `pkexec` launcher entry; read-only without root.

### Setup and enrolment
- `faceup enroll` is ONE guided window that enrols **and** calibrates in a single sweep: a ring
  frames your face, an arrow points the way, the ring fills as you hold each pose. The gate is
  derived from held-out frames of the same sweep (real fresh-frame distances), so it is honest
  and no longer capped by transition frames. `--classic` keeps the old free-move map.
- Named **identities**: several people can be registered on one login (`enroll --identity "Alice"`),
  and **conditions** (glasses, headset, night) are added with a short sweep (`enroll --add glasses`).
- `faceup calibrate` re-derives the gate and the whitening threshold on demand; the whitening
  threshold then follows your real unlocks within bounds (rolling refresh).
- The IR emitter is found by measurement in any lighting (`emitter probe`), its switch speed
  benchmarked (`emitter toggle-bench`), and it is turned on before the cameras open.

### Unlock
- Native PAM module `pam_faceup.so` (built with plain gcc, no -dev headers): forwards the
  helper's messages to GDM/SDDM/KDE/polkit, and hands PAM a **sealed keyring credential** so
  GNOME Keyring / KWallet unlock after a face login (`faceup keyring enable`). Host-key sealing
  on encrypted disks (instant), TPM2 optional; the decrypt runs in parallel with recognition.
- Feedback on every surface: PAM info lines, a terminal spinner for sudo, transient notifications.
- Faster: cameras open in parallel with model loading, no probes on the auth path, early reject
  of strangers, online learning that learns (per-frame samples, sane novelty).
- Shadow-test liveness voter active once the emitter is benchmarked; motion voter thresholds are
  resolution-independent; honest `liveness` failure reason with per-camera light states.

### Engine
- Pluggable recogniser: dlib (default) or **ArcFace** via OpenCV DNN (`faceup recogniser install`);
  dlib is optional on the ArcFace + YuNet path.
- Replay harness: `lab --record-frames`, `auth --replay` — the whole pipeline in CI without hardware.

### Compatibility
- Python 3.10 (TOML fallback) to 3.13 (crypt via libcrypt); OpenCV 4.6+; profiles A (IR+RGB),
  B (IR only), C (RGB only) degrade honestly; other frame sizes; no TPM / no FDE / no gcc /
  no libadwaita all fall back with a clear message. `tests/test_compat.py` is the matrix.

### Security policy
- `faceup tune policy convenience|balanced|strict` presets; spoof lockout after repeated
  liveness FAILs (password always available); per-service overrides.

## 0.2.0 — 2026-08-31
- Own engine replacing Howdy: dual-spectrum recognition, multi-frame fusion, liveness voters,
  free-move enrolment with a personal 3D face model, whitening, the lab bench.
