# Changelog

## Unreleased — 2026-10-22

Enrol once in normal light, unlock in any light; one coherent enrolment window; safer PAM; other distros.

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
