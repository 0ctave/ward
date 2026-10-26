# Ward — Roadmap

*Written 2026-09-07 against v0.2.0 + session feedback. Companion to `DESIGN.md` (what exists and why) and `README.md` (the guide). This document says what comes next, in what order, and why.*

## 0. Where we stand

Measured on the reference laptop (Ryzen 5 6600HS, Windows-Hello IR + RGB module, GNOME 46 Wayland, Ubuntu 24.04):

| Fact | Value | Source |
|---|---|---|
| Time to accept, first frame already matching | 1.6 – 2.3 s | `/var/log/ward.log`, 15 attempts |
| Fixed overhead before the first frame | ≈ 1.5 s | cv2 import 390 ms · hw probes 83 ms · dlib models 320 ms · IR open 517 ms · RGB open 210 ms · first frames 200–280 ms |
| Design budget for the same path | 250 – 450 ms | `DESIGN.md` §12 |
| Whitened distance vs threshold, live | 11 – 37 vs ≈ 5.7 | never accepts; raw gate loosened to 0.517 to compensate |
| Liveness votes on a still head | cross PASS, motion ABSTAIN | one real voter decides |
| IR emitter | "configured", IR frame still dim (33/255) | `ward doctor` |
| Session feedback | terminal spinner, PAM info lines, notification | added 2026-09-05 |
| Tests | 56, no camera needed | `python3 -m unittest discover -s tests` |

Everything below is ordered by *user-visible payoff per unit of risk*. Golden rules from `CLAUDE.md` stay in force: distro Python only, fail open, tests green without a camera.

---

## 1. Responsiveness — "it just opens"

Goal: **accept in under 500 ms wall-clock at the lock screen**, under 700 ms for sudo, with the feedback line already on screen when the camera starts.

| # | Item | Mechanism | Saves | Effort |
|---|---|---|---|---|
| R1 | **Parallel camera open** | Open IR and RGB in two threads while dlib models load in the main thread; `VideoCapture.open` releases the GIL in the ioctls | ≈ 500 ms | S |
| R2 | **No probes on the auth path** | `hw.profile()` spawns `powerprofilesctl` and probes OpenCL on every attempt; cache the profile in `/etc/ward/hw.json` at `setup`/`doctor` time, read it in `auth` | ≈ 80 ms | S |
| R3 | **Camera busy → short retry** | Two retries 150 ms apart before giving up (`camera` failures in the log are the previous attempt still releasing the device) | fewer false "camera" fallbacks | S |
| R4 | **Resident helper (`ward-authd`)** | systemd socket-activated, root, holds cv2 + dlib models in memory, idle-exits after 10 min. `ward-auth` becomes a 5 ms client (`socket` + JSON) that falls back to the in-process engine when the daemon is absent. Same code path (`Engine`), no logic duplication | ≈ 800 ms | M |
| R5 | **Pre-warm on lock** | The daemon subscribes to logind `Lock`/`Unlock`/`PrepareForSleep` and GNOME/KDE screensaver `ActiveChanged`. On lock it opens the cameras and starts the exposure settle; the first frame is ready before the user lifts the lid. Camera released again after 20 s idle so calls/browsers can use it | ≈ 700 ms at the lock screen | M |
| R6 | **YuNet at 240 px for the first frames** | Detect at 240 on frames 1–3, 320 afterwards; detection is 4 ms → 2 ms but the resize saves more on 1080p RGB | ≈ 20 ms/frame | S |
| R7 | **Window ends early on a confident accept** — **done** (reject on the median of the last frames, 2026-10-22) | Already true; keep it. Add: end early on a confident *reject* (raw > 0.65 on 6 fused frames) instead of running to the 2.5 s window — the password prompt appears sooner | ≈ 1.5 s on a stranger | S |

Order: R1 + R2 + R3 + R7 together (one day, no architecture change), then R4 + R5 (the daemon; two to three days including the fallback tests), R6 last.

Verification: `ward bench --path` prints the stage timeline (import / open / first frame / detect / embed / decide); the lab CSV gets an `elapsed_ms` column; `doctor` shows the median of the last 20 attempts and flags > 800 ms.

---

## 2. Reliability — "it works in the dark, it doesn't take a stranger"

Goal: **FRR (false reject) < 2 % at the desk across day/night/glasses; FAR (false accept) unmeasurable on the lab's stranger recordings; phone/print spoof rejected 100 %.**

| # | Item | What is wrong today | Plan | Effort |
|---|---|---|---|---|
| L1 | **Emitter, measured not assumed** | `doctor` trusts the tool's config file while the IR frame is dim | `doctor` and `setup` toggle the emitter (linux-enable-ir-emitter's UVC quirk) and *meter* the IR face; `emitter toggle-bench` measures on/off latency | S |
| L2 | **Whitening recalibrated** | Threshold is a leave-one-out percentile over consecutive enrollment frames, far tighter than a live fused embedding; the path never fires | Fit the threshold on *held-out enrollment cells* and on real accepted attempts (lab recordings, learned samples); target: `w_median` < 1 for the user, > 2 for strangers in `ward lab` summary. Until then `doctor` flags a raw gate above 0.45 | M |
| L3 | **Online learning that learns** — **done** (bounded 2026-10-22: novelty vs enrolled only, per-camera cap) | Novelty 0.15 is below frame noise (0.15–0.32), so nearly every accept "learns"; each learned model has one sample, so whitening never sees it | Store the window's per-frame embeddings as samples; novelty 0.25; learn only after ≥ 1 real PASS; cap 2 per bin; whitening refit from them | S |
| L4 | **Shadow-test voter** | Metric done, hook off | Needs L1's toggle: two frames on/off inside the window; PASS on self-shadowing CV. Second real voter → the "2 PASS" rule becomes true | M |
| L5 | **Anti-spoof CNN** | MiniFASNetV2 bundled (2026-10-23), PASS on real daylight faces, conservative FAIL | `doctor --spoof`: record a phone photo / print and measure, then tighten FAIL_P from data | S |
| L6 | **Emitter-differential imaging** | Daylight swamps the IR | `I_on − I_off` frames once L1/L4 exist; cancels ambient light, the principled fix for windows-behind-the-user | L |
| L7 | **Replay harness** | Engine behaviour can only be tested live | `WARD_REPLAY=dir` feeds recorded frame sets (from `lab --record-frames`) through `Engine.authenticate`; a regression corpus (day, night, glasses, phone, print, stranger) runs in CI without hardware | M |
| L8 | **Exposure for faces near a window** (dark-room half done: ALS → IR only, watchdog) | Firmware auto-exposure on IR, no controls | Meter the face (done) → when the face is blown, drop to the RGB path earlier (adaptive primary already exists; tune its 3-set delay to 2) | S |
| L9 | **Multi-user desks** | Untested | PAM_USER selects the store; the lab's per-subject summary doubles as the household FAR test; document | S |

Order: L1 → L3 → L2 → L7 (so L4/L5 can be validated offline) → L4 → L5 → L6.

---

## 3. UI / UX — "one app, no root hacks"

What exists: GTK4/libadwaita windows for Enroll, Emitter and Lab, launched by root CLI commands over the invoking user's Wayland socket; a terminal `doctor`; on-screen feedback at every prompt.

| # | Item | Plan | Effort |
|---|---|---|---|
| U1 | **Privilege split** | The GTK app runs *as the user*; privileged work (store, config, cameras, PAM) goes through a system D-Bus service `org.ward.Manager` (the same daemon as R4) guarded by polkit actions `org.ward.enroll` / `org.ward.configure` (admin auth, retained 5 min). Removes `GTK_A11Y=none` and the socket-borrowing hacks; the app appears in the launcher | L |
| U2 | **One window** | `ward` (no args) opens an Adwaita window with a sidebar: **Status** (doctor as cards with fix buttons), **Enroll** (existing window as a page; "add a condition" 10-second mini-enrol for glasses/night), **Lab**, **Attempts** (log as a table with accept rate, median time, reasons), **Settings** (policy level, notify, cameras). KDE runs GTK4 apps natively; no Qt port | M–L |
| U3 | **Setup wizard** | The `setup` flow as pages: cameras → emitter test (lights-off prompt with live IR/RGB) → enroll → PAM → self-check → "try sudo". Text `setup` stays for servers/SSH | M |
| U4 | **Enrollment polish** | Auto-finish 1.5 s after coverage completes; glasses on/off pass suggested when a spectacle frame is detected; a summary card (poses, samples, whitening threshold, 3D model rms) at the end | S |
| U5 | **`ward log` grows up** | `--summary` (accept rate, median t, reason histogram, per-service), `--follow`, colour, `--json`. Same data feeds the Attempts page | S |
| U6 | **`doctor` gets opinions** | Flags drift: raw gate > 0.45, PAM line outdated, emitter dim in daylight, learned-model churn, store older than the recogniser; each with the one command that fixes it | S |
| U7 | **Desktop file + icon** | `ward.desktop`, symbolic icon, so the notification's `desktop-entry` hint resolves and the app is in the launcher | S |
| U8 | **Accessibility & i18n** | PAM messages and window labels via gettext; feedback lines are plain text so screen readers and greeters read them | S |

Order: U5 + U6 + U7 + U4 now (they need nothing else), U1 with the daemon, then U2/U3 on top of U1.

---

## 4. Integration — distros, greeters, engines

### 4.1 PAM stacks per distro

| Family | Stack file | The right tool | Plan |
|---|---|---|---|
| Debian / Ubuntu | `/etc/pam.d/common-auth` | `pam-auth-update` profile in `/usr/share/pam-configs/ward` | Ship the profile; `ward pam enable` uses it when present, direct edit otherwise (today) |
| Fedora / RHEL | `system-auth`, `password-auth`, `fingerprint-auth` | `authselect` custom profile (`authselect create-profile ward -b local`, `with-ward` feature) | Ship the profile; SELinux: label the helper `ward_auth_exec_t`, allow `xdm_t`/`sudo_t` transitions and `v4l_device_t` access (policy module, tested in a Fedora VM) |
| Arch / Manjaro | `/etc/pam.d/system-auth`, `sudo`, `sddm`, `kde` | direct edit (documented) | AUR `ward` package; `pam enable` edits `system-auth` |
| openSUSE | `common-auth-pc` | `pam-config` | `pam-config` module definition; else direct edit |
| NixOS | `security.pam.services.*` | module option `security.pam.services.<n>.wardAuth` | NixOS module in the repo, like Howdy's |

`ward pam enable` becomes stack-aware: detect the tool, prefer it, fall back to the current in-place edit with backup. `doctor` reports which stack manager is in use.

### 4.2 Greeters, lockers, prompts — status matrix

| Surface | PAM service | Starts PAM without a keypress? | Shows PAM info? | Status |
|---|---|---|---|---|
| GDM login, GNOME Shell unlock | `gdm-password` | yes | yes | works, feedback shown |
| GNOME polkit agent | `polkit-1` | yes | yes | works |
| sudo / su | `sudo`, `su` | yes | yes (stderr) | works, spinner |
| KDE lock screen (Plasma 6) | `kde` | yes — Plasma 6 runs an *active* and a *fingerprint* PAM stack side by side ([kscreenlocker !318](https://invent.kde.org/plasma/kscreenlocker/-/merge_requests/318)); Ward belongs in the fingerprint-style stack | yes | to verify on a Plasma 6 VM |
| KDE polkit agent | `polkit-1` | yes | yes | expected to work |
| SDDM | `sddm` | **no** — PAM runs when the user submits; press Enter with an empty password ([Arch forum](https://bbs.archlinux.org/viewtopic.php?id=309702)) | yes | document; consider an SDDM "face" hint theme snippet |
| LightDM | `lightdm` | no (same as SDDM) | yes | document |
| greetd + tuigreet/gtkgreet | `greetd` | no | partial | document |
| swaylock / hyprlock / gtklock | `swaylock`, `hyprlock` | no — Enter with empty password | no (silent) | document; feedback via notification only |
| `login` on a tty | `login` | yes | yes | policy default: **off** on ttys (`behaviour.tty = false`) |
| `run0` (systemd) | via polkit | yes | yes | works through the agent |

Fix for the "press Enter" surfaces: a `behaviour.retry_on_empty_password` note in docs; nothing to do in code, PAM decides.

### 4.3 Cameras

- **libcamera-only sensors** (Intel IPU6 on 2023+ laptops): no V4L2 node for the RGB sensor; IR modules usually stay UVC. Plan: IR-only profile B works; document `v4l2-relayd`/libcamerify for RGB; refuse `v4l2loopback` remains (spoof surface).
- **PipeWire camera portal**: root PAM helper cannot use the portal; the daemon (R4) could later be a PipeWire client so browsers and Ward share the camera instead of "busy".
- **Multiple IR modules**: `cameras.pick_pair` prefers the same USB module; add a `ward cameras` command to list, test and choose.

### 4.4 Engines (recognisers and detectors)

| Engine | Status | Why it matters |
|---|---|---|
| dlib ResNet (128-d) + YuNet | current | proven; needs `python3-dlib`, which Debian, Ubuntu and Fedora package ([python3-dlib on Fedora](https://packages.fedoraproject.org/pkgs/dlib/python3-dlib/)), Arch via AUR |
| **OpenCV-only path**: YuNet (detect + 5 landmarks) + SFace (OpenCV zoo, 128-d, Apache) | planned | drops the dlib dependency entirely → runs on every distro with `python3-opencv` + `numpy`, no compile; must be *measured* in the lab against dlib on IR before it becomes a default |
| ArcFace r50 ONNX via `cv2.dnn` | optional | higher accuracy, ~100 MB, 150 ms/embedding on CPU; opt-in for machines that can afford it |
| MiniFASNet (anti-spoof) | L5 | RGB liveness voter |

The store already records `recogniser`; a change forces re-enrol. The lab compares engines on the same frames (add an `--engine` switch) so the choice is data, not taste.

### 4.5 Packaging and release

`.deb` (dh-python, depends on distro python3-opencv/dlib), RPM spec (Fedora COPR), AUR PKGBUILD, NixOS module; files move from `/usr/local` to `/usr/lib/ward` + `/usr/libexec/ward/ward-auth`; a `ward` systemd unit + socket for the daemon; signed tags; GitHub Actions running the test suite and the replay corpus (L7). Publish (`<you>` placeholders) is the first step of this section.

---

## 5. Keyring — GNOME Keyring and KWallet

**The fact:** `pam_gnome_keyring.so` and `pam_kwallet5.so` unlock the login keyring with the password PAM received (`PAM_AUTHTOK`). A face accept produces none, so after a face *login* the keyring stays locked and a password dialog appears later ([GNOME Discourse](https://discourse.gnome.org/t/unlocking-gnome-keyring-after-non-password-pam-authentication/36030)). The same is true for fingerprints.

**What already works:** screen unlock, sudo, polkit. The session is running, the keyring is already open; face auth there never touches it. The gap is **the first login of a session** only.

Plan, in order of how much it stores:

| Stage | Approach | Stores a secret? | Ships when |
|---|---|---|---|
| K0 | **Explain.** `doctor` and the README say: face unlock at login leaves the keyring locked until first use; face unlock everywhere else is unaffected. Option to disable Ward for `gdm-password`'s *login* phase only (PAM_TYPE + no existing session → skip), so login stays password and everything after is face | no | now |
| K1 | **Sealed login credential, opt-in.** `ward keyring enable` asks for the login password, verifies it against PAM, and seals it with `systemd-creds encrypt --with-key=tpm2` bound to PCR 7 (Secure Boot state) — or with the host key on machines without a TPM, which then requires full-disk encryption (checked, refused otherwise). A native module `pam_ward.so` (≈ 300 lines of C, replaces `pam_exec`) runs the helper, and on an accept *that meets the strict policy* (§6) unseals the credential and calls `pam_set_item(PAM_AUTHTOK)`. `pam_gnome_keyring`/`pam_kwallet5` then unlock as if the password had been typed | yes, sealed to TPM/FDE | v0.6 |
| K2 | **Separate keyring password.** If the keyring password differs from the login password, K1 stores that one instead; verified by unlocking the keyring over the Secret Service API from the user's session | yes | with K1 |
| K3 | **No-secret alternatives, documented.** KWallet with a GPG-key wallet; a blank login keyring on FDE-only machines (GNOME's own trade-off). These avoid K1 at the price of weaker at-rest protection | no | docs |

The native module also cleans up the integration: proper conversation messages instead of the stdout hack, `PAM_AUTHTOK` for the keyring, and the same failure semantics (`PAM_AUTH_ERR` → next module). It never replaces the helper's Python engine; it only launches it.

---

## 6. Security policy

Ward is **convenience-grade authentication with a password behind it**, on a 2D sensor. The policy makes that explicit and tunable.

**Principles**

1. **Fail open, always.** Every error path ends in the password. No lockout can ever remove the password path.
2. **Never weaken the stack.** `pam enable` inserts one `sufficient` line before `pam_unix` and never removes or reorders anything else; `faillock`, `pam_deny`, 2FA modules stay where they are. A face accept skips *only* the password.
3. **Least privilege.** The helper runs as root only to read the store and the camera; the daemon (R4) drops CAP_* except what V4L2 needs; the GUI never runs as root (U1). Model files 0600 root, config 0644 root, log 0640 root:adm.
4. **No images at rest.** Embeddings only; `log_images` is removed from the config (it was never implemented) rather than left as a promise. `ward models --wipe` erases everything for a user; documented as biometric data under GDPR-style rules.
5. **Data stays local.** No network at runtime; models bundled or fetched at setup with pinned SHA-256; no telemetry.

**Policy levels** (`ward tune policy <level>`, per PAM service overridable: `policy.sudo`, `policy.polkit`, `policy.login`, `policy.unlock`)

| Level | Recognition | Liveness | Learning | Use |
|---|---|---|---|---|
| `convenience` (today's behaviour) | raw ≤ 0.40 or whitened path | ≥ 1 PASS, 0 FAIL | on | lock screen on a private laptop |
| `balanced` (**new default**) | raw ≤ 0.40 or whitened; `require_rgb_match` when RGB is bright | ≥ 2 PASS, 0 FAIL — until a second real voter exists this means *cross + motion* (move your head) or *cross + shadow* | on, strict novelty | lock screen, sudo |
| `strict` | raw ≤ 0.35, RGB match required, no whitened path | 2 PASS including one of shadow/CNN; disabled if neither is available | off | polkit, keyring release (K1), shared machines |
| `off` | — | — | — | per-service opt-out (e.g. `login` on ttys) |

**Rate limiting.** `lockout_failures` gets implemented: N liveness FAILs (spoof signals, not "no face") within 10 min → face disabled for 15 min for that user, with a notification and a log line. Password unaffected. Default N = 3 on `balanced`.

**Context checks.** Skip face auth when: SSH (exists), `PAM_RHOST` set, inside a container, tty login under `policy.login = off`, and when the model store is newer than the recogniser build (tamper/mismatch).

**Store hygiene.** Learned models capped, listed, forgettable (exists); a notification when a learned model is added ("Ward learned your night-time look") so silent drift is visible; `doctor` warns when learned models outnumber enrolled ones.

**Audit.** One line per attempt (exists) plus structured journald fields (`WARD_RESULT`, `WARD_SERVICE`, distances) so `journalctl -t ward` works; the Attempts page reads both.

**Threat model, restated.** Defeated: phone/tablet/monitor replay, printed photo, IR-tuned print (with shadow test), darkness, camera DoS. Partial: two aligned prints, look-alike siblings (raise level to `strict`). Not defended: 3D masks, root compromise, physical coercion. The README keeps saying this in plain words.

---

## 6a. Status log

- **2026-09-07** — 0.3 started: R1 R2 R3 R7 (fast path: async parallel camera open with busy retry, cheap hw.apply, early reject), L3 (learning keeps per-frame samples, novelty 0.25), U5 (`ward log --summary/--follow/--json`, `attempts.py`), U6 (doctor flags loose gate, learned-model churn, slow median, emitter not firing in a lit room), U7 (`data/ward.desktop` + icons installed, hidden entry), K0 (`behaviour.skip_login`, doctor/README explain the keyring), L1 partial (doctor re-applies the emitter and meters the IR frame; the on/off toggle bench is still open). Measured first-frame time 1.1 s (was 1.4 s); the remaining floor is the IR module's 520 ms open + 200 ms stream start → R4/R5 next.

- **2026-09-07 (later)** — 4.4 engines: pluggable recogniser (`engine/recognise.py`), ArcFace mbf/r50 through OpenCV DNN with pinned downloads (`ward recogniser install|use|status`, setup step), stores keyed by recogniser, one distance scale for all gates. Measured on the reference camera: mbf 5 ms, r50 57 ms per embedding; same-person spread comparable to dlib after scaling; impostor calibration (the point) still needs a second subject in the lab. Windows-Hello parity table added to the README; remaining gaps are the emitter-lit liveness (L1/L4) and a measured FAR (L7 + lab).

- **2026-09-07 (evening)** — L1 done: `engine/emitter.py` drives the LEDs through the UVC extension unit using linux-enable-ir-emitter's instruction; `ward emitter toggle-bench` measures the switch and writes `liveness.shadow_toggle_ms`; doctor reads the tool's real state (on this laptop: **no working instruction** — the emitter was never actually configured, hence dim IR). L4 done: the shadow voter runs once per attempt after recognition when the bench proved the hardware (`shadow_test = auto`). L7 done: `lab --record-frames`, `auth --replay`, `ReplayCapture`; the engine loop now runs in the test suite on synthetic frames. Hardware validation of L1/L4 waits on a successful `emitter configure`.

- **2026-09-07 (night)** — **K1 done ahead of schedule**: `data/pam_ward.c` native module (no -dev headers needed, `[success=N]` placement so pam_gnome_keyring still runs), `engine/keyring.py` (systemd-creds TPM2/host sealing, shadow verification, PAM-line repair), `ward keyring status|enable|disable|repair`, real-stack test through `pam_start_confdir`. Found and flagged: an earlier experiment had commented out the keyring PAM lines on the reference machine. **L1 corrected**: the emitter was working all along (LEDs on by default; the tool refuses to configure a working emitter) — `ward emitter probe` finds the LED control by measurement (unit 5 sel 2 on the reference module), `toggle-bench` measured 100–133 ms, shadow voter active; doctor meters the face instead of the frame. Also: `liveness` failure reason with per-camera light states in the log; motion voter no longer FAILs on ≤ 6 px shifts; sudo spinner on PAM_TTY; RGB-less attempts traced to the environment, not the code.

## 6b. Calibration for other machines (added 2026-09-07)

Principle: no constant in the config is a guess — every number is measured on the machine or expressed relative to a measurement.

| Item | Status |
|---|---|
| Recognition gate + whitening threshold from a guided session; rolling refresh from accepts; evidence in the store | **done** (`engine/calibration.py`, `ward calibrate`, end of enrollment) |
| Emitter control by measurement (any lighting) | **done** for small controls; generalise: read extension units and control sizes from USB descriptors, try both directions (LEDs off by default) |
| Brightness/liveness thresholds relative to face size and measured face level; shadow delta relative to the toggle bench | todo |
| ArcFace radius verified by the session; tightened by the lab with a second subject | todo |
| Timing (settle, window) from measured auto-exposure and first-frame latency | todo |
| `/etc/ward/calibration.toml` with provenance; doctor compares live readings and says "recalibrate" | partial (evidence lives in the store's `calib.calibration`) |
| `ward report`: anonymised hardware/calibration summary for issues → module database in the repo | todo |
| Replay corpora from other machines in CI | harness done; corpora needed |

- **2026-09-08** — startup & UX fixes: emitter turned ON before the cameras open so the ~400 ms LED ramp overlaps the open (fixes "LEDs slow to light then recognise"; also makes the shadow voter reliably PASS); the calibration session offloads dlib embeddings to the `_Sampler` process (UI thread 13 ms/frame vs 85 ms inline — smooth preview) with movement cues drawn on the camera view (minimal text); keyring credential defaults to the fast host key on an encrypted disk and the PAM module decrypts it in parallel with recognition (removes the 2–4 s TPM freeze at unlock).

- **2026-09-09 (v0.3.0)** — the lab fixed for unprivileged/other-hardware runs (recordings dir honours `WARD_ETC`; one-camera guard); **compatibility matrix** `tests/test_compat.py` (profiles A/B/C, frame sizes, stranger, no emitter, learning, the lab); Python 3.10 TOML fallback, 3.13 crypt fallback, dlib optional with ArcFace+YuNet, resolution-independent motion thresholds, text fallback without libadwaita; **identities** (`enroll --identity`) and **conditions** (`enroll --add glasses`); **policy presets** + per-service overrides + **spoof lockout** (`engine/policy.py`); `setup` rewritten as the complete 7-step chain; **the app** (`ward app`, U2/U3 delivered as a CLI driver; U1 privilege split still open); version 0.3.0, CHANGELOG, CI workflow. Still open: the resident daemon (R4/R5) for the sub-second unlock, packaging (.deb/RPM/AUR), the anti-spoof CNN, `ward report`.

- **2026-10-22** — night unlock root cause found and fixed (RGB open in the dark starves the IR stream on this module → ALS picks IR only below 5 lux, watchdog reopens IR; IR-only dark vs daylight models 0.31); enrolment frame selection (12° pose tolerance, last-third holdout, usable-sample completion) and in-window glasses sweep; bounded learning (novelty vs enrolled only, per-camera cap, whitening ±20 % with floor) — **L3 hardened**; early reject on the median (**R7 actually fires now**); PAM: `poll()` timeout + process-group kill, `SIGALRM` backstop, `system-auth`, symlink-safe writes, authselect refused (**4.1 partial**: direct edit on Debian/Fedora/Arch, no pam-auth-update/authselect profile yet); distro-aware install. 168 tests.

## 6c. Open decision: liveness votes (2026-10-22)

Measured in the log: **~92 % of accepts carry exactly one PASS** (cross-spectrum *or* shadow, rarely both). `_decide` accepts with `n_pass ≥ 1` when the other voters abstain, so `required_votes = 2` is effectively 1. Enforcing 2 today would reject most IR-only night unlocks (no RGB → no cross-spectrum; only shadow can vote). Options, for the user to choose:

| Option | Effect | Cost |
|---|---|---|
| a. Keep 1 PASS + 0 FAIL, say so in docs/doctor | honest status quo | one voter stands between a print and an accept |
| b. Require 2 only when both cameras are open (lit room); 1 in IR-only dark | daytime spoof bar rises | daytime unlocks fail when shadow abstains |
| c. Make shadow run every attempt (not once) so it nearly always votes, then require 2 | true 2-voter rule | +1 toggle (~130 ms) per attempt; depends on E4 |
| d. Wait for the anti-spoof CNN (L5) as a second voter | principled | RGB-only voter: no help in the dark |

Not changed without the user's decision.

## 6d. Remaining review items (2026-10-22)

| # | Item | Plan |
|---|---|---|
| D1 | **doctor → health unification** | `doctor` still has its own checks beside `health.collect()`; make doctor a renderer of `health` so the app and the CLI cannot disagree |
| E4 | **Shadow voter metric hardening** | the self-shadowing metric is uncalibrated against prints/phones; make it relative to the toggle bench and the face level, add replay cases |
| E5 | **Cross-spectrum IoU ≈ 0 → lockout** | when IR and RGB boxes don't overlap (misalignment, not a spoof) cross FAILs as `spoof:*`; repeated, that trips the spoof lockout. Treat iou ≈ 0 as ABSTAIN/misalignment, re-align |
| E6 | **shadow_pair off-frame check** | verify the "off" frame is actually dark (LED switched) before comparing, else ABSTAIN instead of voting on two lit frames |
| C1 | **GREY fourcc / `is_ir` heuristic** | IR detection by fourcc/greyness misclassifies some cameras (GREY RGB, coloured IR); prefer USB interface + descriptor facts, keep the heuristic as fallback |

## 7. Milestones

| Version | Theme | Contents | Exit criterion |
|---|---|---|---|
| **0.3** | fast & visible | R1 R2 R3 R7 · U5 U6 U7 · L1 L3 · K0 · publish to GitHub | median accept < 1.0 s in the log; feedback on every surface; repo public |
| **0.4** | reliable | L2 L7 L4 L5 · `balanced` policy default · lockout · PAM tools for Debian/Fedora | lab: user FRR < 2 % day+night, spoof corpus 0 accepts; CI green on replay corpus |
| **0.5** | resident | R4 R5 · U1 (D-Bus + polkit) · U2 U3 (the app) · desktop file | accept < 500 ms at the lock screen; `ward` app in the launcher, no root windows |
| **0.6** | keyring & native PAM | `pam_ward.so` · K1 K2 · `strict` policy for the release | keyring unlocked after a face login on GNOME and KDE with TPM sealing |
| **0.7** | everywhere | OpenCV-only engine measured · .deb / RPM / AUR / NixOS · Plasma 6, SDDM, swaylock verified | installs from a package on four distros; matrix in §4.2 has no "to verify" |
| **1.0** | done | L6 differential imaging · docs pass · security policy published | six months of use without a lockout or a false accept on the lab corpus |

Each milestone ends with the three checks that already exist: tests green, `ward doctor` clean, `ward lab` summary attached to the release notes.

---

## 8. How progress is measured

- **Latency:** `ward log --summary` median and p90 per PAM service; target column in §1.
- **Accuracy:** `ward lab` recordings per subject (you day/night/glasses, a friend, a phone photo, a print) → accept rate per policy; the CSVs are the regression corpus of L7.
- **Coverage:** the greeter matrix (§4.2) as a checklist in the release notes; each row verified in a VM or on hardware, date noted.
- **Safety:** zero lockouts by construction (the password path is never touched); every failure reason visible in the log and in the Attempts page.

Sources consulted for §4 and §5: [kscreenlocker !318 (flexible authenticators)](https://invent.kde.org/plasma/kscreenlocker/-/merge_requests/318), [Howdy + SDDM, Arch forums](https://bbs.archlinux.org/viewtopic.php?id=309702), [python3-dlib on Fedora](https://packages.fedoraproject.org/pkgs/dlib/python3-dlib/), [GNOME Keyring after non-password PAM auth](https://discourse.gnome.org/t/unlocking-gnome-keyring-after-non-password-pam-authentication/36030), [GNOME/Keyring on ArchWiki](https://wiki.archlinux.org/title/GNOME/Keyring).
