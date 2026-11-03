# Ward — Roadmap

*Written 2026-09-07 against v0.2.0 + session feedback. **§0 and the milestones updated 2026-11-03 for v0.5.0 (launched).** Companion to `DESIGN.md` (what exists and why) and `README.md` (the guide). This document says what comes next, in what order, and why. The dated status log (§6a–§6e) is the running record of what landed when; the numbered plans (§1–§6) are kept as written, with shipped items marked **done**.*

## 0. Where we stand

Ward is at **v0.5.0** — released 2026-10-28 and publicly launched 2026-10-29. Since this roadmap was first written (v0.2.0), the whole *fast*, most of *reliable*, and the *resident* + *app* + *packaging* work has shipped:

| Area | State at v0.5.0 |
|---|---|
| Resident engine | **shipped** — `ward.service` keeps OpenCV/dlib/models warm; `ward-auth` is a ~30 ms client that falls back in-process when the daemon is absent; zero-capability sandbox (`systemd-analyze security` 0.9), `Type=notify` watchdog, ~200 MB flat RSS (**R4 done**; pre-warm-on-lock **R5** still open) |
| The app | **shipped** — native GTK4/libadwaita manager (`ward`, the front door), runs as the user with every privileged action via `pkexec`/polkit (no GTK as root), and a full setup wizard drawn in-window (**U1/U2/U3 done**) |
| Recognition | one guided enrol sets the gate from held-out frames of your own face; ArcFace optional; the whitened path was measured never to fire live → **off by default**, documented experimental (**L2 closed by decision**) |
| Liveness | IR cross-spectrum + **bundled anti-spoof CNN** (MiniFASNetV2, **L5 done**) + motion + optional emitter shadow; single-PASS + 0-FAIL is the honest rule, spoof-lockout **off by default** (a lockout only creates DoS; the password is always there) |
| Emitter | works by default on the reference module; found by measurement (`emitter probe`), shadow bench wired (**L1 done**) |
| Keyring | native `pam_ward.so`; sealed login credential (host-key default / TPM2) unlocks GNOME Keyring / KWallet after a face login (**K1/K2 done**) |
| Packaging | **`.deb` + Ubuntu PPA + Fedora COPR live; AUR primed (pending account)**; one `stage.sh` layout; CI builds the `.deb` and auto-publishes AUR/COPR on a `v*` tag (**§4.5 largely done**) |
| Tests | **319**, no camera; portability matrix (Debian/Ubuntu/Fedora/Arch/Tumbleweed) + replay corpus, both in CI |

The original v0.2.0 snapshot (56 tests, a dim emitter, a whitened path that never accepted, ~1.5 s cold overhead, one real voter) is preserved in the dated status log below as the baseline these milestones were measured against; live latency now reads from `ward log --summary`.

Everything below is ordered by *user-visible payoff per unit of risk*. Golden rules from `CLAUDE.md` stay in force: distro Python only, fail open, tests green without a camera.

---

## 1. Responsiveness — "it just opens"

Goal: **accept in under 500 ms wall-clock at the lock screen**, under 700 ms for sudo, with the feedback line already on screen when the camera starts.

| # | Item | Mechanism | Saves | Effort |
|---|---|---|---|---|
| R1 | **Parallel camera open** | Open IR and RGB in two threads while dlib models load in the main thread; `VideoCapture.open` releases the GIL in the ioctls | ≈ 500 ms | S |
| R2 | **No probes on the auth path** | `hw.profile()` spawns `powerprofilesctl` and probes OpenCL on every attempt; cache the profile in `/etc/ward/hw.json` at `setup`/`doctor` time, read it in `auth` | ≈ 80 ms | S |
| R3 | **Camera busy → short retry** | Two retries 150 ms apart before giving up (`camera` failures in the log are the previous attempt still releasing the device) | fewer false "camera" fallbacks | S |
| R4 | **Resident helper** — **done** (`ward.service` + `ward/daemon.py`, v0.4.0) | systemd service, root, holds cv2 + dlib models warm; `ward-auth` (`ward/client.py`) is a ~30 ms client that falls back to the in-process engine when the daemon is absent. Same `Engine`, no logic duplication; zero-cap sandbox + `Type=notify` watchdog | ≈ 800 ms | M |
| R5 | **Pre-warm on lock** — **open (top remaining latency win)** | The daemon subscribes to logind `Lock`/`Unlock`/`PrepareForSleep` and GNOME/KDE screensaver `ActiveChanged`. On lock it opens the cameras and starts the exposure settle; the first frame is ready before the user lifts the lid. Camera released again after 20 s idle so calls/browsers can use it. (The daemon from R4 exists; this is the subscription + pre-open it still lacks.) | ≈ 700 ms at the lock screen | M |
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
| L5 | **Anti-spoof CNN** — **done** (active since v0.4.0) | MiniFASNetV2 bundled, a second independent daylight PASS; conservative sticky FAIL | *Tail still open:* `doctor --spoof` — record a phone photo / print and tighten FAIL_P from data (no spoof corpus exists yet) | S |
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
| U1 | **Privilege split** — **done** (v0.4.0, via pkexec not D-Bus) | The GTK app runs *as the user*; every privileged action (store, config, cameras, PAM) runs through `pkexec` against the polkit action `org.ward.app` (`auth_admin`, **not** remembered, so one grant can't be reused for another subcommand), never GTK as root. The resident daemon (R4) carries the auth path. A persistent `org.ward.Manager` D-Bus service was *not* needed — pkexec + the CLI meet the same no-root-GUI goal | L |
| U2 | **One window** — **done** (v0.4.0) | `ward` (no args) opens the Adwaita manager: **Overview** (on/off, reliability verdict, test unlock, things to fix), **People** (identities, cameras, conditions, register/re-enrol), **Reliability** (last 50 attempts by light/service/person), **Settings** (policy + per-service, behaviour, keyring), **Advanced** (setup, recogniser, lab, doctor). Register/calibrate run *inside* the window via a streamed UI | M–L |
| U3 | **Setup wizard** — **done** (v0.4.0) | The `setup` flow as in-window pages: cameras → emitter → anti-spoof → register → keyring → turn-on, each reversible with Skip/Back. Text `setup` stays for servers/SSH | M |
| U4 | **Enrollment polish** — **mostly done** (v0.3–0.5) | Auto-closes when coverage completes; in-window glasses sweep ("I also wear glasses"); a done card lists missed poses and leads with "Your face is set up". *Still open:* auto-suggest the glasses pass when a spectacle frame is detected; richer end-of-sweep summary (samples, whitening threshold, 3D-model rms) | S |
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

The store already records `recogniser`; a change forces re-enrol. The lab compares engines on the same frames so the choice is data, not taste. **Added in v0.4.0:** offline accuracy harnesses — `ward benchmark {lfw,spoof,ir,redteam}` (LFW impostor/genuine EER, CelebA-Spoof APCER/BPCER, CASIA-SURF/WMCA cross-spectrum PAD, and a red-team of digital attacks on your own frames). **Still open:** the **OpenCV-only SFace** path (drops the dlib dependency entirely — must be *measured* against dlib on IR before it can be a default), and ArcFace impostor calibration of `ARCFACE_RADIUS` (needs a second subject in the lab).

### 4.5 Packaging and release — **largely done** (v0.4.0–0.5.0, launched 2026-10-29)

**Done:** one `packaging/stage.sh` layout shared by `install.sh` and the packages (files under `/usr`, `pam_ward.so` compiled at build time, the `ward.service` unit, never edits PAM); `build-deb.sh` (offline `dpkg-deb`), `ward.spec` (RPM/COPR with a `.copr/Makefile` `make_srpm` entry), `PKGBUILD` + `.SRCINFO` (AUR), native `debian/` source package (PPA). **Live channels:** `.deb` on each GitHub release, **Ubuntu PPA** `ppa:0ctave/ward` (noble), **Fedora COPR** `0ctave/ward`. CI: `tests`, `lint`, `codeql`, `distros` (portability matrix incl. the replay corpus), `release` (tag → `.deb` + GitHub release), `publish-packages` (tag → AUR + COPR, secret-gated), `changelog` (fragment required). Release tooling: `tools/release.py` + `changelog.d/` fragments.

**Still open:** publish the **AUR** package (account registration was closed for bot-abuse — PKGBUILD + CI are primed, waiting on the account); a **NixOS** module (like Howdy's); more Ubuntu series in the PPA (only noble so far).

---

## 5. Keyring — GNOME Keyring and KWallet

**The fact:** `pam_gnome_keyring.so` and `pam_kwallet5.so` unlock the login keyring with the password PAM received (`PAM_AUTHTOK`). A face accept produces none, so after a face *login* the keyring stays locked and a password dialog appears later ([GNOME Discourse](https://discourse.gnome.org/t/unlocking-gnome-keyring-after-non-password-pam-authentication/36030)). The same is true for fingerprints.

**What already works:** screen unlock, sudo, polkit. The session is running, the keyring is already open; face auth there never touches it. The gap is **the first login of a session** only.

Plan, in order of how much it stores:

| Stage | Approach | Stores a secret? | Ships when |
|---|---|---|---|
| K0 | **Explain.** `doctor` and the README say: face unlock at login leaves the keyring locked until first use; face unlock everywhere else is unaffected. Option to disable Ward for `gdm-password`'s *login* phase only (PAM_TYPE + no existing session → skip), so login stays password and everything after is face | no | now |
| K1 | **Sealed login credential, opt-in.** `ward keyring enable` asks for the login password, verifies it against PAM, and seals it with `systemd-creds encrypt --with-key=tpm2` bound to PCR 7 (Secure Boot state) — or with the host key on machines without a TPM, which then requires full-disk encryption (checked, refused otherwise). A native module `pam_ward.so` (replaces `pam_exec`) runs the helper, and on an accept unseals the credential and calls `pam_set_item(PAM_AUTHTOK)`. `pam_gnome_keyring`/`pam_kwallet5` then unlock as if the password had been typed | yes, sealed to TPM/FDE | **done v0.3.0** (host-key default, scoped to login services in v0.5.0) |
| K2 | **Separate keyring password.** If the keyring password differs from the login password, K1 stores that one instead; verified by unlocking the keyring over the Secret Service API from the user's session | yes | **done** (with K1) |
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

## 6c. Decision: liveness votes (2026-10-22, resolved v0.4.0)

Measured in the log: **~92 % of accepts carry exactly one PASS** (cross-spectrum *or* shadow, rarely both). `_decide` accepts with `n_pass ≥ 1` when the other voters abstain, so `required_votes = 2` is effectively 1. Enforcing 2 today would reject most IR-only night unlocks (no RGB → no cross-spectrum; only shadow can vote). Options, for the user to choose:

| Option | Effect | Cost |
|---|---|---|
| a. Keep 1 PASS + 0 FAIL, say so in docs/doctor | honest status quo | one voter stands between a print and an accept |
| b. Require 2 only when both cameras are open (lit room); 1 in IR-only dark | daytime spoof bar rises | daytime unlocks fail when shadow abstains |
| c. Make shadow run every attempt (not once) so it nearly always votes, then require 2 | true 2-voter rule | +1 toggle (~130 ms) per attempt; depends on E4 |
| d. Wait for the anti-spoof CNN (L5) as a second voter | principled | RGB-only voter: no help in the dark |

**Resolved in v0.4.0 — option (a).** The honest single-PASS + 0-FAIL rule stands and is documented as such (`balanced` accepts on one liveness PASS when the others abstain — the normal IR-only night case); "two of two" is a soft ceiling, not a gate. Spoof-lockout defaults **off** (a lockout only creates denial-of-service; the password is always available). Option (c) — shadow every attempt for a true 2-voter rule — remains available later, gated on E4/E6 below.

## 6d. Remaining review items (2026-10-22)

| # | Item | Plan |
|---|---|---|
| D1 | **doctor → health unification** | `doctor` still has its own checks beside `health.collect()`; make doctor a renderer of `health` so the app and the CLI cannot disagree |
| E4 | **Shadow voter metric hardening** | the self-shadowing metric is uncalibrated against prints/phones; make it relative to the toggle bench and the face level, add replay cases |
| E5 | **Cross-spectrum IoU ≈ 0 → lockout** | when IR and RGB boxes don't overlap (misalignment, not a spoof) cross FAILs as `spoof:*`; repeated, that trips the spoof lockout. Treat iou ≈ 0 as ABSTAIN/misalignment, re-align |
| E6 | **shadow_pair off-frame check** | verify the "off" frame is actually dark (LED switched) before comparing, else ABSTAIN instead of voting on two lit frames |
| C1 | **GREY fourcc / `is_ir` heuristic** | IR detection by fourcc/greyness misclassifies some cameras (GREY RGB, coloured IR); prefer USB interface + descriptor facts, keep the heuristic as fallback |

## 6e. Status log — v0.4.0 → launch

- **2026-10-25 (v0.4.0)** — **R4 done**: resident `ward.service` (`ward/daemon.py`) + `ward-auth` client (`ward/client.py`), zero-capability systemd sandbox (`systemd-analyze security` 0.9), `Type=notify` watchdog, ~195–203 MB flat RSS over 200 replayed attempts. **U1/U2/U3 done**: the manager app (`ward`) runs as the user with every privileged action via `pkexec`/polkit (no GTK as root), pages Overview/People/Reliability/Settings/Advanced, and a full setup wizard drawn in-window (register/calibrate stream their UI into the window). **L5 done**: MiniFASNetV2 anti-spoof CNN bundled and active as a daylight voter. **Night-unlock root cause fixed** (RGB open in a dark room starves the IR stream → ALS picks IR below 5 lux, a watchdog reopens IR; IR-only dark sits 0.31 from daylight models). Wider-angle honest enrol (yaw ±28°, flat arrival band). Accuracy **benchmarks** (`ward benchmark {lfw,spoof,ir,redteam}`). **Security audit**: PAM-helper `clearenv()` + `python3 -I`, fail-open unseal with a kill/reap bound, sticky liveness FAIL, pkexec grant not remembered. Distro packaging (.deb/RPM/AUR) + full CI. 285 tests.

- **2026-10-28 (v0.5.0)** — launch-readiness and correctness: fixed a **train/serve embed skew** (enrol embedded a raw crop, unlock a brightness-normalised one — one shared `brightness.embed_rgb` now; re-enrol affected models); `setup` installs **ArcFace when dlib is absent** (Debian/Ubuntu/Fedora ship no `python3-dlib` and pip is banned); the **keyring unseal is scoped to login/unlock services** only (never sudo/su/polkit); `pam enable` **refuses to bypass a 2FA module**; the liveness accept/reject core extracted to `Engine._accept_votes` with a unit-test table; YuNet warm-up folded into `Models.preload()`; new `distros` CI. Install docs lead with the live package channels.

- **2026-10-29 (launch)** — **published**: `.deb` on the GitHub release, **Ubuntu PPA** `ppa:0ctave/ward` (noble), **Fedora COPR** `0ctave/ward`; **AUR** primed, waiting on account registration reopening. Release & changelog tooling (`tools/release.py`, conflict-free `changelog.d/` fragments), auto-publish workflow on a `v*` tag, PPA lint fixes + COPR `make_srpm` (`.copr/Makefile`). Pinned release Discussion; Show HN / per-distro launch posts. 319 tests.

## 7. Milestones

**Shipped**

| Version | Date | Shipped |
|---|---|---|
| **0.2.0** | 2026-08-31 | Own engine replacing Howdy — dual-spectrum recognition, multi-frame fusion, liveness voters, free-move enrol with a personal 3D model, whitening, the lab bench |
| **0.3.0** | 2026-09-09 | *Fast & visible, and yours:* parallel camera open + early reject (R1–R3, R7), `ward log` + doctor opinions + desktop file (U5–U7), one guided enrol-and-calibrate, identities & conditions, pluggable recogniser incl. ArcFace, native `pam_ward.so` + keyring unlock (K0–K2), policy presets + lockout, replay harness (L7), the app (CLI-driven), compatibility matrix |
| **0.4.0** | 2026-10-25 | *Reliable & resident:* the **resident daemon** (R4, sandbox + watchdog), the **graphical manager app + in-window wizard** (U1/U2/U3 via pkexec), **anti-spoof CNN** active (L5), night-unlock fix, wider-angle honest enrol, accuracy **benchmarks**, a security audit, distro packaging + CI |
| **0.5.0** | 2026-10-28 | *Launch-ready:* train/serve embed fix, setup steers around missing dlib, PAM 2FA/keyring-scope hardening, install docs lead with packages, **PPA/COPR/.deb live**, release & changelog tooling |

**Remaining** (themes re-pointed at what is genuinely left; keyring and the daemon shipped earlier than the original plan expected)

| Version | Theme | Contents | Exit criterion |
|---|---|---|---|
| **0.6** | everywhere it installs | **AUR published** · pam-auth-update (Debian) + authselect + SELinux (Fedora) profiles (§4.1) · greeter matrix verified on Plasma 6 / SDDM / swaylock (§4.2) · NixOS module | installs **and authenticates** from a package on four distros; §4.2 has no "to verify" row |
| **0.7** | measured accuracy & snappier | **pre-warm on lock** (R5) · OpenCV-only **SFace** engine measured vs dlib (§4.4) · a real **spoof corpus** → `doctor --spoof`, FAIL_P from data (L5 tail) · ArcFace impostor `ARCFACE_RADIUS` with a second subject · `ward report` (anon hw/calibration → module DB) | accept < 500 ms at the lock screen; lab FRR < 2 % day+night, spoof corpus 0 accepts |
| **1.0** | done | **L6** emitter-differential imaging · a full docs pass · the security policy published as a standalone page | six months of use without a lockout or a false accept on the lab corpus |

Each milestone ends with the three checks that already exist: tests green, `ward doctor` clean, `ward lab` summary attached to the release notes.

---

## 8. How progress is measured

- **Latency:** `ward log --summary` median and p90 per PAM service; target column in §1.
- **Accuracy:** `ward lab` recordings per subject (you day/night/glasses, a friend, a phone photo, a print) → accept rate per policy; the CSVs are the regression corpus of L7.
- **Coverage:** the greeter matrix (§4.2) as a checklist in the release notes; each row verified in a VM or on hardware, date noted.
- **Safety:** zero lockouts by construction (the password path is never touched); every failure reason visible in the log and in the Attempts page.

Sources consulted for §4 and §5: [kscreenlocker !318 (flexible authenticators)](https://invent.kde.org/plasma/kscreenlocker/-/merge_requests/318), [Howdy + SDDM, Arch forums](https://bbs.archlinux.org/viewtopic.php?id=309702), [python3-dlib on Fedora](https://packages.fedoraproject.org/pkgs/dlib/python3-dlib/), [GNOME Keyring after non-password PAM auth](https://discourse.gnome.org/t/unlocking-gnome-keyring-after-non-password-pam-authentication/36030), [GNOME/Keyring on ArchWiki](https://wiki.archlinux.org/title/GNOME/Keyring).
