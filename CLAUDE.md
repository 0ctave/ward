# faceup — session handout

Reliable face-unlock for Linux. This file seeds a fresh session: read it, then `git log --oneline | head`, then `python3 -m unittest discover -s tests` (must be green) before changing anything.

## What it is

A face-authentication stack for login / `sudo` / polkit that **replaces Howdy** (Howdy is still installed but its job is done). It has its **own engine** — `faceup-auth`, called by PAM via `pam_exec` — built on dlib (ResNet embedding), OpenCV (YuNet detector), V4L2. Repo: `~/dev/faceup`, MIT, v0.2.0. Reference machine: Lenovo Yoga Slim 7 ProX 14ARH7 — Ryzen 5 6600HS, RGB `/dev/video*` iface 00 + IR iface 02 (Windows-Hello module), GNOME 46 Wayland.

## Golden rules (do not break these)

1. **Distro Python only.** Runs under `/usr/bin/python3` with `python3-dlib`, `python3-opencv`, `dlib-models`. **Zero pip dependencies.** A venv can't see dlib/opencv. Never add a `requirements.txt`/pip install step.
2. **Tests must stay green** and need no camera: `python3 -m unittest discover -s tests`. Env overrides for unprivileged testing: `FACEUP_ETC`, `FACEUP_LOG`, `FACEUP_PAM_FILE`, `FACEUP_LAB`.
3. **Fail open.** Every auth failure (dark/busy camera, no match, spoof, timeout) must fall through to the password. A bug must never lock the user out.
4. **Two modules named ui:** `faceup/ui.py` = terminal output helpers (ok/warn/banner…) used by every CLI command. `faceup/appui.py` = the GTK4/libadwaita windows. Never merge them (a past agent overwrote ui.py and broke every command — there's a regression test).
5. **One display-space convention:** `gui.to_display()` converts a camera-convention pose to what the user sees exactly once (display yaw = −camera yaw when the preview is mirrored). The coverage map, target, hints and the coverage view are all display-space and apply no sign of their own. Applying a sign twice inverted the dot once.
6. sudo can't prompt in these shells (no tty) — the user runs sudo commands themselves.

## Architecture (`faceup/`)

- `cli.py` — commands: `doctor setup enroll auth tune pam log models bench emitter lab calibrate`.
- `engine/auth.py` — the pipeline: capture → quality gate → detect → embed → **multi-frame fusion** → recognise → **liveness voters** → decide. `Engine.authenticate(user)`. Adaptive spectrum (IR primary, RGB carries if IR face is dark).
- `engine/capture.py` — threaded reader per camera (never set `CAP_PROP_BUFFERSIZE=1` — it halves IR fps); face metering + exposure steering hook. MJPG on RGB.
- `engine/detect.py` — **YuNet** (bundled `models/face_detection_yunet_2022mar.onnx`, the 2022 model because the 2023 needs OpenCV≥4.8 and Ubuntu has 4.6) as default detector (~4 ms, pose-robust); HOG/CNN selectable. dlib landmarks + embedding. `DLIB_DIRS` = where the shared dlib models live (`/usr/share/dlib`).
- `engine/pose.py` — head pose with **no PnP in the generic path**: chin-free ratio estimators (pitch = nose between eye/mouth lines; yaw = nose between eyes/mouth/wings ×1.2; roll = eye line), **ratio-space neutral calibration** (per-user zero point before inversion). One Euro filter on landmarks, windowed `Stability`. `Coverage` = continuous (yaw,pitch) map; required ellipse **fixed by default** (`ADAPTIVE=False`, axes 30×14). Personal 13-point PnP once `sfm` builds the model.
- `engine/sfm.py` — structure-from-motion over enrollment landmarks → the user's own 3D face model (`calib.shape3d`); Tomasi-Kanade + metric upgrade + **chirality repair** (a mirrored model flips yaw sign — the "rms 36°" bug).
- `engine/store.py` — `/etc/faceup/models/<user>.json` (0600, embeddings only), **v4**: per-model samples + **within-class whitening (WCCN)** for lighting invariance (2nd acceptance path under a hard raw cap). `fit_whitening`, `wdist`.
- `engine/brightness.py` — face metering (not frame), exposure steering when the camera allows it, CLAHE/gamma normalisation.
- `engine/liveness/` — `cross_spectrum`, `motion` (implemented); `antispoof_cnn` (needs a MiniFASNet ONNX, not bundled → disabled); `shadow` (metric done, toggle hook off). PASS/FAIL/ABSTAIN, 2 PASS + 0 FAIL needed, ABSTAIN never counts against.
- `engine/pam.py` — PAM line uses **`seteuid`** (pam_exec runs as the caller's real uid — for sudo that's the user, who can't read the root-only store; seteuid runs it as root). `enable` upgrades an old line in place.
- `engine/lab.py` + `appui.LabWindow` — live evaluator comparing 3 unlock policies, subject recording to CSV, per-subject summary. This is the reliability test bench.
- `appui.py` — GTK4 windows (Enroll, Emitter, Lab) on `_Shell`; snapshot-dict interface; `display.py` picks GTK (native Wayland, `kind=`) or OpenCV fallback. Root Wayland: uses the invoking user's socket, `GTK_A11Y=none`.
- `gui.py` — the enrollment step function (builds snapshots), the spawn-process sampler (`_Sampler` — dlib holds the GIL, so embeddings run in a separate PROCESS; `__main__` is guarded).
- `hw.py` — CPU/AVX2/OpenCL/power profile; sets OpenCV threads.

Docs of record: `docs/DESIGN.md` (full rationale, kept in sync), `README.md` (the "why face unlock fails on Linux" guide).

## Install / run

```
cd ~/dev/faceup && sudo ./install.sh          # /usr/local/lib/faceup + /usr/local/bin/{faceup,faceup-auth}
faceup doctor                                  # diagnoses everything; run first
sudo faceup enroll --fresh                     # GTK window; move head, fill the ellipse
sudo faceup lab                                # reliability bench (record subjects, compare policies)
sudo faceup pam enable                         # wire into sudo/login (seteuid); offers to retire howdy
faceup log -n 5                                # every attempt, raw+whitened distances, votes
```

## Known-good state (2026-08-31)

- Self-check passes: IR ~0.12–0.35, RGB ~0.15–0.29, well inside the 0.40 gate. Cross-spectrum PASS, motion ABSTAIN when still.
- 46 tests green at HEAD `135f45f`.
- **The IR emitter is NOT configured** — in a dark room the IR frame is black and detection fails; in daylight it's fine. `sudo faceup emitter check` (lights off) then `sudo faceup emitter configure`. This is the #1 real-world reliability gap.
- **Whitening/online-learning need a fresh enroll** to populate per-model samples (store v4).
- PAM state varies — `faceup doctor` reports whether `faceup-auth` is currently in `/etc/pam.d/common-auth` (it was enabled with seteuid, but verify; Howdy's own line may still be there and failing on the old camera path).

## Open tasks (roughly in priority order)

1. **Emitter shadow-test liveness** — measure LED toggle speed (`emitter toggle-bench`), wire `liveness/shadow.py`'s hook. Until a 2nd reliable voter exists, the "2 PASS" rule leans on cross-spectrum alone.
2. **Anti-spoof CNN** — bundle a MiniFASNet ONNX (pinned + checksum, like YuNet) so `liveness/antispoof_cnn.py` activates; then `doctor --spoof` self-test (hold up a phone photo → reject).
3. **Emitter-differential imaging** — `I_on − I_off` cancels daylight; the principled lighting fix (needs the toggle from #1).
4. **Publish** — GitHub repo, replace `<you>` in `pyproject.toml`/README, push.
5. Nice-to-haves: `LiveEvaluator.summary()` per-subject face fraction; Adw≥1.6 `accent_color_rgba` instead of the hidden-label hack.

## History gotchas (so they aren't re-litigated)

- Camera `/dev/videoN` numbers change on reconnect → config stores stable `usb:vid:pid:iface` ids, resolved at runtime (`cameras.resolve`).
- Tan–Triggs photometric normalisation HURTS dlib (network never trained on it) — don't reach for it; whitening is the answer.
- pitch was chin-dominated via PnP → replaced by the nose-ratio method.
- coverage % stuck at 0 once because the map-update line was dropped in a refactor.
- The user prefers open-source, proven, native-Wayland, hassle-free. Design doc altitude: explain the mechanism, keep it honest about limits (2D sensor, not Face ID).
