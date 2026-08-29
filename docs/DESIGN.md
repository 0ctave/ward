# faceup v0.2 — Design

Status: **implementing (v0.2.x)** · updated 2026-08-29 · supersedes the Howdy-wrapper scope of v0.1

## 1. Goal

Face authentication for Linux (login, `sudo`, polkit) that is **reliable at every head angle and in every lighting condition a laptop user actually encounters**, and **resistant to the spoofs a 2D RGB+IR camera can physically detect** — without ever locking the user out: every failure falls through to the password.

### Non-goals

- Matching a structured-light / ToF sensor (Face ID). Without depth hardware there is a residual attack (§9) and the docs say so.
- Replacing PAM, GDM, or the password. faceup is one `sufficient` step before them.
- Supporting cameras the kernel does not expose as V4L2 devices.
- Cloud, network, telemetry. The engine never opens a socket.

### Design principles

1. **Fail open to the password.** No check may veto authentication on its own except the recogniser itself. Liveness checks *vote*; abstaining is always allowed.
2. **Reliability comes from the recogniser and enrollment; security comes from layered liveness.** These are tuned independently and must not be traded against each other implicitly.
3. **Distro Python, zero pip dependencies.** The engine runs under `/usr/bin/python3` with `python3-dlib`, `python3-opencv`, `dlib-models`. Optional models (anti-spoof CNN) are ONNX files loaded through OpenCV's DNN module.
4. **Every decision is explainable.** Each attempt logs one line naming which stage decided the outcome (`no-face`, `dark`, `match`, `spoof:cross-spectrum`, …). Tuning is evidence-based.
5. **Reversible.** Enabling is one PAM line; disabling restores the previous state byte-for-byte.

## 2. Threat model

| Attack | In scope | Defence (§7) |
|---|---|---|
| Photo/video of the user on a phone, tablet or monitor | yes | cross-spectrum (screens emit ~no IR), anti-spoof CNN, motion (planar) |
| Printed colour photo | yes | anti-spoof CNN (print texture), cross-spectrum (print looks wrong in IR), motion (planar), shadow test |
| Printed *IR* photo (grey print tuned to the IR sensor) | yes | cross-spectrum (wrong in RGB), shadow test (flat under near light), motion (planar) |
| Two aligned prints (one per spectrum) | partial | shadow test, motion; documented residual risk |
| 3D mask, silicone replica | no | out of scope for 2D sensing |
| Look-alike / sibling | partial | dual-spectrum embedding match; `certainty` tuning; documented |
| Attacker with root, or tampering with models/config | no | root-only files (0600), but a root attacker has already won |
| Frame injection via a fake V4L2 device (v4l2loopback) | yes | engine opens only devices bound to a physical USB/PCI parent; refuses loopback drivers |
| Denial of service (hold the camera open) | yes | hard timeout → password fallback; never blocks login |

## 3. Hardware profiles

Detected at setup and at every attempt (cheap V4L2 ioctls). The engine adapts; nothing is hard-coded to a laptop.

| Profile | Cameras | Recognition | Liveness available |
|---|---|---|---|
| **A: IR + RGB** (Windows-Hello laptops) | GREY stream + MJPG/YUYV stream, same USB device | IR primary, RGB secondary | cross-spectrum, CNN, shadow, motion |
| **B: IR only** | GREY stream | IR | shadow, motion |
| **C: RGB only** | colour stream | RGB (needs ambient light) | CNN, motion |

IR detection rule: a capture node whose format set is a subset of `{GREY, Y8, Y10, Y12, Y16}`. Emitter presence is not detectable a priori; §7.3 measures it.

Cross-spectrum calibration (profile A): the two lenses are a fixed few millimetres apart. During enrollment the engine records the mean offset and scale between the IR and RGB face boxes (`calib.dx, dy, s`). At auth time boxes must agree within a tolerance after applying the calibration (§7.1).

## 4. Architecture

```
 PAM (pam_exec.so)  ──▶  faceup-auth  ──▶  exit 0 = authenticated
                          │
   ┌──────────────────────┼───────────────────────┐
   │ capture   → sync frame pairs (IR, RGB)        │
   │ quality   → brightness / sharpness gate       │
   │ detect    → HOG (default) | CNN (opt-in)      │
   │ track     → per-camera face track + landmarks │
   │ recognise → embeddings, multi-frame fusion,   │
   │             match vs. model store             │
   │ liveness  → 4 voters, 2 required              │
   │ decide    → first frame-set that satisfies    │
   │             recognition ∧ liveness            │
   └───────────────────────────────────────────────┘
                          │
        /etc/faceup/config.toml   /etc/faceup/models/<user>.json
        /var/log/faceup.log       (optional) /var/lib/faceup/snapshots
```

Components (one Python module each, `faceup/engine/`):

- `capture.py` — opens the selected devices with OpenCV/V4L2, produces time-stamped frames; pairs frames whose timestamps differ by < 40 ms.
- `quality.py` — brightness (mean, p5/p95), sharpness (variance of Laplacian), face size; drops frames below thresholds *before* spending CPU on detection.
- `detect.py` — dlib HOG by default; dlib CNN (`mmod_human_face_detector.dat`) when `detector = "cnn"`; 5-point landmarks for alignment, 68-point landmarks for motion analysis (loaded lazily).
- `recognise.py` — dlib ResNet embeddings; fusion; matching against the store.
- `liveness/` — `cross_spectrum.py`, `antispoof_cnn.py`, `shadow.py`, `motion.py`; a common `Vote` result: `PASS | FAIL | ABSTAIN` + reason + metric.
- `decide.py` — the decision rule (§6.5), timing budget, log line.
- `store.py` — model file read/write, versioning, permissions.
- `pam.py` (CLI side) — enable/disable the PAM line with a backup.
- `emitter.py` — wraps `linux-enable-ir-emitter`: status, `run`, and the UVC instruction it found (for §7.3).

The v0.1 CLI (`doctor`, `enroll`, `tune`, `bench`, `emitter`) is kept and extended; `setup` becomes the installer of the whole pipeline.

## 5. Data

### 5.1 Config — `/etc/faceup/config.toml` (root:root 0644)

```toml
[cameras]
ir  = "/dev/video2"        # "" = none
rgb = "/dev/video0"
ir_dark_threshold  = 25    # mean brightness below → frame "dark"
rgb_dark_threshold = 40

[recognition]
certainty        = 0.40    # max embedding distance (dlib scale, 0.6 = "same person" in dlib docs; 0.40 is strict)
rgb_certainty    = 0.38    # RGB is a weaker signal in poor light → slightly stricter
fusion_frames    = 4       # embeddings averaged per camera before matching
require_rgb_match = false  # if true, both spectra must match (profile A, bright rooms only)
detector         = "hog"   # "hog" | "cnn"
max_height       = 320

[liveness]
required_votes = 2         # of the enabled voters that did not abstain… see §6.5
cross_spectrum = true
antispoof_cnn  = true      # needs the ONNX model file
shadow_test    = "auto"    # "auto" = enabled if emitter toggling is available
motion         = true
motion_challenge = false   # active "turn your head slightly" prompt

[timing]
settle_ms   = 1200         # wait for the camera's auto-exposure to stop drifting (max)
window_ms   = 2000         # capture window, starts after settle
hard_timeout_ms = 4000     # absolute cap incl. settle and model loading

[behaviour]
abort_if_ssh    = true
abort_if_no_tty = false    # GDM has no tty; keep false
lockout_failures = 0       # 0 = never lock out (password still available anyway)
log_images      = false    # save failed frames to /var/lib/faceup/snapshots (opt-in, 0700)
```

### 5.2 Model store — `/etc/faceup/models/<user>.json` (root:root 0600)

```json
{
  "version": 2,
  "user": "octave",
  "created": "2026-08-28T20:41:00+02:00",
  "recogniser": "dlib_face_recognition_resnet_model_v1",
  "calib": { "dx": 0.012, "dy": -0.004, "s": 1.02 },
  "models": [
    { "id": 1, "label": "straight", "spectrum": "ir",  "embedding": [128 floats], "quality": 0.91, "time": "…" },
    { "id": 2, "label": "straight", "spectrum": "rgb", "embedding": [128 floats], "quality": 0.88, "time": "…" }
  ]
}
```

Embeddings are the only biometric data stored; no images are kept unless `log_images = true`. `version` gates migrations; a store written by an unknown recogniser is refused, not misused.

## 6. Authentication pipeline

Budget on a Ryzen 6600HS (no AVX dlib build): **typical accept in 250–450 ms**, hard cap 2.5 s.

### 6.1 Preconditions (≈ 5 ms)
- Refuse if `abort_if_ssh` and `SSH_CONNECTION` is set; refuse if the user has no models. Exit non-zero → PAM continues to the password. No log spam for these.
- Open cameras from config; if a device is missing, degrade to the available profile (§3). Refuse devices whose driver is a loopback (`v4l2loopback`, `vivid`).

### 6.2 Settle, then capture window (≤ 1200 + 2000 ms)
Laptop IR cameras run auto-exposure in firmware with no manual control and start each session over- or under-exposed; the face is undetectable until it converges (~1–2 s measured). The engine reads frames until the primary camera's brightness is stable for 4 frames (max `settle_ms`), and only then starts the window. dlib models are loaded before the camera opens so that cost overlaps nothing. Both cameras are read in lock-step (20 pairs/s measured); the window ends early on success.

### 6.3 Quality gate (≈ 1 ms/frame)
Drop frames that are dark (below threshold), blown out (p95 > 250), blurred (Laplacian variance < 60 at 320 px), or where the face box would be < 60 px. A window with *only* dark IR frames ends with reason `dark` — this is the emitter-failure signature and `doctor` reports it.

### 6.4 Detect · track · fuse (≈ 60 ms/frame HOG, 700 ms CNN@240)
- Detect on the best frame first, then track (correlation tracker) — detection is not re-run every frame.
- Align with 5-point landmarks, compute the 128-d embedding per frame per camera.
- **Fusion:** keep the top `fusion_frames` by quality, average the embeddings (L2-normalised). Averaging removes per-frame noise — the largest single reducer of false rejects after multi-pose enrollment.

### 6.5 Decision rule
```
recognised_ir  = min_dist(fused_ir,  store.ir)  ≤ certainty
recognised_rgb = min_dist(fused_rgb, store.rgb) ≤ rgb_certainty
recognised     = recognised_ir  (profile A/B)  |  recognised_rgb (profile C)
                 ∧ (recognised_rgb if require_rgb_match)

votes = [v.result for v in enabled_voters]          # PASS / FAIL / ABSTAIN
pass_count = count(PASS);  fail_count = count(FAIL)
live = pass_count ≥ required_votes ∧ fail_count == 0
       or (pass_count ≥ 1 ∧ fail_count == 0 ∧ all_others_abstained)   # degraded profiles

accept = recognised ∧ live
```
- A **FAIL** vote blocks acceptance for that frame-set only; the window continues (a spoof detector can misfire on one frame). A window that ends with any FAIL logs `spoof:<voter>`.
- ABSTAIN never counts against the user. This is how a still user, a dark room, or a missing optional model degrade gracefully.
- `required_votes` is automatically capped at the number of *available* voters for the profile.

### 6.6 Result
Exit 0 → PAM `sufficient` succeeds. Any other outcome → exit 1, one log line:
`2026-08-28T21:02:11 user=octave result=accept ir=0.31 rgb=0.36 votes=cross:PASS cnn:PASS shadow:ABSTAIN motion:PASS t=380ms`

## 7. Liveness voters

Each returns `PASS`, `FAIL`, or `ABSTAIN` with a numeric metric; thresholds are config keys with the defaults below.

### 7.1 Cross-spectrum consistency (profile A) — deterministic, ~0 ms
Requires a face in the IR frame **and** the paired RGB frame, and their boxes to agree after calibration: `IoU(box_ir, calib(box_rgb)) ≥ 0.5`.
- Screen replay: face in RGB, none in IR → **FAIL**.
- IR-tuned print: face in IR, RGB sees flat grey → **FAIL**.
- Dark room where RGB legitimately sees nothing: IR mean ok, RGB mean < `rgb_dark_threshold` → **ABSTAIN** (not FAIL). This distinction is what keeps night-time unlocks working.

### 7.2 Anti-spoof CNN (RGB) — ~15 ms
Silent-Face-Anti-Spoofing (MiniFASNet v2, MIT) exported to ONNX, run with `cv2.dnn`. Input: the RGB face crop with the model's expected margin. Output: real / print / replay probabilities.
- `p_real ≥ 0.7` → PASS; `p_real ≤ 0.3` → FAIL; between → ABSTAIN.
- Frame too dark or crop < 80 px → ABSTAIN.
- Model file absent → voter disabled (documented in `doctor`). The model is downloaded by `faceup setup` from a pinned release with a checksum; never at auth time.

### 7.3 Emitter-modulation shadow test (IR) — ~2 frames
`linux-enable-ir-emitter` discovers the UVC extension-unit instruction that switches the LEDs. If that instruction can be applied at runtime in < 50 ms (measured once in `setup`, stored as `shadow_test = true/false`), the engine captures one frame **emitter off** and one **on**, aligned by landmarks, and computes:
- `Δ = mean(on − off)` inside the face: must be large (the face is near the LEDs) — a distant screen barely changes.
- **Gradient ratio**: on a real face the illumination change is strongest on the nose/brow and weakest under them (self-shadowing); on a flat print the change is spatially uniform. Metric: coefficient of variation of Δ over face regions; `≥ 0.25` → PASS, `≤ 0.10` → FAIL, else ABSTAIN.
- Any toggling failure → ABSTAIN and the voter is marked unavailable for the session.

### 7.4 Non-rigid motion (any profile) — reuses tracked landmarks, ~1 ms
Over the window, take the 68-point landmarks of the first and last good frames (and mid-window pairs). Fit a homography `H` (RANSAC) to the point correspondences and measure the residual: a flat object's landmarks are explained by `H` almost exactly; a real head's are not (nose vs ears parallax, jaw/mouth deformation).
- Mean displacement < 1.5 px → **ABSTAIN** (no motion to judge; the user is still).
- Displacement ≥ 1.5 px and residual RMS ≥ 0.8 px → PASS.
- Displacement ≥ 1.5 px and residual RMS ≤ 0.3 px → FAIL (moved like a plane).
- Blink detected (eye-aspect-ratio dip) → an extra PASS-weight, never required.

### 7.5 Motion challenge (opt-in)
`motion_challenge = true` shows "turn your head slightly" (console or GDM message via PAM conv) and requires §7.4 to PASS within the window. Off by default — it costs the user a second every time.

## 8. Enrollment

### 8.1 Free-move enrollment (default)

A camera window tracks the head and the user simply moves it around, as with Face ID:

1. **Head pose per frame.** dlib's 68 landmarks → `cv2.solvePnP` against a generic 3D face
   (nose tip, chin, eye corners, mouth corners) → yaw / pitch / roll, exponentially smoothed.
   Verified on synthetic projections to ±4°.
2. **Pose grid.** 5 yaw cells (far-left … far-right, edges ±9/±24/±45°) × 3 pitch cells
   (down / level / up, edges −13/+9°). The seven cells on the centre cross are **required**
   (that is where a face sits at a laptop, including "looking at the keyboard"); the eight
   corners are optional.
3. **Sampling.** When the pose is stable (< 4° change between frames), the frame passes the
   quality gate and the face is ≥ 80 px, an embedding is computed for every camera and
   pushed into the current cell (best 3 per cell per spectrum by quality). A cell counts
   as filled after 2 samples of the primary spectrum.
4. **Guidance.** The window shows the mirrored camera image, the grid with filled / required
   / optional cells, a dot for the current pose, the next target cell, and a text hint
   ("turn a little left, look a little down"). Because guidance is *visual* — move the dot
   into the cell — it does not depend on getting the angle sign conventions right for a
   given camera mounting.
5. **Finish.** Automatically when all 15 cells are filled, 12 s after the required ones are,
   or on Enter. Per cell and spectrum the samples are fused into one model with its mean
   pose; cells are labelled `C/level`, `L1/down`, …

### 8.2 Interpolation between poses

The store synthesises a **virtual model** for every pair of grid-adjacent filled cells: the
normalised mean of their embeddings. dlib's embedding space is close enough to locally
linear that the midpoint approximates the face at the intermediate angle. Virtual models
take part in matching but are never persisted. This is what lets a 7-cell enrollment cover
the angles between cells without the user holding them.

### 8.3 Guided enrollment (`--guided`, fallback)

The scripted eight-pose text flow of v0.2, used when no window can be opened (headless, no
display for root). Models carry no pose and get no interpolation.

### 8.4 Common to both

- Calibration first (10 synchronized pairs) → `calib`.
- Wrong-person rejection: an embedding > 0.6 from every existing model of that spectrum.
- Self-check at the end: three full authentication attempts with per-voter results.
- Re-enroll when the camera assignment changes.

### 8.5 Emitter check

`faceup emitter check` opens the same window with IR and RGB side by side and infers the
emitter state without any hardware knowledge: with the room lights **off** the colour camera
must be dark; if the IR camera still sees a face (or is bright), the emitter is firing;
if both are dark, it is not. `doctor` runs the same inference silently whenever the RGB
frame happens to be dark.

## 9. Security considerations

- **PAM placement:** `auth sufficient pam_exec.so quiet stdout /usr/local/bin/faceup-auth` inserted *before* `pam_unix` in `common-auth` (or the distro equivalent), with the original file backed up. `pam_exec` runs the engine as root with `PAM_USER` set; the engine authenticates only that user.
- **Files:** models 0600 root, config 0644 root, engine 0755 root-owned, log 0640 root:adm. Snapshots off by default.
- **Residual risk (documented):** two high-quality prints aligned for both lenses, plus a way to fake near-field shadowing, could defeat a profile-A setup. This is the same order of difficulty as attacking Windows Hello on the same hardware. faceup is convenience-grade with strong anti-casual-spoofing, not a hardware security module.
- **Lockout:** none by default — the password is always available, so lockout only creates denial-of-service. Optional `lockout_failures` for users who want a cooling-off after N spoof detections.
- **No timing side channels of note:** results are all-or-nothing and the password path is unchanged.
- **Coexistence with Howdy:** `setup` offers to disable `pam_howdy` (two face steps in a row is pointless) and can *import* nothing — dlib embeddings from RGB frames don't match IR; re-enroll.

## 10. Degraded modes

| Situation | Behaviour |
|---|---|
| Emitter dead, dark room | IR frames dark → window ends `dark`; RGB also dark → password. `doctor` flags the emitter. |
| Emitter dead, lit room | IR sees you by ambient light; works. Shadow test ABSTAINs (no toggle). |
| RGB covered (privacy shutter) | cross-spectrum ABSTAINs; CNN ABSTAINs; liveness relies on shadow + motion. If both abstain → degraded rule accepts on recognition alone *only if* `require_liveness_when_available = false` (default true → password). |
| Anti-spoof model missing | voter disabled; `doctor` says so. |
| Camera busy (video call) | open fails → immediate password fallback, one log line. |
| CPU under heavy load | hard timeout → password. Never an indefinite hang. |

## 11. CLI (v0.2)

```
faceup doctor                      diagnose everything, incl. a spoof self-test (--spoof: hold your phone up)
sudo faceup setup                  cameras → emitter → anti-spoof model → PAM → enrollment → self-check
sudo faceup enroll [--fresh]       pose-guided, dual-spectrum
sudo faceup tune KEY VALUE         edits config.toml with validation
faceup bench                       detector / recogniser / voter timings on this CPU
sudo faceup emitter install|configure|test|toggle-bench
sudo faceup pam enable|disable     PAM line with backup/restore
faceup log [-n 20]                 recent attempts, one line each
sudo faceup-auth                   the engine (PAM calls this; also runnable by hand for debugging)
```

## 12. Performance budget (Ryzen 5 6600HS, distro dlib without AVX)

| Stage | Cost | Notes |
|---|---|---|
| Camera open + first frames | 80–150 ms | dominates the fast path; keep devices open for the whole window |
| Quality gate | 1 ms/frame | |
| HOG detect | 60 ms (once) then tracking ~5 ms/frame | CNN@240: 700 ms — opt-in |
| Landmarks + embedding | 30 ms/frame/camera | 4 fused frames × 2 cameras ≈ 240 ms spread over the window |
| Cross-spectrum, motion | < 2 ms | |
| Anti-spoof CNN | 15 ms | |
| Shadow test | 2 frames ≈ 100 ms | only if toggle is fast |
| **Typical accept** | **250–450 ms** | |

## 13. Testing

- **Unit:** recorded frame sequences under `tests/fixtures/` (real, phone replay, print, IR print, dark) — the pipeline must produce the expected reason for each. Fixtures are contributed frames, never embeddings of real users.
- **Voter tests:** synthetic homography motion vs jittered 3D motion for §7.4; synthetic uniform vs gradient Δ for §7.3.
- **Integration:** `faceup doctor --spoof` interactive self-test; `faceup-auth` exit codes under missing camera / busy camera / no models / timeout.
- **Bench:** `faceup bench` output committed for reference hardware in `docs/bench/`.

## 14. Roadmap after v0.2

- GTK4/libadwaita front-end for enrollment (live preview with pose guidance).
- Better recogniser as an option (ArcFace ONNX via `cv2.dnn`) once its CPU cost is measured — the store's `recogniser` field already allows it.
- Packaging: `.deb`, AUR, Fedora COPR.
- Optional Howdy-model import for RGB-only profiles.

## 15. Open questions

1. Can the emitter be toggled fast enough on common Windows-Hello modules for §7.3? Measure on the reference laptop first; keep the voter "auto".
2. Should `require_rgb_match` default to true in bright rooms (auto-switch by RGB brightness)? Adds security at zero reliability cost when light is good.
3. GDM message channel for the motion challenge — PAM conversation text is shown by GDM but not by all greeters.
