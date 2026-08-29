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

**Identification.** `/dev/videoN` numbers are assigned in enumeration order and change whenever the module re-enumerates (observed: RGB 0 → 1, IR 2 → 4 after a reconnect). The config therefore stores a **stable identity** — `usb:<vendor>:<product>:<interface>` read from sysfs (one physical module exposes its RGB and IR sensors as different USB interfaces, here `00` and `02`) — and every command resolves it to today's path at start (`cameras.resolve`). The pixel format is the role check: an id that resolves to a colour stream is refused as the IR camera. Legacy `/dev` specs are honoured while they exist and self-heal to the same-role camera with a warning; `auto` picks by role.

Generality rules (`cameras.py`, all unit-tested on synthetic device lists):
- **USB:** `usb:<vid>:<pid>:<iface>`, plus `:<serial>` when the device reports a real one (identical webcams are told apart by serial; devices with generic serials such as `01.00.00` by `@<port>`, which setup appends only when a twin is connected).
- **PCI / platform sensors** (MIPI cameras): `pci:<address>:<node name>` / `platform:<name>:<node name>` from sysfs.
- **IR role:** greyscale-only formats (GREY/Y8/Y10/Y12/Y16), or a greyscale format without MJPG, or a device name containing IR/infrared/depth. Modules that stream IR as YUYV: `cameras.force_roles = true` trusts the configured assignment.
- **Pair selection:** the RGB sensor on the *same physical module* as the IR sensor beats an external webcam of higher resolution.
- **Unusable nodes** are reported, not hidden: no permission (`video` group), and PCI/platform sensors that expose no V4L2 capture format (IPU6/rkisp behind a media controller — need libcamera/PipeWire, unsupported for now). Metadata nodes are filtered silently.
- **Capture:** MJPG is requested on colour cameras (YUYV falls to 5–10 fps above VGA on USB 2); any resolution and aspect ratio is accepted downstream.

Cross-spectrum calibration (profile A): the two lenses are a fixed few millimetres apart. During enrollment the engine records the mean offset and scale between the IR and RGB face boxes (`calib.dx, dy, s`). At auth time boxes must agree within a tolerance after applying the calibration (§7.1).

### 3a. Hardware profile and brightness management

**Profile** (`faceup/hw.py`, shown by `doctor`): CPU model, threads, AVX2; OpenCV version and
whether an OpenCL runtime is present (then `recognition.accel = "opencl"` runs the DNN models
on the GPU — opt-in, because for a 4 ms model the dispatch cost usually exceeds the gain);
dlib build flags; power profile (power-saver is reported: everything runs ~1.5–2× slower);
per-camera exposure capabilities. The engine sets OpenCV's thread count from it.

**Brightness** (`engine/brightness.py`), camera-agnostic, in three layers:

1. *Meter the face, not the frame.* Once a face is detected, its inner region is measured
   (mean, p5, p95) and classified `dark / dim / ok / bright / blown` per spectrum. Room
   light, windows and dark desks no longer decide whether a frame is usable; the frame-level
   gate only remains for frames with no face at all.
2. *Steer exposure when the camera allows it.* On open, each camera is probed for manual
   exposure (V4L2 via OpenCV: `AUTO_EXPOSURE`, `EXPOSURE`, `GAIN`) and put back on auto.
   When the face leaves the target band and auto-exposure cannot fix it, faceup takes over
   with bounded proportional steps (≤ 1.6× per step, ≤ 4 steps/s) and restores the original
   settings on close. Cameras that run auto-exposure in firmware and expose no controls (the
   reference IR module) are left alone. Reference laptop: IR = firmware auto only, RGB =
   manual control available.
3. *Normalise in software* what is still dim or blown out: CLAHE, plus a gamma lift for dark
   frames, applied to luminance only and deterministically from the frame's own mean — so
   enrollment and authentication apply the same transform. Identity for frames in range.

**Adaptive spectrum.** IR is preferred, but when its face is dark/unusable for three
consecutive sets (emitter off, out of LED range) while the RGB face is in range, RGB carries
that attempt; cross-spectrum abstains rather than fails when the other camera is dark.

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
- `detect.py` — **YuNet** (OpenCV DNN, bundled `face_detection_yunet_2022mar.onnx`, Apache-2.0) by default: ~4 ms per frame at 480 px and trained on turned/tilted faces; dlib HOG (`"hog"`, frontal only) and dlib CNN (`"cnn"`, ~700 ms) remain selectable. YuNet boxes are re-squared to HOG-style boxes before dlib's landmark predictors run. 5-point landmarks for alignment, 68-point for pose and motion (loaded lazily).
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
ir  = "usb:30c9:0057:02"   # stable USB id (vendor:product:interface); "auto"; or a /dev path; "" = none
rgb = "usb:30c9:0057:00"
ir_dark_threshold  = 25    # mean brightness below → frame "dark"
rgb_dark_threshold = 40

[recognition]
certainty        = 0.40    # max embedding distance (dlib scale, 0.6 = "same person" in dlib docs; 0.40 is strict)
rgb_certainty    = 0.38    # RGB is a weaker signal in poor light → slightly stricter
fusion_frames    = 4       # embeddings averaged per camera before matching
require_rgb_match = false  # if true, both spectra must match (profile A, bright rooms only)
detector         = "yunet" # "yunet" | "hog" | "cnn"
max_height       = 320

[liveness]
required_votes = 2         # of the enabled voters that did not abstain… see §6.5
cross_spectrum = true
antispoof_cnn  = true      # needs the ONNX model file
shadow_test    = "auto"    # "auto" = enabled if emitter toggling is available
motion         = true
motion_challenge = false   # active "turn your head slightly" prompt

[timing]
settle_ms   = 0            # >0: wait for auto-exposure before the window (needed HOG; YuNet copes)
window_ms   = 2500         # capture window
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

### 6.4 Detect · track · fuse (≈ 4 ms/frame YuNet; 60 ms HOG; 700 ms CNN@240)
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

1. **Head pose per frame — no PnP in the generic path.** A 6-point PnP against a generic head
   is dominated by the chin landmark, dlib's least accurate one, and couples position with
   rotation. Instead every angle comes from chin-free landmark geometry, closed-form against the
   generic 3D proportions: **pitch** from where the nose tip sits between the eye line and the
   mouth line (the nose protrudes ~135 mm, so pitch shifts its projection by ~0.75 % of the
   eye–mouth span per degree); **yaw** from where the nose tip sits horizontally between the
   eyes and between the mouth corners (two references averaged, ×1.2 gain for the orthographic
   formula's compression under perspective); **roll** from the eye line. All use averaged eye
   sets (12 points) and mouth points. Synthetic sweep over ±35° yaw × −20…+12° pitch: yaw
   1.6° rms (max 4°), pitch within 4°. Once the personal 3D model exists (8.1.2) a 13-point
   PnP against the user's own geometry replaces this. Angles pass through a **One Euro filter** (min_cutoff 0.6 Hz, β 0.05):
   ±1.5° landmark noise becomes ≤1.3° peak-to-peak when still, while a 30° turn is followed
   within 0.5 s — no wiggle, no lag.
1b. **Neutral calibration — in ratio space.** The first ~30 face frames (user looking at the
   screen naturally) define the neutral pose. Two effects are absorbed: the camera sits above
   the eyes (looking at the screen reads as pitch +10…+20°), and the generic proportions are
   not the user's (a long nose or a narrow eye line reads as a large constant angle — observed
   +35° pitch / +20° yaw on the reference user). Subtracting such offsets *after* the
   nonlinear ratio→angle inversion pushed the working range into the region where the
   inversion compresses, making the top rows unreachable. Instead the user's median landmark
   **ratios** at neutral become the zero point: each ratio is shifted so the user's neutral
   maps onto the generic model's neutral *before* inversion, keeping every user in the linear
   region. Cues: nose tip between the eyes, between the mouth corners, between the nose wings
   (yaw; averaged), nose tip between eye and mouth lines (pitch). Synthetic face with
   different proportions: neutral reads 0/0 after calibration; residual error is a per-user
   *gain* (a longer nose moves more per degree, up to ~25 % on the test face) that only the
   personal 3D model (8.1.2) can remove. Stored as `calib.neutral_ratios` / `calib.pose_offset`.
2. **Continuous coverage, not cells.** Every sample paints a Gaussian footprint (σ = 7°) on a
   2°-resolution (yaw, pitch) map. The **required region is an ellipse** around the neutral
   pose (semi-axes 34° yaw × 19° pitch, centred 4° below level because laptop use is
   keyboard-biased): far-left+up style corners are not required — physically awkward and
   covered by interpolation. Sampling happens wherever local coverage is below 1.0, so
   there are no hard cell boundaries to hit.
3. **Sampling.** When the pose is stable (< 4° change between frames), the frame passes the
   quality gate and the face is ≥ 60 px, an embedding is computed for every camera.
   Storage bins remain discrete — 7 yaw × 3 pitch cells (12° yaw bins, 18° centre; pitch
   down < −10°, level, up > +6°) — one fused model per bin and spectrum (best 3 samples by
   quality), labelled `C/level`, `L2/down`, …
4. **Guidance.** A GTK 4 / libadwaita window (`faceup/appui.py`, native Wayland, follows the
   system theme): the mirrored camera preview with a soft face outline (accent while
   tracking, green pulse on sample, red when dark); a coverage view — the required ellipse as
   a fixed outline, coverage as a smooth accent-coloured field, the current pose as an eased
   dot with a roll tick, the target as a pulsing ring; a guidance card with the phase and the
   hint ("slowly turn left toward the ring"); coverage % and face-light state; Cancel/Finish.
   The engine hands the window a *snapshot* (frame, box, pose, coverage, hint, phase…) per
   frame; an OpenCV window renders the same snapshot as a fallback where GTK is unavailable.
   **One coordinate convention for everything the user sees:** `gui.to_display()` converts
   the camera-convention pose to display space exactly once (display yaw = −camera yaw when
   the preview is mirrored, because camera yaw is negative when the user turns to *their*
   right); the coverage map, target, hints and view all live in display space and apply no
   sign of their own. (A second sign inside the view once inverted the dot.)
5. **Finish.** Automatically 3 s after the required ellipse is ≥ 97 % covered, or on Enter.
   Roll is displayed and used for de-rotation in the pitch measure but is not an enrollment
   axis: dlib aligns the face chip by the eyes, so embeddings are roll-invariant already.

### 8.1.1 Keeping the preview at camera rate

Measured on the reference laptop (both cameras are hardware-limited to **30 fps**; the IR
module offers 15 or 30, nothing above):

- `CAP_PROP_BUFFERSIZE = 1` halves the IR frame rate (29.7 → 15.3 fps). Never set it; a
  reader **thread per camera** keeps the newest frame instead, and the UI never blocks on the
  driver.
- dlib's Python bindings hold the GIL during detection and embedding, so a worker *thread*
  froze the preview (30 → 14 fps). Sampling runs in a separate **process** (spawn context;
  frames travel over a pipe, ~0.7 MB per sample). The UI loop then does only ROI
  tracking (~5 ms), 68 landmarks + PnP (~2.5 ms) and drawing, and holds 29–30 fps while
  samples are being embedded concurrently.
- Face detection tracks a region around the previous box without upsampling; the
  expensive 2× full-frame search runs only on every fourth miss.
- The window is **native Wayland**: GTK 4 (`faceup/display.py`), frames uploaded as
  `Gdk.MemoryTexture` and composited by the GPU, 0.4 ms per frame measured. The engine runs as
  root (models are root-owned), so the window connects to the *invoking user's* Wayland
  socket (`/run/user/<SUDO_UID>/wayland-0`), which sudo otherwise strips. OpenCV's Qt/X11
  HighGUI remains as a fallback when PyGObject or a session is unavailable.

### 8.1.2 Post-calibration: a personal 3D face model

The live estimator uses a *generic* head, so it carries a per-user bias (nose length,
eye–mouth spacing). Enrollment records the 68 landmarks of every processed frame (up to 900,
landmarks only — no images). When the window closes, a batch **structure-from-motion** solve
over the 13 rigid landmarks (eye corners, nose bridge/base, mouth corners — no jaw, chin,
lids, lips or brows) recovers the user's own 3D geometry and the true rotation of every
frame: Tomasi–Kanade factorization under scaled orthography → metric upgrade (least-squares
on the rotation-row constraints) → similarity Procrustes alignment to the generic model's
frame on the shared landmarks. Orthography cannot tell a face from its mirror image, and a
mirrored model makes PnP return yaw with the wrong sign (observed on the reference user:
"rms 36°" against the fast estimator), so after a first alignment that may include a
reflection the chirality is checked on anatomy — left eye corner x < right, nose tip nearer
the camera than the eye corners — flipped if needed, and the model re-aligned with a proper
rotation. `engine/sfm.py`; needs ≥ 40 frames.

The model (13 × 3 mm) is stored as `calib.shape3d` and from then on the live estimator runs a
13-point PnP against the user's own face — all three angles, ~1 ms. A report compares the
fast generic estimator to the personal model over the recorded frames — as centred series,
since the fast estimator is relative to the user's neutral and PnP is absolute: the slope
(gain, 1.0 = same scale) and the residual after a linear fit, per angle — printed after
enrollment; `faceup calibrate` re-runs the solve from the saved
tracks (`<user>.tracks.npz`). Synthetic check (face with individual proportions, random poses ±35° yaw / −22…+18° pitch,
1 px landmark noise, 150 frames): fit residual 1.9 px, recovered poses within 2.8° rms for
both yaw and pitch, pitch bias −1.1°.

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
| YuNet detect | 4 ms/frame (full frame, no tracking needed) | HOG 60 ms / 5 ms ROI; CNN@240 700 ms — opt-in |
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
