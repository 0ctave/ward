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
antispoof_cnn  = true      # bundled MiniFASNetV2; `faceup antispoof off` to disable
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

**Emitter warm-up (2026-09-08).** The IR LEDs take ~400 ms to reach full brightness after the control is set (measured on the reference module). `authenticate` therefore turns the emitter **on first, before the cameras open** (its control fd on the IR node opens without blocking the streaming open), so the ramp overlaps the ~600 ms camera-open + model-load window and recognition starts on a lit face — instead of the first ~800 ms seeing a dark IR frame (the "LEDs light up slowly, then it starts" delay). This runs whenever an emitter instruction is known, not only when the shadow test is enabled, since a probe or a previous run may have left the LEDs off. The emitter is restored/left on in the `finally`.

**Fast path (roadmap R1–R3, 2026-09-07).** The cameras are opened in the background (`Capture.open_async`, one thread per device — the two interfaces of a Windows-Hello module open in parallel without contending) while the dlib models load; a busy device is retried twice 150 ms apart before the attempt fails. `hw.apply()` on this path only sets the OpenCV thread count — no subprocess, no OpenCL probe. Measured: first frame set at ≈ 1.1 s after start instead of ≈ 1.4 s; the remaining floor is the IR module itself (≈ 520 ms in the V4L2 open, ≈ 200 ms to the first frame), which only a resident helper that keeps the device open removes (roadmap R4/R5).

**Dark-room camera rule and watchdog (2026-10-22).** On a Windows-Hello module (RGB and IR as interfaces of one USB device), **opening the RGB camera in a dark room starves the IR stream**: the IR node keeps delivering a flat pedestal frame (mean ≈ 24, std 0.3–0.7 against ≥ 9 for a live frame), the LED goes out, and closing RGB does not revive it — IR has to be reopened (≈ 1.4 s). Every night-time no-match in the log had both cameras open. Two layers fix it. (1) `engine/ambient.py` reads the laptop's ambient light sensor over iio (≈ 36–99 lux lit, 0 dark on the reference machine; a node without a scale is ignored, `FACEUP_LUX` overrides for tests) and, below `DARK_LUX` = 5 lux and only when both nodes are one USB device, opens **IR only**; the attempt logs `ir-only (dark)`. In the dark RGB sees nothing anyway, so nothing is lost. (2) Where there is no sensor, `Capture._watch` drops RGB (`_release`) and reopens IR after `STARVED_FRAMES` = 6 flat IR frames (std < 1) *while RGB is dark too* — a flat IR frame next to a usable RGB picture is a different fault and is left alone. Measured: IR-only dark frames sit at 0.31 from daylight-enrolled models (gate 0.40), so no separate night enrolment is needed. LED-only imaging (on − off) gained ≈ 0.02 and was not worth its toggle. Caveat: the driver reports the LED control as on at open, but the LED may not be lit until a SET — the emitter warm-up above always writes it. The rule is specific to modules that share a USB device; separate cameras never trigger it.

**Phases.** Every log line carries `ph=models,cams,face` (ms to models loaded, cameras open, first face), so a slow attempt says where the time went.

### 6.3 Quality gate (≈ 1 ms/frame)
Drop frames that are dark (below threshold), blown out (p95 > 250), blurred (Laplacian variance < 60 at 320 px), or where the face box would be < 60 px. A window with *only* dark IR frames ends with reason `dark` — this is the emitter-failure signature and `doctor` reports it.

### 6.4 Detect · track · fuse (≈ 4 ms/frame YuNet; 60 ms HOG; 700 ms CNN@240)
- Detect on the best frame first, then track (correlation tracker) — detection is not re-run every frame.
- Align with 5-point landmarks, compute the 128-d embedding per frame per camera.
- **Fusion:** keep the top `fusion_frames` by quality, average the embeddings (L2-normalised). Averaging removes per-frame noise — the largest single reducer of false rejects after multi-pose enrollment.

### 6.4a Recognisers (pluggable, `recognition.recogniser`)

`engine/recognise.py` puts the embedding network behind one interface; `Models.embed` dispatches, the store records which space its vectors live in (`recogniser` field — a change forces `enroll --fresh`; the engine refuses a mismatched store with reason `store`).

| Backend | Model | Dim | Cost | Trained on | Same-person radius |
|---|---|---|---|---|---|
| `dlib` (default) | ResNet-34, distro `dlib-models` | 128 | 30 ms | colour photos | 0.6 (99.38 % LFW) |
| `arcface` | insightface `w600k_mbf` (MobileFaceNet) or `w600k_r50` (ResNet-50), OpenCV DNN | 512 | 5 ms / 57 ms | WebFace600K, ArcFace margin | ≈ cos 0.40 → 1.10 Euclidean (provisional) |

**Why ArcFace.** Additive-angular-margin training produces far tighter identity clusters than dlib's metric loss: on IJB-C, r50 reaches ≈ 95 % TAR at FAR 1e-4, the regime a Windows-Hello-class false-accept target (1 : 100 000) lives in, where dlib's model was never characterised. It is what makes a look-alike land well outside the gate instead of near it.

**One scale for every gate.** Both backends report the Euclidean distance between unit embeddings; ArcFace's is multiplied by `0.6 / ARCFACE_RADIUS` (= 0.545) so that every threshold in the config — `certainty`, `hard_cap`, `early_reject`, `learn_*` — keeps its dlib meaning. Whitening thresholds are learned per store and need no scaling. `Store.scale` carries the factor; `Store.distance()` / `min_distance()` apply it.

**Alignment.** ArcFace expects the 112×112 five-point crop. YuNet's own five landmarks (image-left eye, image-right eye, nose, mouth corners — order verified on 35/35 live frames) feed `recognise.align()`; when a box has none (HOG/CNN, enrollment hints) the dlib 5-point predictor's eye centres + nose give a 3-point similarity. Measured on the reference IR camera (face ≈ 116 px): YuNet landmarks beat 68-landmark-derived points (p95 spread 0.57 vs 0.78 raw for r50) and flip-averaging gained < 0.02, so neither extra is used.

**Measured so far (one subject, single frames, raw units).** dlib: median 0.10 / p95 0.29 from the mean; ArcFace r50: 0.36 / 0.57 (= 0.20 / 0.31 in dlib units); mbf: 0.39 / 0.66. Multi-frame fusion roughly halves these. The *impostor* side — the point of the exercise — needs a second person in `faceup lab`; until that calibrates `ARCFACE_RADIUS`, ArcFace is opt-in.

**Install.** `sudo faceup recogniser install [mbf|r50]` downloads the pinned insightface release zip (sha256 of the zip and of the ONNX member both verified), writes `/etc/faceup/arcface.onnx`, and loads it once. The packs are licensed for non-commercial research use, so faceup fetches, never bundles, and says so. `recogniser use arcface` switches the config; `doctor` reports the active backend; `setup` offers it.

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

**Early reject (roadmap R7).** Each frame also carries its own raw distance (`Obs.dist`). When `fusion_frames + 2` consecutive embeddings of the primary camera all sit above `recognition.early_reject` (0.65 by default, above dlib's 0.6 same-person radius; 0 disables), the window ends at once with `no-match` instead of running to `window_ms`: a stranger reaches the password prompt ≈ 1.5 s sooner. The user's own frames never all sit that far.

### 6.5a Lighting: what actually varies, and the whitened gate

Measured on the reference IR camera (dim room, mean 47–64): two consecutive frames of the
same face already differ by 0.15–0.32 in embedding distance (pose/expression/noise), while
simulated lighting changes shift a frame's embedding by 0.10–0.23 (dark γ2.2, bright, side
light, night = ×0.35 + sensor noise). Photometric normalisation does not help: Tan–Triggs
*doubles* the shift (dlib's network was not trained on such images) and CLAHE only helps
gamma-type changes. So the lighting problem is not one of pixels but of **where the user's
embedding cloud extends**, and the principled tool is to learn that cloud:

- **Samples, not just means.** Each stored model keeps the individual embeddings it was fused
  from; enrollment additionally embeds synthetic lighting variants of every sample (night =
  ×0.35 + noise, bright γ0.45, side-lit ramp). The variants are never match targets — they
  only teach the next step which directions are "just lighting".
- **Within-class whitening (WCCN).** From all samples' deviations around their model means,
  Σ_w = within-class scatter; W = (Σ_w + λI)^(−1/2) with λ = trace(Σ_w)/128 (shrinkage — 128
  dims from a few hundred samples). The whitened distance |W(x − μ)| scales down the
  directions this user's face is known to vary in (pose, expression, lighting) and leaves
  identity directions alone: a night-time embedding of the user comes out close, a stranger
  — who differs along identity directions — stays far. Threshold = 95th percentile of
  leave-one-out whitened distances (each sample vs the mean of the others in its bin, the
  same situation as a live fused embedding against a stored model) × 1.15. Synthetic sweep:
  the user at night (raw 0.56) accepted, a 28 %-different "sibling" (raw 0.40) and strangers
  rejected — whitening *reorders* them correctly where the raw distance cannot.
- **Decision.** `recognised = raw ≤ certainty`, OR (`raw ≤ hard_cap` (0.55, below dlib's
  0.6 same-person radius) AND `whitened ≤ threshold`). The whitened path only ever extends
  acceptance along the user's own variation directions. Both distances are logged
  (`irw=2.9/4.1`).

### 6.5b Online adaptation

The only way to get *real* night-time samples from a daytime enrollment is use. After a
confident accept — raw ≤ 0.30 (a stranger never gets near), at least one liveness PASS and
no FAIL — the live fused embedding is stored as a **learned** model of its pose bin if it is
novel (≥ 0.25 from every existing model there — 0.15 was below the frame-to-frame noise floor measured above and learned on nearly every accept; a store with 70 learned models on 168 attempts was the symptom), at most 2 learned models per bin and
spectrum (oldest rotated out), originals never replaced; whitening is refitted. `faceup
models` marks them, `faceup models --forget-learned` drops them. Over a few evenings the
store thus acquires the user's actual dark-room appearance. A learned model keeps the *per-frame* embeddings that were fused (not just their mean), so it feeds the whitening scatter — a single sample never could.

**Bounded learning (2026-10-22).** Novelty is now measured against the *enrolled* models only — measuring against learned ones let each learned model justify the next, a chain that walked away from the enrolment. A learned model needs ≥ 2 fused frames, and on top of the 2-per-bin cap there is a total cap of learned models per camera (oldest removed first). Whitening refits move the threshold by at most ±20 % and never below the calibration session's floor (`store.fit_whitening` clamp), so a run of odd accepts cannot open the whitened path. Limit: this is still self-training on accepted attempts; a false accept would be learned, which is why the confidence bar (raw ≤ 0.30, a PASS, no FAIL) sits well inside the gate.

### 6.5c Hardware path (future): emitter-differential imaging

`I_on − I_off` over an emitter toggle contains only the emitter's light — daylight cancels
exactly, day and night produce the same IR image. Depends on the same toggle-speed
measurement as the shadow test (§7.3, §15.1).

### 6.6 Result
Early reject uses the **median raw distance of the last max(6, fusion_frames + 2) frames** above `early_reject` (`_confident_reject`); the earlier all-frames rule never fired because one near frame vetoed it. Exit 0 → PAM `sufficient` succeeds. Any other outcome → exit 1, one log line:
`2026-08-28T21:02:11 user=octave result=accept ir=0.31 rgb=0.36 votes=cross:PASS cnn:PASS shadow:ABSTAIN motion:PASS t=380ms`

### 6.7 Threshold calibration (`faceup calibrate`, end of enrollment)

No gate is a constant. `engine/calibration.py` derives the recognition thresholds from **this user's real distances on this machine** and keeps them current:

- **Session.** Eight guided passes of ~3 s (screen, keyboard, lean back, lean in, tilt, left, right, change the light) while the lab evaluator measures every frame set exactly as the engine would: the fused live embedding's raw and whitened distance to the store, per spectrum, with the face light state. Then, per spectrum: `gate = clamp(p95(raw) + 0.05, 0.33, 0.45)`, `hard_cap = min(gate + 0.15, 0.55)`, `early_reject = max(0.65, hard_cap + 0.10)`, `learn_max_dist = gate − 0.10`, and the store's whitening threshold `= p95(whitened) × 1.15`. The session's fused embeddings enter the whitening scatter as real, varied samples (the leave-one-out threshold over correlated enrollment frames was 3–6× too tight, §0 of the roadmap). Evidence — counts, percentiles, light conditions seen per spectrum, prompts with a face — is stored in `calib.calibration`; `doctor` shows it and flags a gate edited by hand afterwards. A p95 above the cap is reported as "re-enroll", never as a looser gate. Enrollment ends with the session followed by a two-attempt self-check; `calibrate pose` keeps the old 3D-model rebuild.
- **The window (`appui.CalibrateWindow`, default; `--text` for prompts).** Built like enrollment so it stays at camera rate: the UI-thread loop (`gui.calibration_session`) only reads frames, runs YuNet + 68 landmarks + pose (~13 ms/frame, measured) and draws; the dlib embeddings (~85 ms each, GIL-holding) run in the `_Sampler` PROCESS at most every `SAMPLE_EVERY_MS`, and distances are computed from the returned embeddings. An earlier version embedded inline on the UI thread and stuttered; moving it off-thread is what enrollment already did. The session turns the IR emitter on for its duration (a probe may have left it off) and restores it after. It opens on an **intro** screen that auto-starts ~2.5 s after it first sees your face (or Enter), shows a short "✓ Pass complete" beat between passes, and warns if no face appears. Same shell as enrollment: mirrored preview with the face outline on the left, one card on the right — PASS i OF n, the instruction, a **movement cue drawn on the camera view** (`Preview.set_cue`: big animated glyphs over your own face — a target for centre, bobbing chevrons for down/left/right, chevrons pointing in/out for closer/back, a rotation arc for tilt, a sun for light; the pass fills a ring, everything turns green when the pose is met) with only a short word of text, a progress bar, the live distances, a pill (Looking / Move as shown / Hold it… / Face too dark). Each pass is **verified from the pose estimate**: `calibration.Collector` counts only frame sets whose display-space yaw/pitch/roll — or face width relative to the "screen" pass — satisfy the pass's condition (`PROMPTS[*][3]`), completes after `NEED_FRAMES` = 20 of them or `PASS_TIMEOUT_S` = 15 s, and still measures every frame with a face (transitions are genuine data). Enter skips a pass, Esc cancels with nothing written. Display-space rule: `gui.to_display` sign applied once in `Collector.display_pose` (`sgn = −1` when mirrored); conditions are written for what the user sees. The summary card at the end shows per-camera counts, median/p95, the derived gate and whitening threshold, and the warnings.
- **Rolling refresh.** Every confident accept (no FAIL vote) appends its raw and whitened distances and light state to a 60-entry ring in the store. Every 20 accepts the whitening threshold is recomputed from the ring, bounded to ±20 % per refresh and never below the session's floor. A condition the session never saw (night, new glasses) first passes through the whitened path or online learning, then moves the threshold — adaptation without a human retune. The raw gate stays in config and is never rewritten by the engine; `doctor` says when the ring disagrees with it (`calibrate refresh` re-derives from the ring).
- **Guardrails.** `GATE_MAX = 0.45` is a hard ceiling (a look-alike's territory); `GATE_MIN = 0.33` a floor against a session that was too easy; the session aborts after two faceless prompts; nothing is written unless ≥ 8 measurements per spectrum exist.

### 6.8 Security policy, identities, conditions

**Policy (`engine/policy.py`).** Three presets resolved *at attempt time* from the measured gates (an unlock never rewrites the config): `convenience` (raw gate or whitened path, one PASS), `balanced` (default: the colour camera must agree whenever it sees the face well — `rgb_when_bright` — and two PASSes when two voters exist), `strict` (gate −0.05 capped at 0.35, RGB match mandatory, whitened path off, a shadow-test or anti-spoof PASS required, learning off). Presets only ever *tighten* the calibrated gate. `security.policy_<service>` overrides per PAM service (`sudo`, `polkit`, `login`, `unlock`; `off` disables). Text-console logins are off unless `behaviour.tty_login`. **Lockout**: `security.lockout_failures` liveness FAILs within `lockout_minutes` (spoof signals only — never "no face" or a plain mismatch) pause face unlock for that user for `lockout_minutes`, reason `lockout`; the password path is untouched by construction. The log carries `policy=` when not balanced.

**Identities.** A store belongs to a Linux login; each model carries an `identity` ("" = the owner, else a name). `enroll --identity "Alice"` runs the fused window for another person; `Store.nearest()` returns the matched model so the accept logs `who=Alice`; online learning inherits the matched identity. `--fresh` scoped to the identity never deletes the others; `models --remove-identity` does. Any registered face unlocks the login — the security decision is the registration, made by root.

**Conditions.** `enroll --add glasses` (headset, night…) runs `calibration.CONDITION_POSES` (centre, down, left, right) and adds models tagged `condition`, clearing only that identity's *that* condition's labels. Conditions and extra identities never re-derive the owner's gate; the owner's calibration stays the source of truth.

## 7. Liveness voters

Each returns `PASS`, `FAIL`, or `ABSTAIN` with a numeric metric; thresholds are config keys with the defaults below.

### 7.1 Cross-spectrum consistency (profile A) — deterministic, ~0 ms
Requires a face in the IR frame **and** the paired RGB frame, and their boxes to agree after calibration: `IoU(box_ir, calib(box_rgb)) ≥ 0.5`.
- Screen replay: face in RGB, none in IR → **FAIL**.
- IR-tuned print: face in IR, RGB sees flat grey → **FAIL**.
- Dark room where RGB legitimately sees nothing: IR mean ok, RGB mean < `rgb_dark_threshold` → **ABSTAIN** (not FAIL). This distinction is what keeps night-time unlocks working.

### 7.2 Anti-spoof CNN (RGB) — ~3 ms per frame
Silent-Face-Anti-Spoofing (Minivision MiniFASNetV2, Apache-2.0; ONNX export by yakhyo/face-anti-spoofing, Apache-2.0) **bundled** as `faceup/models/minifasnet_v2.onnx` (1.7 MB, sha256 pinned in the voter and checked at load), run with `cv2.dnn` (loads on OpenCV 4.6). Preprocessing = the reference `crop_face`: detector box ×2.7 around its centre (scale clamped to fit the frame), 80×80, BGR, raw 0–255, no normalisation. Output: print / real / replay softmax.
- Measured on recorded daylight RGB (80 frames): p_real min 0.996, p5 1.000; still ≥ 0.91 under darkening ×0.35, blur, half resolution, JPEG q30, colour tint. On IR frames shown as RGB real faces score down to 0.01 — the net is a colour model, so a grey frame ABSTAINs.
- Asymmetric because a FAIL is sticky: median p_real over the newest ≤ 3 lit RGB faces ≥ 0.90 → PASS; FAIL only when ≥ 2 frames are scored and every one is ≤ 0.02; otherwise ABSTAIN. A single odd frame never sends a genuine user to the password.
- Frame dark, grey, box < 40 px, broken/mismatched model → ABSTAIN (fail open). `faceup antispoof status|on|off|test DIR`; doctor/app expose on/off.
- Not yet measured against real attacks (no printed photo / phone replay recorded): the FAIL threshold is deliberately far from the real-face distribution until a spoof set exists.

### 7.3 Emitter-modulation shadow test (IR) — ~2 frames
`linux-enable-ir-emitter` discovers the UVC extension-unit instruction that switches the LEDs. If that instruction can be applied at runtime in < 50 ms (measured once in `setup`, stored as `shadow_test = true/false`), the engine captures one frame **emitter off** and one **on**, aligned by landmarks, and computes:
- `Δ = mean(on − off)` inside the face: must be large (the face is near the LEDs) — a distant screen barely changes.
- **Gradient ratio**: on a real face the illumination change is strongest on the nose/brow and weakest under them (self-shadowing); on a flat print the change is spatially uniform. Metric: coefficient of variation of Δ over face regions; `≥ 0.25` → PASS, `≤ 0.10` → FAIL, else ABSTAIN.
- Any toggling failure → ABSTAIN and the voter is marked unavailable for the session.

**Implementation (2026-09-07, roadmap L1/L4).** `engine/emitter.py` drives the LEDs directly: it parses linux-enable-ir-emitter's files for the instruction marked `status: start` (unit, selector, bytes) and sends it with the UVC extension-unit ioctl (`UVCIOC_CTRL_QUERY`, ctypes, no dependency). "Off" is whatever the control read before we switched it on, else the device default (GET_DEF), else the minimum — if none differs from "on", the emitter is `unavailable` and the voter abstains. `faceup emitter toggle-bench` switches off/on three times, measures the frame-brightness step and the latency until the brightness follows, and writes `liveness.shadow_toggle_ms` (usable when ≤ 100 ms and the step is ≥ 12 levels). With `shadow_test = "auto"` the engine then runs the test **once per attempt after recognition**: emitter off → 2 settle frames → off frame, on → 2 frames → on frame (≈ 130 ms), `shadow.vote()` on the recognised box, LEDs left on. Any error → ABSTAIN; the LEDs are restored in `finally`.

**Finding the control by measurement (`faceup emitter probe`).** linux-enable-ir-emitter's search assumes the LEDs are off until configured and asks the user to watch for a flash in a dark room. On modules whose LEDs are on by default (the reference laptop: the tool refuses with "already working") it stores nothing, so nothing can be toggled. The probe instead alternates every small extension-unit control between its minimum and its current value every four frames — faster than the firmware auto-exposure can follow — and watches the *face* brightness; the control whose minimum darkens the face by ≥ 12 levels is the LED. Reference module: unit 5 selector 2, range 0–100, default 25 = on, 0 = off, face 154 → 107 while the background moves 54 → 47; values above 25 change nothing. Result stored in config `[emitter]`, used by `Emitter` ahead of the tool's files; `toggle-bench` then measured 100–133 ms to full brightness (3–4 frames), and the shadow test settles ⌈latency/33 ms⌉+1 frames per state. In a lit room the LED adds only a few levels to the face, so the voter ABSTAINs honestly by day and decides in dim light — which is when it matters.

**Earlier reading, corrected.** The handout said the emitter was "not configured" because the IR *frame* mean was dim (≈ 35). With a working emitter the room is black and only the face is lit — a dim frame mean is the signature of success. `doctor` now meters the detected face and reports "lit face on a dark background". **On the reference laptop the tool's search found no working instruction** (every entry is `idle`/`disable`) — `doctor` now says so instead of "a device is configured", and the toggle bench refuses until `emitter configure` succeeds. That single fact explains the dim IR frames noted since 2026-08-31.

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

### 8.1.3 Fused setup (`faceup enroll`, default)

Enrolment and threshold calibration in ONE guided window (`gui.guided_setup`), so setup is a single sweep instead of two commands. It reuses the calibration window's look: mirrored preview, a **target ring** the head is guided into, a **face marker** at the live angle, and an **arrow derived from marker → target** — so left/right is always right regardless of pose sign (the arrow points where the head must go). Poses cover the grid (`calibration.SETUP_POSES`: centre, up, down, left, right, closer, back, light); left targets negative display yaw (the user's left, matching §8.1's hint). Each pose captures embeddings in the `_Sampler` process; `calibration.build_from_captures` fuses the best ~2/3 per pose into the pose's model (the rest **held out**) and measures the held-out frames against the finished store — genuine fresh-frame distances, so `derive_from_holdout` sets the gate honestly in the same sweep (no self-match zeros, no transition inflation that capped the two-step gate). `--classic` keeps the old free-move map + a separate `calibrate`.

**Frame selection and holdout (2026-10-22).** A frame counts for a pose only once the head is within `POSE_TOLERANCE_DEG` = 12° of its target (frames on the way there used to enrol transitions). Dark frames enrol nothing. Each camera is ranked by its own quality, not by the other's. A pose completes on **usable samples** — `NEED_SAMPLES` = 9 pose-true, lit embeddings (6 for the model, 3 held out), not camera frames, so a slow sampler cannot finish a pose with nothing in it; embeddings still in flight when a pose completes are kept. The holdout is the **last third in time** of each pose, not a random third: adjacent frames are near-duplicates, and a random split would measure the model against copies of itself. `calibration.usable_rows`/`select_rows` implement this; the light pass stays open 4 s so a light change can actually happen. Honest limit: the gate is measured on ~3 frames × 8 poses from one sitting; the rolling refresh (§6.7) is what makes it representative.

**Window behaviour.** The window path asks nothing in the terminal; IR↔RGB alignment runs during the intro (while the user settles) instead of as a separate step; the done card lists the poses that were missed and closes itself; closing the window early keeps what was enrolled; an empty sweep is never saved. Launched with neither a terminal nor a display, `enroll` fails with a clear message instead of answering its own prompts.

**Glasses in the same window.** The done card offers "I also wear glasses": a short sweep of `CONDITION_POSES` with the glasses on, models tagged `condition=glasses`. The calibrated gate is untouched — the extra models only widen what matches, they never loosen the threshold. `enroll --add glasses` remains the standalone path.

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

- **PAM placement:** `auth sufficient pam_exec.so quiet stdout seteuid /usr/local/bin/faceup-auth` inserted *before* `pam_unix` in `common-auth` (or the distro equivalent), with the original file backed up. `pam_exec` runs the engine as root with `PAM_USER` set; the engine authenticates only that user.
- **Files:** models 0600 root, config 0644 root, engine 0755 root-owned, log 0640 root:adm. Snapshots off by default.
- **Residual risk (documented):** two high-quality prints aligned for both lenses, plus a way to fake near-field shadowing, could defeat a profile-A setup. This is the same order of difficulty as attacking Windows Hello on the same hardware. faceup is convenience-grade with strong anti-casual-spoofing, not a hardware security module.
- **Lockout:** none by default — the password is always available, so lockout only creates denial-of-service. Optional `lockout_failures` for users who want a cooling-off after N spoof detections.
- **No timing side channels of note:** results are all-or-nothing and the password path is unchanged.
- **Coexistence with Howdy:** `setup` offers to disable `pam_howdy` (two face steps in a row is pointless) and can *import* nothing — dlib embeddings from RGB frames don't match IR; re-enroll.

### 9a. Session feedback (what the user sees)

pam_exec runs the engine behind a password prompt, so without help an attempt is two silent
seconds followed by a prompt. `feedback.py` gives every attempt a visible shape, chosen by the
channel the caller offers — automatically, and never on the critical path (every write is
wrapped; a feedback failure cannot fail authentication):

| Caller | Channel | What appears |
|---|---|---|
| sudo, su, `faceup auth` in a shell | stderr is a terminal → an animated status line (`◐ Looking for your face… 0.8s`) redrawn in place, then `✔ Face recognised · IR 0.21 · cross ✓ · motion – · 0.9 s` in green or `✘ Face not recognised — enter your password` in yellow | |
| GDM / GNOME lock screen, SDDM, KDE lock screen (kscreenlocker), GNOME and KDE polkit agents | the PAM line's `stdout` option: each line printed to stdout becomes a `PAM_TEXT_INFO` message, which the greeters render under the password field — the channel fprintd uses for "Swipe your finger" | "Looking for your face…", then at most one hint ("Face recognised — move your head a little" when only the motion voter is holding) and the result |
| any desktop session (`behaviour.notify`, default `accept`) | a libnotify notification sent as the user on their own session bus, detached; **transient** (GNOME ignores the expiry time and would otherwise keep every attempt in the notification list) and **replacing** the previous one (id kept in `$XDG_RUNTIME_DIR/faceup-notify.id`), so attempts never pile up | "Unlocked with your face · sudo · IR 0.21 · cross ✓ · 0.9 s"; `all` also reports failures, `off` disables |

Rules: a terminal on stderr wins over the PAM channel (sudo would otherwise print both);
messages are few because GNOME Shell queues info messages and shows each for a minimum time;
"looking…" is printed *before* cv2/dlib import so the greeter shows it during the ~0.7 s of
loading; nothing is said for `ssh`, `no-models` and `store` (the user can do nothing about them
at a prompt). The notifier is spawned with stdin/stdout/stderr on `/dev/null` — it must not
inherit pam_exec's stdout pipe, or the greeter would wait for the notifier to exit.

**Keyring (roadmap K0).** `pam_gnome_keyring`/`pam_kwallet5` unlock the login keyring with the password PAM received; a face accept carries none, so after a face *login* the keyring stays locked until first use (fingerprints have the same limit). Screen unlock, sudo and polkit run inside a session whose keyring is already open and are unaffected. `behaviour.skip_login = true` leaves the first login of a session to the password: `feedback.is_session_login()` recognises a greeter service (`gdm-password`, `sddm`, `lightdm`, `greetd`, `login`…) with no session bus for the user yet, and `auth` returns `login` (silent, logged). `doctor` explains this next to the PAM status.

### 9b. Native PAM module and the keyring (roadmap K1)

`data/pam_faceup.c` (built by `install.sh` when gcc is present, installed as `/usr/local/lib/security/pam_faceup.so`; pam_exec stays the fallback) does three things pam_exec cannot: it forwards the helper's stdout lines itself as `PAM_TEXT_INFO`, it runs the helper with both uids root (sudo's real uid is the caller), and on an accept it can hand PAM an authentication token. It is deliberately tiny (fork/exec, a pipe, `pam_set_item`), declares the five libpam entry points it uses instead of needing the -dev headers, and is fail-open by construction: anything but helper exit 0 is `PAM_AUTH_ERR`.

**Stack placement.** `sufficient` would end the whole stack on success — including the caller's own lines after `@include common-auth`, which is exactly where `pam_gnome_keyring.so` sits in `gdm-password`. The module is therefore inserted like `pam_unix`: `auth [success=N default=ignore] pam_faceup.so`, N computed to land on the stack's `pam_permit` line (`pam.render_enabled(native=True)`), so the keyring module still runs and reads the token. `faceup pam enable` picks the native line when the module exists and upgrades a pam_exec line in place.

**The credential.** `faceup keyring enable` asks for the login password, verifies it against `/etc/shadow` (crypt; yescrypt supported), and seals it with `systemd-creds`. **Key choice (`keyring.choose_key`)**: the **host key** by default when the root disk is encrypted — protected at rest by FDE and instant to decrypt; a firmware-TPM decrypt costs 2–4 s and would be felt as a freeze at every login. The **TPM2** (PCR 7, Secure Boot state) is used when there is no FDE, or forced with `--tpm` (stronger, at that latency cost). Host key without FDE is refused unless `--force-host-key`. File `/etc/faceup/keyring/<user>.cred`, 0600 root, credential name `faceup-keyring-<user>` so it cannot be replayed for another user. The module **starts the `systemd-creds decrypt` in the background before running the helper**, so the decrypt overlaps face recognition and even a slow TPM adds nothing to the wall-clock; it collects the plaintext only after the helper reports an accept (aborting the decrypt on a reject) and calls `pam_set_item(PAM_AUTHTOK)`; `pam_gnome_keyring`/`pam_kwallet5` then unlock exactly as after a typed password. `keyring disable` deletes the file; a firmware or Secure Boot change invalidates a TPM-bound credential (re-enable).

**Timeout and portability (2026-10-22).** The module enforces its timeout (`timeout=`, default 8 s) with `poll()` on the helper's pipe and, on expiry, kills the helper's **process group** (the helper runs under `setsid`) — a hung helper that printed nothing used to block the login, since a blocking `read` never returned. `faceup-auth` adds a kernel `SIGALRM` backstop of its own, so even a pam_exec line cannot hang. Both fall through to the password. `engine/pam.py` targets `common-auth` (Debian family) or `system-auth` (Fedora/Arch), writes atomically through symlinks (the link stays a link), and **refuses stacks managed by `authselect`** with instructions instead of editing files the tool will overwrite (`PamError`). `install.sh` gives distro-aware package hints via `faceup/distro.py` (apt/dnf/pacman/zypper), checks every PAM file on uninstall, and relabels for SELinux. Other portability: libadwaita < 1.3 fallbacks, X11 under sudo (XAUTHORITY), any Wayland socket name, YuNet → HOG fallback when the ONNX cannot load, `FACEUP_DLIB_DIR` and more dlib model paths (`detect.dlib_dirs`), 16-bit IR frames (`_to_frame`), libcrypt on minimal/musl systems.

**Repair.** On the reference machine an earlier experiment had commented out `auth optional pam_gnome_keyring.so` in `gdm-password` and the `password` line in `common-password` (tagged `# faceup-keyring`), which left even *password* logins with a locked keyring. `faceup keyring status` detects the tag, `keyring repair` (or `keyring enable`) restores the lines.

**Test.** `tests/test_keyring.py` builds the module and runs a real `pam_authenticate` over a private config directory (`pam_start_confdir`, Linux-PAM ≥ 1.4) with a fake helper and a fake systemd-creds; a `pam_exec expose_authtok` line consumes the token the way the keyring module would. Reject → the stack's `pam_deny`, no token; accept without a credential → success, no token.

## 10. Degraded modes

| Situation | Behaviour |
|---|---|
| Emitter dead, dark room | IR frames dark → window ends `dark`; RGB also dark → password. `doctor` flags the emitter. |
| Emitter dead, lit room | IR sees you by ambient light; works. Shadow test ABSTAINs (no toggle). |
| RGB covered (privacy shutter) | cross-spectrum ABSTAINs; CNN ABSTAINs; liveness relies on shadow + motion. If both abstain → degraded rule accepts on recognition alone *only if* `require_liveness_when_available = false` (default true → password). |
| Anti-spoof model missing/corrupt | voter abstains (checksum checked at load); `doctor` fails the item. |
| Dark room, Windows-Hello module | IR only (ALS < 5 lux) or watchdog drops RGB and reopens IR; logged `ir-only (dark)`. |
| Helper hangs | module `poll()` timeout kills the group; `SIGALRM` in the helper → password. |
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
sudo faceup lab [--text] [--subject NAME]   live evaluation: policies side by side, subject recording, per-subject accept rates
sudo faceup-auth                   the engine (PAM calls this; also runnable by hand for debugging)
```

## 11b. The application (`faceup app`)

`app.py` is one GTK 4 / libadwaita window (Gtk.StackSidebar + Adw.PreferencesPage per page) that renders `health.collect()` — the structured checks behind `doctor`: sections of items with a level, a sentence and, when one command fixes it, that command — and runs every action as a **subprocess of the CLI** (`python -m faceup …`) with the output streamed into a console pane. Design decisions: the CLI is the single source of truth (no second implementation of enrol/keyring/PAM), long windows (enrol, calibrate, lab, emitter check) open as their own windows from the subprocess exactly as from a terminal, the keyring password comes from an Adw dialog and is passed on stdin (`keyring enable --password-stdin`), and without root the app is read-only and says so. The launcher entry runs `pkexec faceup app` through the polkit action `org.faceup.app` (`allow_gui`, `auth_admin` — asked per action, never remembered, so another process of the session cannot reuse it). The roadmap's privilege split (U1: user-side app talking to a root D-Bus service) is the next step; the page model is built for it.

## 12. Performance budget (Ryzen 5 6600HS, distro dlib without AVX)

| Stage | Cost | Notes |
|---|---|---|
| Camera open + first frames | 80–150 ms | dominates the fast path; keep devices open for the whole window |
| Quality gate | 1 ms/frame | |
| YuNet detect | 4 ms/frame (full frame, no tracking needed) | HOG 60 ms / 5 ms ROI; CNN@240 700 ms — opt-in |
| Landmarks + embedding | 30 ms/frame/camera | 4 fused frames × 2 cameras ≈ 240 ms spread over the window |
| Cross-spectrum, motion | < 2 ms | |
| Anti-spoof CNN | ~3 ms/frame (≤ 3 frames) | measured, cv2.dnn CPU |
| Shadow test | 2 frames ≈ 100 ms | only if toggle is fast |
| **Typical accept** | **250–450 ms** | |

## 11a. `faceup log` (the attempts as data)

`attempts.py` parses the log lines back into typed rows. `faceup log` prints coloured one-liners; `--summary` gives accept rate, median/p90 time (median time to accept flagged above 800 ms), IR distance median against the gate, failure reasons, per-service counts, learned-model count and liveness FAILs; `--follow` tails; `--json` for scripts. `doctor` shows the last-20 summary. Reading needs group `adm` or root.

## 12a. The lab (measurement instrument)

`faceup lab` runs the pipeline continuously and shows every intermediate quantity — raw and
whitened distance per camera against their gates, nearest stored pose, liveness votes, face
light, head pose — and the decision under three policies on the same frames: **strict** (raw
≤ certainty), **whitened** (§6.5a path), **whitened + liveness** (the production rule).
Threshold sliders re-decide live; "apply to config" persists them. A subject tag and Record
write every evaluated frame-set to `/var/lib/faceup/lab/<time>-<subject>.csv`; the summary
gives, per subject and policy, the accept rate and the raw/whitened distance statistics.
That is how reliability is measured: the owner across poses and lighting should score
~100 % under all policies with raw distances far under the gate; other people and spoof
attempts should score 0 % with raw distances above the hard cap or liveness FAILs.

## 13. Testing

- **Unit:** recorded frame sequences under `tests/fixtures/` (real, phone replay, print, IR print, dark) — the pipeline must produce the expected reason for each. Fixtures are contributed frames, never embeddings of real users.
- **Voter tests:** synthetic homography motion vs jittered 3D motion for §7.4; synthetic uniform vs gradient Δ for §7.3.
- **Integration:** `faceup doctor --spoof` interactive self-test; `faceup-auth` exit codes under missing camera / busy camera / no models / timeout.
- **Bench:** `faceup bench` output committed for reference hardware in `docs/bench/`.

### 13a. Replay harness (roadmap L7)

`faceup lab --record-frames DIR` saves every frame set as PNGs plus `index.json`; `faceup auth --replay DIR` (or `FACEUP_REPLAY=DIR`) runs the unchanged `Engine.authenticate` on them through `replay.ReplayCapture`, which implements the `Capture` interface. A recorded night, glasses, phone replay or stranger is thereby a regression test that needs no hardware; `tests/test_replay.py` already runs the whole loop on synthetic dark and blank frames (reasons `dark`, `no-face`, `camera`). Recorded corpora are biometric data: keep them out of the repo, 0700, and delete them when the test is done.

## 13b. Compatibility matrix (`tests/test_compat.py`)

The pipeline on recorded frames with a synthetic face (the detector, landmarks and embedding patched to deterministic stand-ins, so faceup's own logic is what runs): profiles A/B/C, other frame sizes (1280×720 IR + 640×480 RGB, 320×240), a stranger (early reject), the shadow test without an emitter, learning + rolling calibration, and the lab evaluator (step, decide, record, summary) with one or two cameras. Portability decisions it protects: TOML parsing without `tomllib` (Python 3.10), `crypt` via libcrypt (3.13), dlib optional on the ArcFace + YuNet path, motion thresholds as fractions of the face width, windows falling back to text without libadwaita, the lab dir honouring `FACEUP_ETC`.

## 14. Roadmap after v0.2

- GTK4/libadwaita front-end for enrollment (live preview with pose guidance).
- Better recogniser as an option (ArcFace ONNX via `cv2.dnn`) once its CPU cost is measured — the store's `recogniser` field already allows it.
- Packaging: `.deb`, AUR, Fedora COPR.
- Optional Howdy-model import for RGB-only profiles.

## 15. Open questions

1. Can the emitter be toggled fast enough on common Windows-Hello modules for §7.3? Measure on the reference laptop first; keep the voter "auto".
2. Should `require_rgb_match` default to true in bright rooms (auto-switch by RGB brightness)? Adds security at zero reliability cost when light is good.
3. GDM message channel for the motion challenge — PAM conversation text is shown by GDM but not by all greeters.
