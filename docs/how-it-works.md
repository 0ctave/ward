# How Ward works

[← back to the README](../README.md)

Why face unlock is unreliable on Linux, and how Ward fixes each part. None of this is magic — it's
the setup Windows does for you and Linux doesn't. Everything here, Ward automates; read it to
understand what's happening or to push further yourself. For the full rationale and exact thresholds,
see [DESIGN.md](DESIGN.md).

- [1. Use the infrared camera](#1-use-the-infrared-camera)
- [2. Make the emitter actually fire](#2-make-the-emitter-actually-fire)
- [3. Enroll many poses, not one face](#3-enroll-many-poses-not-one-face)
- [4. Let the machine set the threshold](#4-let-the-machine-set-the-threshold)
- [5. A detector that handles real heads](#5-a-detector-that-handles-real-heads)
- [6. Liveness: is it a real face?](#6-liveness-is-it-a-real-face)
- [7. Unlocking the keyring](#7-unlocking-the-keyring)
- [Windows Hello parity](#windows-hello-parity)
- [Honest limits](#honest-limits)

## 1. Use the infrared camera

Most laptops with Windows Hello have **two** cameras behind one USB device: a colour one
(`/dev/video0`, MJPG/YUYV, 720p–1080p) and an infrared one (`/dev/video2`, 8-bit `GREY`, usually
640×360). They often share the *same* name, so the only reliable tell is the pixel format — which is
why tools that guess by name pick the wrong one. `ward doctor` identifies them by probing the stream.

Infrared is the right camera because:

- **Lighting stops mattering.** The emitter floods your face with invisible light, so the sensor
  sees the same picture at noon and at midnight.
- **Screens are nearly invisible to it.** A photo on a phone barely registers in IR, so the cheapest
  spoof against colour face unlock simply doesn't work.

IR frames are naturally dim (a lit face on a dark background), so a naïve "too dark, give up" check
rejects them. Ward meters the **face region**, not the whole frame, and knows what a working IR image
looks like.

## 2. Make the emitter actually fire

This is the **#1 cause of "works sometimes."** On Windows the driver switches the IR LEDs on; on
Linux nothing does, unless the camera firmware happens to. So in a bright room the IR sensor sees you
by ambient light and face unlock works — and in a dim room it sees nothing and falls back to the
password.

Ward's `emitter probe` finds the LED control **by measurement**: it toggles each vendor-specific UVC
control and watches your face brightness until the picture lights up, then stores that and re-applies
it at boot and after suspend (USB cameras forget it on resume). It also benchmarks how fast the LED
switches — fast enough, and Ward can use an [emitter-modulation shadow test](#6-liveness-is-it-a-real-face)
as a second liveness check.

## 3. Enroll many poses, not one face

The recogniser (dlib's ResNet embedding, or ArcFace) is **pose-sensitive**: your embedding looking at
the keyboard is far from your embedding looking at the camera. The engine compares a live frame
against *every* stored model and takes the best match — so the fix is one model per pose you actually
use. (Windows Hello does exactly this with its "keep moving your head" enrollment.)

Ward captures them in **one guided sweep**: a ring frames your face, an arrow points where to move,
the ring fills as you hold each target. It covers the poses that matter at a laptop — straight on,
**looking down at the keyboard** (where your face is when a `sudo` prompt appears), tilted and turned
left/right, leaning in and back — and interpolates the angles in between. With enough frames it
solves for *your* own 3D face model, so head pose is then measured against your geometry, not a
generic head. Add conditions (glasses, headset) later with a short extra sweep.

## 4. Let the machine set the threshold

Every camera, room and face produces different distances, so a **fixed** match threshold is wrong
somewhere — too loose and strangers get in, too tight and you get rejected at an angle. Ward measures
instead.

The same enrollment sweep holds out some of your frames and derives the gate from your **real**
distances, with a hard ceiling for safety. (Distances are in dlib units — roughly `0.0` is identical
and `0.6` is "same person, loosely"; a well-measured personal gate lands around **0.33–0.45**.) From
then on every unlock feeds a rolling window, and a lighting-invariance threshold follows your actual
conditions within bounds, so your first night-time unlock is absorbed rather than rejected. `ward
doctor` shows when the thresholds were set and from how many measurements, and warns when your recent
unlocks disagree with them. Re-measure anytime with `ward calibrate`.

## 5. A detector that handles real heads

Every attempt has two stages: **detect** a face in the frame, then **recognise** it. If detection
fails, recognition never runs — so a frontal-only detector silently drops every tilted or turned
head. Ward uses **YuNet** (OpenCV's detector, bundled): about 4 ms per frame on a laptop CPU and
robust to turned and tilted heads, where dlib's HOG is frontal-only and its CNN costs hundreds of
milliseconds. `ward bench` times the options on your CPU.

## 6. Liveness: is it a real face?

Recognising *you* isn't enough — a photo of you is also "you." Ward runs several liveness voters with
**PASS / FAIL / ABSTAIN** semantics: a FAIL ends the attempt and feeds a spoof lock-out, while an
ABSTAIN never counts against a genuine user. The anti-replay guarantee comes from the infrared
camera; the rest add to it.

- **Cross-spectrum consistency** — the face must appear in **both** the IR and the colour frame, at
  the same place. A phone or monitor emits ~no infrared, so a screen replay has no face in IR and is
  rejected. *This is Ward's main anti-spoof.*
- **Anti-spoof CNN** on the colour crop (MiniFASNetV2, bundled, run by OpenCV) — catches the
  *physical* artifacts of a camera filming a screen or print (moiré, reflection, lack of depth). It
  passes real faces, fails only a confident repeated spoof, and otherwise abstains. (It does **not**
  catch a *digitally* re-encoded image — that's the IR layer's job; see `ward benchmark redteam`.)
- **Motion** — a real head moves like a 3D object; a photo or screen moves like a flat plane.
- **Emitter-modulation shadow test** — compares the face with the IR LED on vs off; the way shadows
  shift reveals 3D shape. Enabled once `ward emitter toggle-bench` proves the LED switches fast
  enough.

You can measure all of this on public datasets and against your own frames with `ward benchmark`.

## 7. Unlocking the keyring

GNOME Keyring and KWallet are encrypted with your **login password** — which a face login doesn't
have. So by default a face *login* leaves the keyring locked until first use (screen unlock, `sudo`
and polkit happen inside an already-running session and are unaffected).

Ward closes the gap with a native PAM module and a **sealed credential**:

```bash
sudo ward keyring enable     # verifies your login password once and seals it
```

The password is sealed by your encrypted disk's host key (instant) or, optionally, the TPM2. After a
face accept the module unseals it in parallel with recognition and hands it to PAM exactly as if you'd
typed it, and the keyring opens. `ward keyring status` shows the state; `disable` deletes the sealed
copy. Prefer to keep the first login on the password? `sudo ward tune behaviour.skip_login true`.

## Windows Hello parity

Windows Hello on the same camera does three things worth naming: a recogniser tuned for infrared
faces, emitter-lit liveness, and a specified false-accept rate of 1 in 100 000. Where Ward stands:

| Hello has | Ward | Status |
|---|---|---|
| IR-tuned recogniser, tight identity clusters | ArcFace option (`ward recogniser use arcface`, then re-enroll) | available; impostor calibration needs a second person in `ward lab` |
| FAR of 1 : 100 000 | measurable in the lab, **not yet claimed** | [roadmap](ROADMAP.md) |
| Both cameras must agree | `recognition.require_rgb_match = true` | available |
| Keyring unlocked after a face login | native `pam_ward.so` + sealed credential | available |
| Emitter-lit liveness | shadow test via `ward emitter probe` + `toggle-bench` | available |

## Honest limits

Ward is a **2D** system — no depth sensor. It defeats phone-screen photos and darkness; it does
**not** claim to defeat a high-quality printed infrared photograph, and **identical twins defeat any
2D face system**, Windows Hello included. A strong look-alike is kept out by the gate: keep a tight
`certainty`, turn on `require_rgb_match`, and measure the relative in `ward lab` before trusting the
device in a shared household.

Treat Ward as **strong convenience with a password fallback** — the same class as Windows Hello on
the same hardware, not the same class as Face ID. If you need something deterministic, a FIDO2 key
with `pam-u2f` is the reliable open-source option. And whatever happens, Ward **fails open**: every
failure ends at your password prompt. See [SECURITY.md](../SECURITY.md) for the full threat model.
