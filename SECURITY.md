# Security policy

Ward authenticates login, `sudo` and polkit by face. This document states what it does and does
not protect against, and how to report a vulnerability. Please read the threat model before filing a
security issue — several "bypasses" are known and documented limits, not bugs.

## Reporting a vulnerability

**Do not open a public issue for a security bug.** Use GitHub's **private vulnerability reporting**
(the *Report a vulnerability* button under the repository's *Security* tab), which opens a private
advisory visible only to the maintainers.

Please include: the ward version (`ward --version`), your distro and camera module, and the
steps to reproduce. If the report involves a presentation attack (photo, screen, mask), a short
video or the recorded frames (`ward lab --record-frames`) helps enormously.

We aim to acknowledge within a few days and to fix confirmed, in-scope issues before any public
disclosure. There is no bug-bounty program.

## The guarantee that must never break: fail open

Every authentication failure — a dark or busy camera, no match, a suspected spoof, a timeout, a
crash in the engine, a stuck daemon — **must fall through to the password prompt**. A bug that locks
a user out is treated as the most serious class of defect here, ahead of any false-accept concern.
The PAM module enforces its own deadline and never returns "authenticated" on an error path. If you
find a way to make ward *deny* a login rather than fall back to the password, that is in scope and
important.

## Threat model

Ward is **strong convenience with a password fallback**, in the same class as Windows Hello on the
same 2D camera — **not** the class of Face ID (which has a depth sensor) or a hardware security key.

**In scope** (report these):
- A path where authentication does not fail open (a lockout, or auth granted on an error).
- Privilege escalation: anything that runs attacker-controlled code as root through the PAM module,
  the `ward-auth` helper, the resident daemon (`ward.service`), the `pkexec` paths used by the
  app, or `install.sh`/packaging.
- The resident daemon's socket accepting a request from a non-root process, or one attempt's state
  leaking into another.
- Secret handling: the sealed keyring credential, `PAM_AUTHTOK`, or the model store (0600) being
  exposed to a non-root user.
- A **presentation attack that also defeats the full stack** — see below for what each layer does.

**Out of scope** (known, documented limits of a 2D sensor):
- Identical twins and strong look-alikes. No 2D face system, Windows Hello included, stops these.
  Mitigate with `require_rgb_match`, a tight `certainty`, and measuring the relative in `ward lab`.
- A **high-quality printed IR photograph**: a 2D system cannot fully distinguish it. The anti-spoof
  layers raise the bar but do not claim to defeat a determined, well-resourced attacker.
- Coercion (being forced to look at the camera). Face unlock cannot defend against this; use the
  password/lock-out, and keep the password fallback strong.
- A false-accept **rate** (FAR). Ward does **not** yet claim a calibrated FAR — that needs a lab
  study with other subjects (roadmap). Do not treat it as certified.

## How the layers actually work (so a "bypass" is assessed correctly)

A single trick that fools one layer is usually not a system bypass:

- **Recogniser** decides *who*, not *whether it's real*. Fooling it needs a look-alike or a photo of
  the enrolled person — and that photo still has to pass liveness.
- **Cross-spectrum (IR)** is the real anti-replay: a phone/monitor emits ~no infrared, so a replayed
  face is absent from the IR frame and the attempt FAILs. This is why ward uses an IR camera.
- **Anti-spoof CNN** catches the *physical* artifacts of a camera filming a screen or print (moiré,
  reflection, lack of depth). It does **not** flag a digitally re-encoded image — verified by
  `ward benchmark redteam` — so it is a complement to the IR layer, not a replacement.
- A suspected spoof (a liveness FAIL from cross-spectrum / shadow / CNN) **ends the attempt** and
  feeds the spoof lock-out; repeated spoof attempts pause face unlock (the password still works).

You can measure these yourself: `ward benchmark lfw` (impostor rate, recogniser),
`ward benchmark spoof` (APCER/BPCER, CNN), `ward benchmark ir` (cross-spectrum PAD), and
`ward benchmark redteam` (digital attacks on your own frames).

## Supported versions

Ward is pre-1.0; only the latest released version receives security fixes.
