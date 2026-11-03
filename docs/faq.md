# Ward — FAQ

[← back to the README](../README.md) · Didn't find it here? Ask in
[Discussions](https://github.com/0ctave/ward/discussions).

### Is this as secure as Apple Face ID or Windows Hello with a secure enclave?

No. Ward is a **2D** camera system with no depth sensor or secure enclave — **strong convenience with
a password fallback**, in the class of Windows Hello's camera login on the same hardware, not Face ID.
It defeats phone/monitor replays (via infrared) and darkness; it does not claim to defeat a
high-quality printed infrared photo, and identical twins defeat any 2D system. See
[SECURITY.md](../SECURITY.md) for the full threat model.

### Can Ward lock me out?

No. **Every** failure — no match, a dark or busy camera, a suspected spoof, a timeout, a bug — falls
through to your password, with a hard timeout backstop. That is a golden rule of the project; the
password path is never removed.

### Does it send my face anywhere? Are images stored?

No network at runtime and no telemetry — everything is local. Only **embeddings** (number vectors) are
stored, never images, in `/etc/ward/models/<user>.json` (mode 0600, root). Remove a person's data with
`sudo ward models --wipe`.

### Does it work in the dark?

Yes, if your laptop has a **Windows Hello infrared camera** — Ward lights the IR emitter and
recognises you by infrared. A plain colour webcam works in normal light but has no IR liveness and no
night unlock.

### I don't have an IR camera / can I use an external USB webcam?

A colour-only webcam (built-in or USB) works for recognition plus the anti-spoof CNN — you lose
infrared liveness and night unlock. External USB IR modules usually work too; run `ward doctor`, and
if an IR stream isn't detected see [Troubleshooting](install.md#troubleshooting).

### It stopped working after a suspend/resume or a reboot.

The emitter is re-applied at boot and resume, so this is usually a camera that came back busy or
renumbered. `ward doctor` diagnoses it; Ward stores cameras by stable USB id (not `/dev/videoN`) so
renumbering is handled. If unlock feels slow, `sudo ward daemon enable` keeps the models warm.

### Will it survive an OpenCV or kernel update?

The package depends on your distro's `python3-opencv`, so a normal update keeps a compatible build,
and Ward bundles both YuNet detector graphs (2022 + 2023) and picks by OpenCV version. After a big
system upgrade, run `ward doctor` — it checks every part and prints the fix for anything that moved.

### My keyring is locked after a face login.

A face accept produces no password, so GNOME Keyring / KWallet stay locked at the *first* login of a
session unless you opt in: `sudo ward keyring enable` (and `sudo ward keyring repair` if an old config
disabled the keyring PAM lines). Screen unlock, sudo and polkit are unaffected. See
[Usage → keyring](usage.md).

### Can more than one person use the same login? What about glasses or a headset?

Yes — register others on your login with `ward enroll --identity "Name"`, and add a look with
`ward enroll --add glasses` (or `headset`, `hat`, …). See [Usage](usage.md).

### Which recogniser should I use — dlib or ArcFace?

On **Fedora, Ubuntu and Debian** `python3-dlib` isn't packaged and Ward never uses pip, so `ward setup`
installs the bundled **ArcFace** model automatically. **openSUSE Tumbleweed** packages dlib. Either
works; `ward doctor` tells you which you have. See [Pick a recogniser](install.md#pick-a-recogniser).

### Does it replace Howdy? Can I run both?

Ward replaces Howdy with its own engine and coexists while you migrate — it can retire Howdy for you
during setup. See [Migrating from Howdy](migrating-from-howdy.md).

### How do I remove Ward and my face data?

`sudo ward pam disable`, then `sudo apt/dnf/pacman remove ward` (or `sudo ./install.sh --uninstall`
from source). Your models/config in `/etc/ward` are kept unless you also remove that directory. See
[Uninstall](install.md#uninstall).
