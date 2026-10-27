# Migrating from Howdy

[← back to the README](../README.md)

[Howdy](https://github.com/boltgolt/howdy) pioneered Windows-Hello-style face login on Linux and is
pleasant when it works. Ward is a from-scratch alternative that targets the parts that make Howdy
unreliable in practice: finding the right camera, lighting the infrared emitter, enrolling more than
one pose, measuring the match threshold from *your* face, and adding real liveness — while running on
your distro's own Python with no pip. This guide moves you across without locking you out.

- [Why switch](#why-switch)
- [1. Install Ward alongside Howdy](#1-install-ward-alongside-howdy)
- [2. Let Ward retire Howdy for you](#2-let-ward-retire-howdy-for-you)
- [3. Verify](#3-verify)
- [4. Fully remove Howdy (optional)](#4-fully-remove-howdy-optional)
- [Rollback](#rollback)

## Why switch

Howdy matches a face against a fixed dlib distance threshold from a single enrolled capture, and
leaves the camera, the IR emitter and the PAM wiring for you to configure by hand. Its own README is
honest that it has **no liveness** ("a well-printed photo of you could be enough", "DO NOT USE HOWDY
AS THE SOLE AUTHENTICATION METHOD"). On current distros its dlib dependency is also fragile — the
wiki's fix for "dlib not found" is `pip install --break-system-packages dlib`, which it warns can
damage the system Python.

Ward keeps the good part — the Windows-Hello-class convenience for login, `sudo` and polkit — and
fixes the setup around it:

- **Finds the right camera** by probing the stream's pixel format, not guessing by name.
- **Lights the IR emitter** by measurement and re-applies it at boot and after suspend, so it works
  in the dark.
- **Enrolls many poses in one guided sweep** instead of one still capture.
- **Measures the threshold from held-out frames of your own face** instead of shipping a default.
- **Adds liveness** — the IR cross-spectrum check, an anti-spoof CNN, motion, and an optional emitter
  shadow test.
- **Runs on distro Python, zero pip** — no `--break-system-packages`.

See [How it works](how-it-works.md) for the mechanism behind each, and the
[Ward vs Howdy table](../README.md#ward-vs-howdy) for a row-by-row comparison. Ward is still a **2D**
system — strong convenience with a password fallback, the class of Windows Hello on the same
hardware, [not Face ID](how-it-works.md#honest-limits).

## 1. Install Ward alongside Howdy

You don't have to remove Howdy first — Ward installs beside it and shares the same dlib/OpenCV. Follow
the [Install guide](install.md): install the dependencies for your distro, then

```bash
git clone https://github.com/0ctave/ward && cd ward
sudo ./install.sh
ward doctor                # run this first — cameras, emitter, recogniser, PAM
```

`install.sh` **never edits your PAM config** — Howdy keeps authenticating exactly as before until you
turn Ward on in the next step. Ward does **not** import Howdy's enrolled models: Howdy enrolls from
RGB frames and Ward authenticates against infrared, so the embeddings don't transfer — you'll
re-enroll your face once, which is the point of the guided multi-pose sweep.

Then set Ward up (the app, or `sudo ward setup` on a headless box):

```bash
ward                       # opens the app → "Set up Ward → Start setup"
# or, in a terminal / over SSH:
sudo ward setup
```

## 2. Let Ward retire Howdy for you

Ward wires itself into the shared PAM auth stack with a single line, keeping a backup
(`<stack>.ward-bak`). When it goes in — through the app's switch, the setup wizard's **PAM** step, or
the CLI — it notices an active `pam_howdy` line and offers to retire it, because two face steps in a
row are pointless:

```bash
sudo ward pam enable
# → enabled ward in /etc/pam.d/common-auth via … (backup: …/common-auth.ward-bak)
# → pam_howdy is also active. Disable it (two face steps are pointless)? [Y/n]
```

Answering yes **comments out** the `pam_howdy` lines (it does not uninstall anything), so Ward is now
the only face step. The backup captures the file as it was before Ward touched it. `ward doctor`
flags a still-active `pam_howdy` line if you skipped the prompt, and `sudo ward pam enable` offers it
again.

> [!NOTE]
> A face *login* can leave your GNOME Keyring / KWallet locked, because they're sealed with your
> login password. Turn on Ward's keyring unlock with `sudo ward keyring enable`. If your keyring was
> already left locked by a past experiment that commented out `pam_gnome_keyring` / `pam_kwallet`,
> `sudo ward keyring repair` uncomments those lines. See [How it works → keyring](how-it-works.md#7-unlocking-the-keyring).

## 3. Verify

```bash
sudo ward pam status       # file, whether ward (and pam_howdy) are active, backup present
ward doctor                # full health check with a fix command for anything wrong
sudo -k && sudo true       # should accept by face; a failure falls through to the password
ward log -n 5              # the last few attempts: distances, liveness votes, timing, service
```

`ward log --summary` gives an accept rate and median time-to-accept once you've used it a few times.
If a face unlock never triggers at the greeter or lock screen, check `ward pam status` and the
[Troubleshooting table](install.md#troubleshooting).

## 4. Fully remove Howdy (optional)

Once Ward is accepting you reliably, you can uninstall Howdy entirely — it's no longer on the auth
path. Remove the package the way you installed it, for example:

```bash
sudo apt remove howdy               # Debian/Ubuntu (PPA)
sudo dnf remove howdy               # Fedora (COPR)
sudo pacman -Rns howdy              # Arch (AUR)
```

Package removal leaves its config and enrolled models behind; delete those too if you want a clean
slate:

```bash
sudo rm -rf /etc/howdy              # config.ini and RGB face models
```

If you had Ward comment out the `pam_howdy` line rather than remove the package, uninstalling Howdy
also removes `pam_howdy.so`; the commented line is then inert, and `sudo ward pam enable` /
`ward doctor` will no longer mention it.

## Rollback

Ward is built to never lock you out, so backing out is safe at every step:

- **Every failed attempt already falls through to your password** — the hard rule of the engine, with
  a kernel SIGALRM backstop if anything wedges. You are never stuck at a face prompt.
- **Turn Ward off** and keep Howdy: `sudo ward pam disable` removes Ward's line from the auth stack.
  If you'd had Ward comment out `pam_howdy`, re-enable Howdy by restoring the backup
  (`sudo cp /etc/pam.d/common-auth.ward-bak /etc/pam.d/common-auth`, or the `system-auth` backup on
  Fedora/Arch) or by uncommenting the `#…pam_howdy` line.
- **Remove Ward entirely**: `sudo ward pam disable`, then `sudo ./install.sh --uninstall` (this keeps
  `/etc/ward` with your models and config; `sudo rm -rf /etc/ward` removes those too). Distro-package
  removal is refused while `/etc/pam.d` still references Ward, so disable PAM first.
