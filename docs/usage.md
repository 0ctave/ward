# Using Ward

[← back to the README](../README.md)

Two ways to drive Ward: the **app** (a native window) or the **command line**. They're the same
thing — the app runs the matching `ward …` command for every action and streams its output into a
console pane, so they can never disagree, and everything works headless.

- [The app](#the-app)
- [Everyday tasks](#everyday-tasks)
- [Headless / SSH](#headless--ssh)
- [Command reference](#command-reference)

## The app

Open **Ward** from your applications, or run `ward`. It runs as **you**; anything that changes the
system (register a face, enable PAM, the keyring) asks for the admin password through the standard
prompt. Five pages:

- **Overview** — an on/off switch for face unlock, a reliability verdict, a *Test unlock* button, and
  anything that needs fixing. On first run it leads with **Set up Ward → Start setup**.
- **People** — you and anyone else who can unlock this login: register, re-enrol, add
  glasses/headset conditions, remove.
- **Reliability** — success rate by light, service and person; why attempts failed; the recent log.
- **Settings** — security policy, behaviour, keyring, the fast resident engine.
- **Advanced** — the setup wizard, the lab, maintenance.

Registering and calibrating happen **inside** the window — the camera view, the target ring and the
guidance arrow are drawn right there, no separate pop-up.

## Everyday tasks

Each of these is a button in the app *and* a command. The commands:

**Register your face** — one guided sweep that also measures your recognition threshold:
```bash
sudo ward enroll --fresh
```

**Add another person** to this login (any registered face unlocks; the log records `who=`):
```bash
sudo ward enroll --identity "Alex"
```

**Glasses, a headset, the dark** — a short extra sweep that adds the condition without touching the
rest of your models:
```bash
sudo ward enroll --add glasses
ward models                              # list identities and conditions
sudo ward models --remove-condition glasses
```

**Unlock your keyring** after a face login (GNOME Keyring / KWallet), by sealing your login password:
```bash
sudo ward keyring enable                 # host key on an encrypted disk, or TPM2
```

**Faster unlocks** — the resident engine keeps the models loaded:
```bash
sudo ward daemon enable                  # on by default; `disable` loads per-attempt instead
```

**Tighten or loosen security** — presets only ever *tighten* your measured gate:
```bash
sudo ward tune policy convenience|balanced|strict
# per service: security.policy_sudo / policy_polkit / policy_login / policy_unlock  ("off" = password only there)
```
`balanced` (default) makes the colour camera agree whenever it sees you well. `strict` requires a
liveness PASS and turns learning off. Three suspected-spoof failures in 15 minutes pause face unlock
for 15 minutes — the password always works.

**See what's happening:**
```bash
ward log -n 10            # every attempt: distances, liveness votes, time, why it fell back
ward log --summary        # accept rate, median/p90 time, reasons, per service
ward doctor               # the full health check, with a fix for each issue
```

## Headless / SSH

Everything works without a display. The camera must be on the machine you're setting up (Ward reads
`/dev/video*` locally), but you can drive it over SSH.

```bash
sudo ward setup                 # the full chain in the terminal; registration falls back to a text
                                # pose flow automatically when there's no display
sudo ward enroll --fresh --guided   # or force the text pose flow directly
sudo ward calibrate --text          # thresholds in the terminal
sudo ward lab --text                # the reliability bench, text mode
```

Running bare `ward` on a headless box tries the app, finds no GTK, and prints the CLI commands to use.

## Command reference

`*` = root needed for the action but not for `status`.

| Command | Root | What it does |
|---|---|---|
| `ward` / `ward app` | no | The manager window (runs as you; changes go through the admin prompt). |
| `ward doctor` | no | Full health check: cameras, live lighting, emitter, engine, models, resident engine, PAM. Exit 1 if anything needs fixing. |
| `ward setup` | yes | The guided chain: cameras → emitter → recogniser → register → keyring → PAM. |
| `ward enroll [--identity N] [--add C]` | yes | One guided sweep that registers AND sets the gate. `--identity` adds a person; `--add glasses` adds a condition; `--guided` text flow; `--classic` old coverage-map. |
| `ward calibrate [session\|refresh\|pose]` | yes | Re-measure the gate from guided passes (`session`), from recent unlocks (`refresh`), or rebuild your 3D face model (`pose`). `--text` for the terminal. |
| `ward models [--remove-identity N \| --remove-condition C \| --forget-learned \| --json]` | yes | List or remove identities, conditions and learned models. |
| `ward pam enable\|disable\|status` | yes | The PAM line, with backup/restore; offers to retire `pam_howdy`. |
| `ward keyring status\|enable\|disable\|repair` | yes* | Seal your login password so a face login unlocks the keyring; `repair` restores keyring PAM lines an old experiment disabled. |
| `ward daemon status\|enable\|disable` | yes* | The resident engine (`ward.service`) that keeps the models loaded for faster unlocks. |
| `ward emitter probe\|toggle-bench\|check` | yes | Find the IR LED control by measurement, benchmark its switch (enables the shadow liveness test), or view IR/RGB side by side. |
| `ward recogniser status\|install [mbf\|r50]\|use dlib\|arcface` | yes* | The embedding network: dlib (distro) or ArcFace (downloaded, pinned by checksum). |
| `ward antispoof status\|on\|off\|test DIR` | yes* | The anti-spoof CNN voter (bundled MiniFASNetV2); `test` scores a folder of frames. |
| `ward tune [section.key VALUE \| policy LEVEL]` | yes | Show or change the config (validated); set the security preset. |
| `ward log [--summary\|--follow\|--json]` | no* | Recent attempts; summary rates and latency; live follow. |
| `ward lab` | yes | Live evaluation window: distances, votes and three policies side by side; record subjects to CSV. `--text` for the terminal. |
| `ward bench` | no | Time the detectors on a live frame. |
| `ward benchmark lfw\|spoof\|ir\|redteam` | no | Offline accuracy on external datasets (LFW, CelebA-Spoof, CASIA-SURF/WMCA) and a red-team of digital attacks on your own frames. Datasets are research-only, never bundled. |
| `ward update` | no | Check GitHub for a newer release (explicit; never runs during an unlock). |

Add `--help` to any command for its full options.
