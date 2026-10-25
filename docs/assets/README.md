# Demo assets

Regenerate the README's visuals. All of this is **face-free** — no camera feed, no personal data.

## App screenshots (headless, no camera)

```bash
python3 docs/assets/screenshots.py                 # → docs/assets/app-*.png (dark theme)
python3 docs/assets/screenshots.py --theme light --pages overview,reliability
```
Renders the real `ward.app` window with invented sample data and snapshots each page. Only pages
without a camera preview are available (`overview`, `reliability`, `settings`, `advanced`).
Re-run and commit the PNGs after UI changes.

## Terminal demo GIF (you sit in front of the camera)

```bash
docs/assets/record_demo.sh                          # → docs/demo.gif (asciinema + agg)
docs/assets/record_demo.sh --tool screen            # wf-recorder + gifski (actual screen capture)
```
Records a `sudo` prompt unlocking by face **in the terminal only** — the camera recognises you
off-screen, so nothing identifying is captured. Requires Ward enabled for sudo
(`sudo ward pam enable`). Edit the scripted commands inside the script to taste.
