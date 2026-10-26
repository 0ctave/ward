#!/usr/bin/env bash
# Record a FACE-FREE terminal demo of Ward: a sudo prompt unlocking by face, shown only in the
# terminal (the camera does its work off-screen). You sit in front of the laptop for recognition;
# nothing identifying is recorded. Produces a GIF for the README.
#
#   docs/assets/record_demo.sh                 # → docs/demo.gif (asciinema + agg)
#   docs/assets/record_demo.sh --out /tmp/x.gif --cols 90 --rows 16
#
# Requires Ward already enabled for sudo (`sudo ward pam enable`). Tools, either set:
#   asciinema + agg   (default, clean terminal render — recommended)
#   wf-recorder + gifski  (--tool screen: actually screen-records the terminal window, Wayland)
# Re-exec under bash if invoked as `sh record_demo.sh` (dash has no `set -o pipefail`).
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); root=$(cd "$here/../.." && pwd)
out="$root/docs/demo.gif"; tool=asciinema; cols=92; rows=18; speed=1.0
while [ $# -gt 0 ]; do case "$1" in
  --out) out=$2; shift 2;; --tool) tool=$2; shift 2;; --cols) cols=$2; shift 2;;
  --rows) rows=$2; shift 2;; --speed) speed=$2; shift 2;;
  -h|--help) sed -n '2,13p' "$0"; exit 0;; *) echo "unknown arg: $1"; exit 2;; esac; done

have() { command -v "$1" >/dev/null 2>&1; }
hint() { echo "  install: $1"; }

# The scripted session that gets recorded. Types each command, then runs it so the Ward feedback
# line ("Looking for your face…" → "Face recognised") is captured. Edit freely.
demo=$(mktemp); trap 'rm -f "$demo" "${cast:-}"' EXIT
cat > "$demo" <<'DEMO'
set -e
show() { printf '\033[1;32m$\033[0m '; for c in $(echo "$1" | sed 's/./& /g'); do printf '%s' "$c"; sleep 0.04; done; printf '\n'; }
clear
echo "# Ward — face unlock for Linux. sudo without typing a password:"; echo; sleep 1
sudo -k                                   # forget any cached sudo
show "sudo whoami"; sudo whoami; echo; sleep 1.2
show "sudo systemctl status ward --no-pager | head -1"; sudo systemctl status ward --no-pager | head -1 || true; echo; sleep 1
echo "# A failed face check (or covering the camera) just falls back to your password."; sleep 2
DEMO

if [ "$tool" = asciinema ]; then
  if ! have asciinema || ! have agg; then
    echo "need asciinema and agg (asciinema's gif generator):"
    have asciinema || hint "asciinema  — apt/dnf/pacman install asciinema, or pipx install asciinema"
    have agg || hint "agg        — cargo install agg   (https://github.com/asciinema/agg)"
    exit 1
  fi
  cast=$(mktemp --suffix .cast)
  echo ">>> Recording. Sit in front of the camera; Ward will recognise you during 'sudo'. Ctrl-D ends early."
  asciinema rec --overwrite --cols "$cols" --rows "$rows" -c "bash $demo" "$cast"
  agg --cols "$cols" --rows "$rows" --speed "$speed" --theme asciinema "$cast" "$out"
  echo ">>> wrote $out"
elif [ "$tool" = screen ]; then
  if ! have wf-recorder || ! have gifski; then
    echo "need wf-recorder (Wayland screen capture) and gifski:"
    have wf-recorder || hint "wf-recorder — your distro's package"
    have gifski || hint "gifski      — cargo install gifski, or your distro's package"
    exit 1
  fi
  echo ">>> Screen-recording this terminal window for 15s. Keep it focused; sit in front of the camera."
  mp4=$(mktemp --suffix .mp4)
  ( sleep 0.5; bash "$demo" ) & wf-recorder -f "$mp4" & rec=$!
  sleep 15; kill "$rec" 2>/dev/null || true; wait 2>/dev/null || true
  gifski -o "$out" --fps 15 --width $((cols * 9)) "$mp4"; rm -f "$mp4"
  echo ">>> wrote $out"
else
  echo "unknown --tool: $tool (asciinema | screen)"; exit 2
fi

echo
echo "Next: preview $out, then add it to the README hero and commit:"
echo "  git add $out && git commit -m 'docs: demo gif' && git push"
