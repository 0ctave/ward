#!/usr/bin/env bash
# Install faceup system-wide using the distro Python (the one Howdy and dlib live in).
#   sudo ./install.sh            install / upgrade
#   sudo ./install.sh --uninstall
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run with sudo"; exit 1; }
LIB=/usr/local/lib/faceup
BIN=/usr/local/bin/faceup
if [[ "${1:-}" == "--uninstall" ]]; then
  rm -rf "$LIB" "$BIN"; echo "faceup removed"; exit 0
fi
here=$(cd "$(dirname "$0")" && pwd)
python3 -c 'import cv2, dlib' 2>/dev/null || echo "warning: python3-opencv / python3-dlib missing — install Howdy first"
rm -rf "$LIB"; mkdir -p "$LIB"
cp -r "$here/faceup" "$LIB/"
find "$LIB" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
cat > "$BIN" <<EOF
#!/bin/sh
export PYTHONPATH="$LIB\${PYTHONPATH:+:\$PYTHONPATH}"
exec /usr/bin/python3 -m faceup "\$@"
EOF
chmod 755 "$BIN"
echo "installed: $BIN  (package in $LIB)"
echo "next:  faceup doctor   ·   sudo faceup setup"
