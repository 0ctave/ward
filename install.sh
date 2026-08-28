#!/usr/bin/env bash
# Install faceup system-wide using the distro Python (the one Howdy, dlib and OpenCV live in).
#   sudo ./install.sh              install / upgrade
#   sudo ./install.sh --uninstall  remove (keeps /etc/faceup; disable PAM first: sudo faceup pam disable)
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run with sudo"; exit 1; }
LIB=/usr/local/lib/faceup
BIN=/usr/local/bin/faceup
AUTH=/usr/local/bin/faceup-auth
if [[ "${1:-}" == "--uninstall" ]]; then
  if grep -q faceup-auth /etc/pam.d/common-auth 2>/dev/null; then
    echo "faceup-auth is still referenced in /etc/pam.d/common-auth — run 'sudo faceup pam disable' first"; exit 1
  fi
  rm -rf "$LIB" "$BIN" "$AUTH"; echo "faceup removed (config/models kept in /etc/faceup)"; exit 0
fi
here=$(cd "$(dirname "$0")" && pwd)
/usr/bin/python3 -c 'import cv2, dlib, numpy' 2>/dev/null || echo "warning: python3-opencv / python3-dlib / python3-numpy missing (apt install python3-opencv python3-dlib dlib-models)"
rm -rf "$LIB"; mkdir -p "$LIB"
cp -r "$here/faceup" "$LIB/"
find "$LIB" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
cat > "$BIN" <<EOF
#!/bin/sh
export PYTHONPATH="$LIB\${PYTHONPATH:+:\$PYTHONPATH}"
exec /usr/bin/python3 -m faceup "\$@"
EOF
# The PAM entry point: fixed interpreter, fixed path, no user-controlled environment.
cat > "$AUTH" <<EOF
#!/bin/sh
export PYTHONPATH="$LIB"
exec /usr/bin/python3 -m faceup auth "\$@"
EOF
chmod 755 "$BIN" "$AUTH"
mkdir -p /etc/faceup/models; chmod 700 /etc/faceup/models
touch /var/log/faceup.log; chmod 640 /var/log/faceup.log; chgrp adm /var/log/faceup.log 2>/dev/null || true
echo "installed: $BIN, $AUTH  (package in $LIB)"
echo "next:  faceup doctor   ·   sudo faceup setup"
