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
  # every PAM file, not just common-auth (Fedora/Arch: system-auth; hand edits: sudo, gdm-password…)
  refs=$(grep -lE 'faceup-auth|pam_faceup\.so' /etc/pam.d/* 2>/dev/null | grep -v '\.faceup-bak$' || true)
  if [[ -n "$refs" ]]; then
    echo "faceup is still referenced in: $refs — run 'sudo faceup pam disable' (and edit any hand-made lines) first"; exit 1
  fi
  systemctl disable --now faceup.service 2>/dev/null || true
  rm -rf "$LIB" "$BIN" "$AUTH" /etc/systemd/system/faceup.service /usr/local/lib/security/pam_faceup.so /usr/share/applications/faceup.desktop /usr/share/polkit-1/actions/org.faceup.policy \
         /usr/share/icons/hicolor/scalable/apps/faceup.svg /usr/share/icons/hicolor/symbolic/apps/faceup-symbolic.svg
  command -v gtk-update-icon-cache >/dev/null && gtk-update-icon-cache -q /usr/share/icons/hicolor || true
  echo "faceup removed (config/models kept in /etc/faceup)"; exit 0
fi
here=$(cd "$(dirname "$0")" && pwd)
hint() { PYTHONPATH="$here" /usr/bin/python3 -m faceup.distro "$@" 2>/dev/null || echo "$*"; }
/usr/bin/python3 -c 'import cv2, numpy' 2>/dev/null || echo "warning: OpenCV/numpy missing for /usr/bin/python3 — $(hint opencv)"
/usr/bin/python3 -c 'import dlib' 2>/dev/null || echo "note: dlib missing (the ArcFace recogniser works without it) — $(hint dlib dlib-models)"
/usr/bin/python3 -c 'import gi; gi.require_version("Gtk","4.0"); gi.require_version("Adw","1")' 2>/dev/null \
  || echo "note: GTK 4 + libadwaita bindings missing: windows fall back to text — $(hint gtk)"
rm -rf "$LIB"; mkdir -p "$LIB"
cp -r "$here/faceup" "$LIB/"
find "$LIB" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
# byte-compile now so the first faceup-auth on the PAM path doesn't pay for it
/usr/bin/python3 -m compileall -q -j0 "$LIB" >/dev/null 2>&1 || true
# Both launchers run as root (sudo, pkexec, PAM): python3 -I ignores every PYTHON* variable, the
# user's site-packages and the cwd, so nothing in the caller's environment can load code; the
# package path is set inside instead of through PYTHONPATH.
launcher() {
  cat <<EOF
#!/bin/sh
exec /usr/bin/python3 -I -c 'import runpy, sys; sys.path.insert(0, "$LIB"); sys.argv[0] = "$2"; runpy.run_module("$1", run_name="__main__")' "\$@"
EOF
}
launcher faceup faceup > "$BIN"
launcher faceup.client faceup-auth > "$AUTH"
chmod 755 "$BIN" "$AUTH"
# native PAM module (keyring token + conversation messages); pam_exec stays the fallback
if command -v gcc >/dev/null 2>&1; then
  mkdir -p /usr/local/lib/security
  if gcc -Os -s -fPIC -shared -ffunction-sections -fdata-sections -Wl,--gc-sections -Wl,--as-needed -o /usr/local/lib/security/pam_faceup.so.tmp "$here/data/pam_faceup.c" 2>/tmp/pam_faceup.build.log; then
    chmod 644 /usr/local/lib/security/pam_faceup.so.tmp
    mv /usr/local/lib/security/pam_faceup.so.tmp /usr/local/lib/security/pam_faceup.so
    echo "built /usr/local/lib/security/pam_faceup.so (native PAM module)"
  else
    rm -f /usr/local/lib/security/pam_faceup.so.tmp; echo "warning: pam_faceup.so did not build (see /tmp/pam_faceup.build.log); pam_exec line stays"
  fi
else
  echo "note: gcc not found — the native PAM module (keyring unlock) was not built; $(hint gcc) and re-run"
fi
# desktop entry + icons: notifications get the app name and icon, the DE can group them (roadmap U7)
install -Dm644 "$here/data/faceup.desktop" /usr/share/applications/faceup.desktop
install -Dm644 "$here/data/org.faceup.policy" /usr/share/polkit-1/actions/org.faceup.policy   # launcher → pkexec faceup app
install -Dm644 "$here/data/faceup.svg" /usr/share/icons/hicolor/scalable/apps/faceup.svg
install -Dm644 "$here/data/faceup-symbolic.svg" /usr/share/icons/hicolor/symbolic/apps/faceup-symbolic.svg
command -v gtk-update-icon-cache >/dev/null && gtk-update-icon-cache -q /usr/share/icons/hicolor || true
command -v update-desktop-database >/dev/null && update-desktop-database -q /usr/share/applications || true
mkdir -p /etc/faceup/models; chmod 700 /etc/faceup/models
touch /var/log/faceup.log; chmod 640 /var/log/faceup.log; chgrp adm /var/log/faceup.log 2>/dev/null || chgrp wheel /var/log/faceup.log 2>/dev/null || true
# SELinux (Fedora/RHEL): files copied from a home directory keep user_home_t, which confined
# PAM callers (gdm, polkit) may not load or execute — give them the default system labels
if command -v selinuxenabled >/dev/null 2>&1 && selinuxenabled && command -v restorecon >/dev/null 2>&1; then
  restorecon -RF "$LIB" /usr/local/lib/security "$BIN" "$AUTH" /etc/faceup /var/log/faceup.log 2>/dev/null || true
  echo "SELinux: restored default labels"
fi
# resident engine (faceup/daemon.py): models stay loaded, an unlock skips ~1 s of start-up.
# On by default on first install; `sudo faceup daemon disable` turns it off and stays off.
if command -v systemctl >/dev/null 2>&1 && [[ -d /run/systemd/system ]]; then
  first=0; [[ -e /etc/systemd/system/faceup.service ]] || first=1
  install -Dm644 "$here/data/faceup.service" /etc/systemd/system/faceup.service
  systemctl daemon-reload
  if [[ $first == 1 ]]; then systemctl enable --now faceup.service && echo "resident engine enabled (fast unlock)"
  elif systemctl is-enabled --quiet faceup.service; then systemctl restart faceup.service && echo "resident engine restarted"; fi
fi
echo "installed: $BIN, $AUTH  (package in $LIB)"
echo "next:  faceup doctor   ·   sudo faceup setup"
