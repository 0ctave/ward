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
first=0; [[ -e /etc/systemd/system/faceup.service ]] || first=1
# the file layout is shared with the distro packages (packaging/): package, launchers, PAM module,
# desktop entry, polkit action, icons, unit
"$here/packaging/stage.sh" / /usr/local /etc/systemd/system
[[ -e /usr/local/lib/security/pam_faceup.so ]] || echo "note: no native PAM module (keyring unlock) — $(hint gcc) and re-run"
# byte-compile now so the first faceup-auth on the PAM path doesn't pay for it
/usr/bin/python3 -m compileall -q -j0 "$LIB" >/dev/null 2>&1 || true
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
  systemctl daemon-reload
  systemd-analyze verify /etc/systemd/system/faceup.service 2>&1 | grep -v '^$' || true
  # Type=notify: start/restart return only once the models are warm (READY=1), so a sandbox the
  # engine cannot run in shows up here. Unlock never depends on it (the client falls back).
  daemon_fail="  ! resident engine did not start — unlock still works (in-process). See: journalctl -u faceup -n 30"
  if [[ $first == 1 ]]; then
    if systemctl enable --now faceup.service; then echo "resident engine enabled (fast unlock)"; else echo "$daemon_fail"; fi
  elif systemctl is-enabled --quiet faceup.service; then
    if systemctl restart faceup.service; then echo "resident engine restarted"; else echo "$daemon_fail"; fi
  fi
fi
echo "installed: $BIN, $AUTH  (package in $LIB)"
echo "next:  faceup doctor   ·   sudo faceup setup"
