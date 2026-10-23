#!/usr/bin/env bash
# Shared install layout for install.sh (from source) and the distro packages.
#   packaging/stage.sh DESTDIR PREFIX UNITDIR
#   e.g.  stage.sh /            /usr/local /etc/systemd/system     (install.sh)
#         stage.sh "$pkgdir"    /usr       /usr/lib/systemd/system  (deb / rpm / Arch)
# Copies the package, writes the two root launchers, builds pam_faceup.so (when gcc exists) and
# installs the desktop entry, polkit action, icons and systemd unit with PREFIX substituted.
# Touches nothing outside DESTDIR; no PAM, no systemctl, no /etc/faceup — callers do that.
set -euo pipefail
umask 022
dest=${1:?DESTDIR}; prefix=${2:?PREFIX}; unitdir=${3:?UNITDIR}
dest=${dest%/}
src=$(cd "$(dirname "$0")/.." && pwd)
lib=$prefix/lib/faceup
rm -rf "$dest$lib"; mkdir -p "$dest$lib" "$dest$prefix/bin"
cp -r "$src/faceup" "$dest$lib/"; install -m644 "$src/README.md" "$dest$lib/README.md"
find "$dest$lib" -name '__pycache__' -type d -prune -exec rm -rf {} +
# Both launchers run as root (sudo, pkexec, PAM): python3 -I ignores every PYTHON* variable, the
# user's site-packages and the cwd, so nothing in the caller's environment can load code; the
# package path is set inside instead of through PYTHONPATH.
launcher() {
  cat <<EOS
#!/bin/sh
exec /usr/bin/python3 -I -c 'import runpy, sys; sys.path.insert(0, "$lib"); sys.argv[0] = "$2"; runpy.run_module("$1", run_name="__main__")' "\$@"
EOS
}
launcher faceup faceup > "$dest$prefix/bin/faceup"
launcher faceup.client faceup-auth > "$dest$prefix/bin/faceup-auth"
chmod 755 "$dest$prefix/bin/faceup" "$dest$prefix/bin/faceup-auth"
# native PAM module (keyring token + conversation messages); pam_exec stays the fallback
secdir=$dest$prefix/lib/security
if command -v "${CC:-gcc}" >/dev/null 2>&1; then
  mkdir -p "$secdir"
  log=$(mktemp)
  # shellcheck disable=SC2086  # LDFLAGS is a flag list
  if "${CC:-gcc}" -Os -s -fPIC -shared -ffunction-sections -fdata-sections -Wl,--gc-sections -Wl,--as-needed \
       ${LDFLAGS:-} -o "$secdir/pam_faceup.so.tmp" "$src/data/pam_faceup.c" 2>"$log"; then
    chmod 644 "$secdir/pam_faceup.so.tmp"; mv "$secdir/pam_faceup.so.tmp" "$secdir/pam_faceup.so"
    echo "built $prefix/lib/security/pam_faceup.so (native PAM module)"
  else
    rm -f "$secdir/pam_faceup.so.tmp"; cat "$log" >&2
    echo "warning: pam_faceup.so did not build; the pam_exec line stays" >&2
    [[ -z "${FACEUP_REQUIRE_PAM_MODULE:-}" ]] || exit 1
  fi
  rm -f "$log"
else
  echo "note: gcc not found — the native PAM module (keyring unlock) was not built" >&2
  [[ -z "${FACEUP_REQUIRE_PAM_MODULE:-}" ]] || exit 1
fi
sub() { sed "s#/usr/local/#$prefix/#g" "$1"; }
put() { install -d "$(dirname "$dest$2")"; sub "$src/data/$1" > "$dest$2"; chmod 644 "$dest$2"; }
put faceup.desktop /usr/share/applications/faceup.desktop
put org.faceup.policy /usr/share/polkit-1/actions/org.faceup.policy     # launcher → pkexec faceup app
put faceup.svg /usr/share/icons/hicolor/scalable/apps/faceup.svg
put faceup-symbolic.svg /usr/share/icons/hicolor/symbolic/apps/faceup-symbolic.svg
put faceup.service "$unitdir/faceup.service"
