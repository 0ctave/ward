#!/usr/bin/env bash
# Shared install layout for install.sh (from source) and the distro packages.
#   packaging/stage.sh DESTDIR PREFIX UNITDIR
#   e.g.  stage.sh /            /usr/local /etc/systemd/system     (install.sh)
#         stage.sh "$pkgdir"    /usr       /usr/lib/systemd/system  (deb / rpm / Arch)
# Copies the package, writes the two root launchers, builds pam_ward.so (when gcc exists) and
# installs the desktop entry, polkit action, icons and systemd unit with PREFIX substituted.
# Touches nothing outside DESTDIR; no PAM, no systemctl, no /etc/ward — callers do that.
set -euo pipefail
umask 022
dest=${1:?DESTDIR}; prefix=${2:?PREFIX}; unitdir=${3:?UNITDIR}
dest=${dest%/}
src=$(cd "$(dirname "$0")/.." && pwd)
lib=$prefix/lib/ward
rm -rf "$dest$lib"; mkdir -p "$dest$lib" "$dest$prefix/bin"
cp -r "$src/ward" "$dest$lib/"; install -m644 "$src/README.md" "$dest$lib/README.md"
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
launcher ward ward > "$dest$prefix/bin/ward"
launcher ward.client ward-auth > "$dest$prefix/bin/ward-auth"
chmod 755 "$dest$prefix/bin/ward" "$dest$prefix/bin/ward-auth"
# native PAM module (keyring token + conversation messages); pam_exec stays the fallback
secdir=$dest$prefix/lib/security
if command -v "${CC:-gcc}" >/dev/null 2>&1; then
  mkdir -p "$secdir"
  log=$(mktemp)
  # shellcheck disable=SC2086  # LDFLAGS is a flag list
  if "${CC:-gcc}" -Os -s -fPIC -shared -ffunction-sections -fdata-sections -Wl,--gc-sections -Wl,--as-needed \
       ${LDFLAGS:-} -o "$secdir/pam_ward.so.tmp" "$src/data/pam_ward.c" 2>"$log"; then
    chmod 644 "$secdir/pam_ward.so.tmp"; mv "$secdir/pam_ward.so.tmp" "$secdir/pam_ward.so"
    echo "built $prefix/lib/security/pam_ward.so (native PAM module)"
  else
    rm -f "$secdir/pam_ward.so.tmp"; cat "$log" >&2
    echo "warning: pam_ward.so did not build; the pam_exec line stays" >&2
    [[ -z "${WARD_REQUIRE_PAM_MODULE:-}" ]] || exit 1
  fi
  rm -f "$log"
else
  echo "note: gcc not found — the native PAM module (keyring unlock) was not built" >&2
  [[ -z "${WARD_REQUIRE_PAM_MODULE:-}" ]] || exit 1
fi
sub() { sed "s#/usr/local/#$prefix/#g" "$1"; }
put() { install -d "$(dirname "$dest$2")"; sub "$src/data/$1" > "$dest$2"; chmod 644 "$dest$2"; }
put ward.desktop /usr/share/applications/ward.desktop
put org.ward.policy /usr/share/polkit-1/actions/org.ward.policy     # launcher → pkexec ward app
put ward.svg /usr/share/icons/hicolor/scalable/apps/ward.svg
put ward-symbolic.svg /usr/share/icons/hicolor/symbolic/apps/ward-symbolic.svg
put ward.service "$unitdir/ward.service"
