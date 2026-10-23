#!/usr/bin/env bash
# Build a .deb from this checkout — offline, no debhelper, just dpkg-deb.
#   packaging/build-deb.sh [OUTDIR]      (default: ./dist)
# Needs gcc (pam_faceup.so is compiled here, so the installed system needs no compiler).
# Installs under /usr (launchers /usr/bin/faceup{,-auth}, package /usr/lib/faceup, module
# /usr/lib/security/pam_faceup.so, unit /usr/lib/systemd/system). Never edits PAM: run
# `sudo faceup setup` or `sudo faceup pam enable` after installing.
set -euo pipefail
umask 022
here=$(cd "$(dirname "$0")/.." && pwd)
out=$(mkdir -p "${1:-$here/dist}" && cd "${1:-$here/dist}" && pwd)
ver=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$here/faceup/__init__.py")
arch=$(dpkg --print-architecture)
root=$(mktemp -d); trap 'rm -rf "$root"' EXIT
pkg=$root/faceup
FACEUP_REQUIRE_PAM_MODULE=1 "$here/packaging/stage.sh" "$pkg" /usr /usr/lib/systemd/system
install -Dm644 "$here/LICENSE" "$pkg/usr/share/doc/faceup/copyright"
install -Dm644 "$here/README.md" "$pkg/usr/share/doc/faceup/README.md"
gzip -9n -c "$here/CHANGELOG.md" > "$pkg/usr/share/doc/faceup/changelog.gz"
chmod -R u+rwX,go+rX,go-w "$pkg"   # a checkout made under umask 002 must not ship group-writable root code
mkdir -p "$pkg/DEBIAN"
size=$(du -sk --exclude=DEBIAN "$pkg" | cut -f1)
cat > "$pkg/DEBIAN/control" <<EOC
Package: faceup
Version: $ver-1
Architecture: $arch
Maintainer: octave <redlab336@gmail.com>
Installed-Size: $size
Section: admin
Priority: optional
Depends: python3 (>= 3.10), python3-opencv, python3-numpy, libc6
Recommends: python3-dlib, dlib-models, python3-gi, gir1.2-gtk-4.0, gir1.2-adw-1, systemd, policykit-1 | polkitd
Suggests: linux-enable-ir-emitter
Homepage: https://github.com/<you>/faceup
Description: reliable face unlock for login, sudo and polkit
 Face authentication through PAM with its own engine (YuNet detector, dlib or
 ArcFace embeddings, IR + RGB cameras, multi-frame fusion, liveness voters).
 Every failure falls through to the password. The package never edits PAM:
 run "sudo faceup setup" (or "sudo faceup pam enable") after installing.
EOC
for s in postinst prerm postrm; do install -m755 "$here/packaging/debian/$s" "$pkg/DEBIAN/$s"; done
(cd "$pkg" && find usr -type f -print0 | sort -z | xargs -0 md5sum) > "$pkg/DEBIAN/md5sums"
# reproducible-ish: fixed mtimes and root ownership without fakeroot
find "$pkg" -exec touch -h -d "@${SOURCE_DATE_EPOCH:-$(git -C "$here" log -1 --format=%ct 2>/dev/null || date +%s)}" {} +
deb=$out/faceup_${ver}-1_${arch}.deb
dpkg-deb --root-owner-group -Zxz --build "$pkg" "$deb" >/dev/null
echo "$deb"
