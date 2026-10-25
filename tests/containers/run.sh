#!/usr/bin/env bash
# Portability matrix in rootless podman (NOT part of `unittest discover`: no test_*.py here).
#   tests/containers/run.sh [fedora arch tumbleweed ubuntu2204 debian12]
# Per distro: install python3/opencv/numpy (+dlib where packaged)/GTK+libadwaita GI/gcc with the
# names from ward/distro.py, copy the repo in, run the unit suite, `ward doctor`, `ward --help`,
# build data/pam_ward.c with stage.sh's gcc flags; on Fedora also rpmbuild packaging/ward.spec.
# Logs: ${OUT:-/tmp/ward-containers}/<distro>.log. Exit status = number of failing distros.
set -uo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
out=${OUT:-/tmp/ward-containers}; mkdir -p "$out"
pkgs() { (cd "$repo" && python3 -c "from ward.distro import PACKAGES as P; t=P['$1']; print(' '.join(t[k] for k in ('opencv','numpy','dlib','dlib-models','gtk','gcc') if t[k]))"); }
declare -A IMG=([fedora]=registry.fedoraproject.org/fedora:latest [arch]=docker.io/library/archlinux:latest
  [tumbleweed]=registry.opensuse.org/opensuse/tumbleweed:latest [ubuntu2204]=docker.io/library/ubuntu:22.04
  [debian12]=docker.io/library/debian:12)
declare -A FAM=([fedora]=fedora [arch]=arch [tumbleweed]=suse [ubuntu2204]=debian [debian12]=debian)
declare -A INST=(
  [fedora]="dnf -y -q install python3 rpm-build"
  [arch]="pacman -Syu --noconfirm --needed python python-numpy"
  [tumbleweed]="zypper -n -q install python3 python3-numpy"
  [ubuntu2204]="apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends python3 python3-numpy"
  [debian12]="apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends python3 python3-numpy")
GCC='gcc -Os -s -fPIC -shared -ffunction-sections -fdata-sections -Wl,--gc-sections -Wl,--as-needed -o /tmp/pam_ward.so data/pam_ward.c'
# rootless podman needs slirp4netns or pasta for its own network namespace; without either, share
# the host's (NET=host). Override with NET=slirp4netns|pasta.
if [[ -z "${NET:-}" ]]; then command -v pasta >/dev/null || command -v slirp4netns >/dev/null && NET=private || NET=host; fi
fails=0
for d in "${@:-fedora arch tumbleweed ubuntu2204 debian12}"; do
 for d in $d; do
  pm=${INST[$d]}
  extra=$(pkgs "${FAM[$d]}")
  script="set -x; ${pm} ${extra} || exit 10
    python3 -c 'import sys,cv2,numpy; print(sys.version, cv2.__version__, numpy.__version__, hasattr(cv2,\"FaceDetectorYN\"))'
    python3 -c 'import dlib; print(\"dlib\", dlib.__version__)' || echo 'no dlib'
    python3 -c 'import gi; gi.require_version(\"Adw\",\"1\"); from gi.repository import Adw; print(\"adw ok\")' || echo 'no adw'
    systemd-creds --version 2>/dev/null | head -1 || echo 'no systemd-creds'
    cp -a /src /w && cd /w && rm -rf .git tests/__pycache__ || exit 11
    python3 -m unittest discover -s tests 2>&1 | tail -25; r1=\${PIPESTATUS[0]}
    WARD_ETC=/tmp/etc WARD_LOG=/tmp/log python3 -m ward doctor; r2=\$?
    python3 -m ward --help >/dev/null; r3=\$?
    $GCC && test -s /tmp/pam_ward.so; r4=\$?
    r5=0
    if command -v rpmbuild >/dev/null; then
      dnf -y -q builddep packaging/ward.spec || r5=12
      mkdir -p ~/rpmbuild/SOURCES && tar --transform 's,^\.,ward-0.3.0,' -czf ~/rpmbuild/SOURCES/ward-0.3.0.tar.gz . &&
      rpmbuild -bb packaging/ward.spec 2>&1 | tail -15; r5=\$((r5 + \${PIPESTATUS[0]}))
      rpm -qpl ~/rpmbuild/RPMS/*/ward-*.rpm | grep -E 'pam_ward.so|yunet' || r5=13
    fi
    echo \"RESULT unittest=\$r1 doctor=\$r2 help=\$r3 gcc=\$r4 rpm=\$r5\"
    [ \$r1 = 0 ] && [ \$r3 = 0 ] && [ \$r4 = 0 ] && [ \$r5 = 0 ]"
  echo "== $d (${IMG[$d]}): $extra"
  if podman run --rm --network="$NET" -v "$repo:/src:ro,Z" "${IMG[$d]}" bash -c "$script" >"$out/$d.log" 2>&1; then
    echo "   PASS  $(grep RESULT "$out/$d.log")"
  else
    echo "   FAIL  $(grep RESULT "$out/$d.log" || tail -3 "$out/$d.log")"; fails=$((fails+1))
  fi
 done
done
exit $fails
