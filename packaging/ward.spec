# RPM spec (Fedora / openSUSE / RHEL-likes). Offline build from a tarball of this repo:
#   git archive --prefix=ward-0.5.0/ -o ~/rpmbuild/SOURCES/ward-0.5.0.tar.gz HEAD
#   rpmbuild -ba packaging/ward.spec
# Fedora COPR: SCM package, Clone URL https://github.com/0ctave/ward, Spec packaging/ward.spec.
# Same layout as packaging/stage.sh under /usr. Never edits PAM: run `sudo ward setup`.
%global debug_package %{nil}

Name:           ward
Version:        0.5.0
Release:        1%{?dist}
Summary:        Reliable face unlock for login, sudo and polkit
License:        MIT
URL:            https://github.com/0ctave/ward
Source0:        %{name}-%{version}.tar.gz

BuildRequires:  gcc
BuildRequires:  systemd-rpm-macros
BuildRequires:  python3-rpm-macros
BuildRequires:  python3
Requires:       python3 >= 3.10
Requires:       python3-opencv
Requires:       python3-numpy
Recommends:     python3-dlib
Recommends:     python3-gobject
Recommends:     gtk4
Recommends:     libadwaita
Recommends:     polkit
%{?systemd_requires}

%description
Face authentication through PAM with its own engine (YuNet detector, dlib or
ArcFace embeddings, IR + RGB cameras, multi-frame fusion, liveness voters).
Every failure falls through to the password. The package never edits PAM:
run "sudo ward setup" (or "sudo ward pam enable") after installing.

%prep
%autosetup

%build
# pam_ward.so is compiled by stage.sh in %%install (one code path with install.sh)

%install
WARD_REQUIRE_PAM_MODULE=1 LDFLAGS="%{?build_ldflags}" packaging/stage.sh %{buildroot} %{_prefix} %{_unitdir}
%py_byte_compile %{python3} %{buildroot}%{_prefix}/lib/ward
install -dm700 %{buildroot}%{_sysconfdir}/ward/models

%post
%systemd_post ward.service
if [ -d /run/systemd/system ]; then
  # like install.sh: on by default on first install; `ward daemon disable` survives upgrades
  if [ $1 -eq 1 ]; then systemctl enable --now ward.service || :; fi
fi
if [ ! -e %{_localstatedir}/log/ward.log ]; then
  touch %{_localstatedir}/log/ward.log; chmod 640 %{_localstatedir}/log/ward.log; chgrp wheel %{_localstatedir}/log/ward.log 2>/dev/null || :
fi
command -v restorecon >/dev/null && restorecon -RF %{_sysconfdir}/ward %{_localstatedir}/log/ward.log 2>/dev/null || :

%preun
if [ $1 -eq 0 ]; then
  # never leave a dangling ward line in the auth stack; this package does not edit PAM itself
  refs=$(grep -lE 'ward-auth|pam_ward\.so' /etc/pam.d/* 2>/dev/null | grep -v '\.ward-bak$' || :)
  if [ -n "$refs" ]; then
    echo "ward is still referenced in: $refs — run 'sudo ward pam disable' first" >&2
    exit 1
  fi
fi
%systemd_preun ward.service

%postun
%systemd_postun_with_restart ward.service

%files
%license LICENSE
%doc README.md CHANGELOG.md
%{_bindir}/ward
%{_bindir}/ward-auth
%{_prefix}/lib/ward/
%{_prefix}/lib/security/pam_ward.so
%{_unitdir}/ward.service
%{_datadir}/applications/ward.desktop
%{_datadir}/polkit-1/actions/org.ward.policy
%{_datadir}/icons/hicolor/scalable/apps/ward.svg
%{_datadir}/icons/hicolor/symbolic/apps/ward-symbolic.svg
%dir %attr(0755,root,root) %{_sysconfdir}/ward
%dir %attr(0700,root,root) %{_sysconfdir}/ward/models

%changelog
* Wed Oct 28 2026 octave <0ctave@users.noreply.github.com> - 0.5.0-1
- Release 0.5.0: packaging for AUR, COPR and a Debian/Ubuntu source package (see CHANGELOG.md)

* Sun Oct 25 2026 octave <0ctave@users.noreply.github.com> - 0.4.0-1
- Rename to Ward; resident engine, anti-spoof CNN, manager app + setup wizard, benchmarks, CI

* Fri Oct 23 2026 octave <0ctave@users.noreply.github.com> - 0.3.0-1
- First packaged release (see CHANGELOG.md)
