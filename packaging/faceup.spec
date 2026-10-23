# RPM spec (Fedora / openSUSE / RHEL-likes). Offline build from a tarball of this repo:
#   git archive --prefix=faceup-0.3.0/ -o ~/rpmbuild/SOURCES/faceup-0.3.0.tar.gz HEAD
#   rpmbuild -ba packaging/faceup.spec
# Same layout as packaging/stage.sh under /usr. Never edits PAM: run `sudo faceup setup`.
%global debug_package %{nil}

Name:           faceup
Version:        0.3.0
Release:        1%{?dist}
Summary:        Reliable face unlock for login, sudo and polkit
License:        MIT
URL:            https://github.com/<you>/faceup
Source0:        %{name}-%{version}.tar.gz

BuildRequires:  gcc
BuildRequires:  systemd-rpm-macros
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
run "sudo faceup setup" (or "sudo faceup pam enable") after installing.

%prep
%autosetup

%build
# pam_faceup.so is compiled by stage.sh in %%install (one code path with install.sh)

%install
FACEUP_REQUIRE_PAM_MODULE=1 LDFLAGS="%{?build_ldflags}" packaging/stage.sh %{buildroot} %{_prefix} %{_unitdir}
%py_byte_compile %{python3} %{buildroot}%{_prefix}/lib/faceup
install -dm700 %{buildroot}%{_sysconfdir}/faceup/models

%post
%systemd_post faceup.service
if [ -d /run/systemd/system ]; then
  # like install.sh: on by default on first install; `faceup daemon disable` survives upgrades
  if [ $1 -eq 1 ]; then systemctl enable --now faceup.service || :; fi
fi
if [ ! -e %{_localstatedir}/log/faceup.log ]; then
  touch %{_localstatedir}/log/faceup.log; chmod 640 %{_localstatedir}/log/faceup.log; chgrp wheel %{_localstatedir}/log/faceup.log 2>/dev/null || :
fi
command -v restorecon >/dev/null && restorecon -RF %{_sysconfdir}/faceup %{_localstatedir}/log/faceup.log 2>/dev/null || :

%preun
if [ $1 -eq 0 ]; then
  # never leave a dangling faceup line in the auth stack; this package does not edit PAM itself
  refs=$(grep -lE 'faceup-auth|pam_faceup\.so' /etc/pam.d/* 2>/dev/null | grep -v '\.faceup-bak$' || :)
  if [ -n "$refs" ]; then
    echo "faceup is still referenced in: $refs — run 'sudo faceup pam disable' first" >&2
    exit 1
  fi
fi
%systemd_preun faceup.service

%postun
%systemd_postun_with_restart faceup.service

%files
%license LICENSE
%doc README.md CHANGELOG.md
%{_bindir}/faceup
%{_bindir}/faceup-auth
%{_prefix}/lib/faceup/
%{_prefix}/lib/security/pam_faceup.so
%{_unitdir}/faceup.service
%{_datadir}/applications/faceup.desktop
%{_datadir}/polkit-1/actions/org.faceup.policy
%{_datadir}/icons/hicolor/scalable/apps/faceup.svg
%{_datadir}/icons/hicolor/symbolic/apps/faceup-symbolic.svg
%dir %attr(0755,root,root) %{_sysconfdir}/faceup
%dir %attr(0700,root,root) %{_sysconfdir}/faceup/models

%changelog
* Fri Oct 23 2026 octave <redlab336@gmail.com> - 0.3.0-1
- First packaged release (see CHANGELOG.md)
