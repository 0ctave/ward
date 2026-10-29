# COPR "make_srpm" entry point.
#
# Our ward.spec has a plain `Source0: %{name}-%{version}.tar.gz` with no lookaside
# cache, so COPR's default "rpkg" SRPM method can't produce the tarball and the
# build fails with "Bad file: ward-<ver>.tar.gz: No such file". This builds the
# SRPM the same way the release workflow does — a `git archive` tarball fed to
# `rpmbuild -bs` — which is exactly what COPR's "make_srpm" method invokes:
#
#   make -f .copr/Makefile srpm outdir=<dir> spec=packaging/ward.spec
#
# When adding the SCM package in COPR, pick SRPM method "make_srpm" (not rpkg).

VER := $(shell sed -n 's/^Version:[[:space:]]*//p' packaging/ward.spec)

.PHONY: srpm
srpm:
	dnf -y install rpm-build git-core
	mkdir -p "$(outdir)" rpmbuild/SOURCES
	git archive --prefix=ward-$(VER)/ -o rpmbuild/SOURCES/ward-$(VER).tar.gz HEAD
	rpmbuild -bs packaging/ward.spec \
	    --define "_topdir $(CURDIR)/rpmbuild" \
	    --define "_sourcedir $(CURDIR)/rpmbuild/SOURCES" \
	    --define "_srcrpmdir $(outdir)"
