"""Where the installed launchers, PAM module and unit live.

install.sh installs under /usr/local (from source); the distro packages (packaging/) under /usr.
The prefix follows the package's own location, so a packaged faceup writes PAM lines pointing at
/usr/bin/faceup-auth and a from-source one at /usr/local/bin/faceup-auth. A checkout run directly
(tests, development) uses whichever install exists, /usr/local first."""
import os

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def prefix(pkg_parent: str = _PKG_PARENT, exists=os.path.exists) -> str:
    for p in ("/usr/local", "/usr"):
        if pkg_parent == f"{p}/lib/faceup":
            return p
    for p in ("/usr/local", "/usr"):
        if exists(f"{p}/bin/faceup-auth"):
            return p
    return "/usr/local"


PREFIX = prefix()
LIB = f"{PREFIX}/lib/faceup"                         # the package directory (holds faceup/)
BIN = f"{PREFIX}/bin/faceup"
AUTH = f"{PREFIX}/bin/faceup-auth"
PAM_MODULE = f"{PREFIX}/lib/security/pam_faceup.so"  # absolute path in the PAM line: any libdir layout works
# install.sh writes the unit to /etc/systemd/system, the packages to the vendor unit directory
UNIT_PATHS = ("/etc/systemd/system/faceup.service", "/usr/lib/systemd/system/faceup.service", "/lib/systemd/system/faceup.service")


def unit_installed(exists=os.path.exists) -> bool:
    return any(exists(p) for p in UNIT_PATHS)
