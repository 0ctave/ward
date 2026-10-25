"""Per-distro package names for install hints (install.sh, doctor): one table, so no command
tells a Fedora or Arch user to run apt. Logical names: opencv, dlib, dlib-models, gtk (GTK 4 +
libadwaita introspection), gcc, numpy. Verified 2026-10-23 by tests/containers/run.sh (podman) on Fedora 44,
Arch, openSUSE Tumbleweed, Ubuntu 22.04 and Debian 12: python3-dlib is NOT in the Debian/Ubuntu
archives (Ubuntu users get it from a PPA such as Howdy's) nor in Fedora, and Arch's `dlib` has no
Python bindings; Debian's libdlib-data carries only the 68-point model, so dlib-models stays a note.
Tumbleweed's python3-dlib resolves to python3XX-dlib through its provides."""
from __future__ import annotations


PACKAGES = {
    "debian": {"cmd": "apt install", "opencv": "python3-opencv", "numpy": "python3-numpy", "dlib": "", "dlib-models": "",
               "gtk": "python3-gi python3-gi-cairo gir1.2-adw-1", "gcc": "gcc libc6-dev"},
    "fedora": {"cmd": "dnf install", "opencv": "python3-opencv", "numpy": "python3-numpy", "dlib": "", "dlib-models": "",
               "gtk": "python3-gobject libadwaita", "gcc": "gcc"},
    "arch": {"cmd": "pacman -S", "opencv": "python-opencv", "numpy": "python-numpy", "dlib": "", "dlib-models": "",
             "gtk": "python-gobject python-cairo libadwaita", "gcc": "gcc"},
    "suse": {"cmd": "zypper install", "opencv": "python3-opencv", "numpy": "python3-numpy", "dlib": "python3-dlib", "dlib-models": "",
             "gtk": "python3-gobject python3-gobject-cairo typelib-1_0-Adw-1", "gcc": "gcc"},
}

NOT_PACKAGED = {
    "dlib": "python3-dlib is not packaged here: use the ArcFace recogniser (`ward recogniser install`), "
            "or get dlib from a PPA (Ubuntu), the AUR (python-dlib) or source",
    "dlib-models": "download shape_predictor_5_face_landmarks.dat, shape_predictor_68_face_landmarks.dat and "
                   "dlib_face_recognition_resnet_model_v1.dat (dlib.net/files) into /usr/share/dlib",
}


def family(os_release: str | None = None) -> str:
    """'debian' | 'fedora' | 'arch' | 'suse' | '' from /etc/os-release ID and ID_LIKE."""
    if os_release is None:
        try:
            os_release = open("/etc/os-release").read()
        except OSError:
            return ""
    ids = []
    for ln in os_release.splitlines():
        k, _, v = ln.partition("=")
        if k in ("ID", "ID_LIKE"):
            ids += v.strip().strip('"').lower().split()
    for i in ids:
        if i in ("debian", "ubuntu", "linuxmint", "pop", "elementary"):
            return "debian"
        if i in ("fedora", "rhel", "centos", "rocky", "almalinux"):
            return "fedora"
        if i in ("arch", "manjaro", "endeavouros", "cachyos"):
            return "arch"
        if i.startswith("opensuse") or i in ("suse", "sles"):
            return "suse"
    return ""


def hint(*what: str, os_release: str | None = None) -> str:
    """One install command for the logical packages `what`, plus notes for what is not packaged."""
    fam = family(os_release)
    table = PACKAGES.get(fam)
    if table is None:
        return "install with your package manager: " + ", ".join(what) + " (Python bindings for the distro python3)"
    names = [table[w] for w in what if table.get(w)]
    notes = [NOT_PACKAGED[w] for w in what if not table.get(w) and w in NOT_PACKAGED]
    out = f"sudo {table['cmd']} {' '.join(names)}" if names else ""
    return "; ".join(x for x in [out] + notes if x)


if __name__ == "__main__":                     # install.sh: python3 -m ward.distro opencv dlib …
    import sys
    print(hint(*sys.argv[1:]))
