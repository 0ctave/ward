"""Per-distro package names for install hints (install.sh, doctor): one table, so no command
tells a Fedora or Arch user to run apt. Logical names: opencv, dlib, dlib-models, gtk (GTK 4 +
libadwaita introspection), gcc. Package names are best knowledge, not verified on every release —
the hint says so when a piece is not packaged."""
from __future__ import annotations


PACKAGES = {
    "debian": {"cmd": "apt install", "opencv": "python3-opencv", "dlib": "python3-dlib", "dlib-models": "dlib-models",
               "gtk": "python3-gi gir1.2-adw-1", "gcc": "gcc"},
    "fedora": {"cmd": "dnf install", "opencv": "python3-opencv", "dlib": "python3-dlib", "dlib-models": "",
               "gtk": "python3-gobject libadwaita", "gcc": "gcc"},
    "arch": {"cmd": "pacman -S", "opencv": "python-opencv", "dlib": "", "dlib-models": "",
             "gtk": "python-gobject libadwaita", "gcc": "gcc"},
    "suse": {"cmd": "zypper install", "opencv": "python3-opencv", "dlib": "", "dlib-models": "",
             "gtk": "python3-gobject typelib-1_0-Adw-1", "gcc": "gcc"},
}

NOT_PACKAGED = {
    "dlib": "dlib is not packaged here: use the ArcFace recogniser (`faceup recogniser install`) or build dlib from the AUR/source",
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


if __name__ == "__main__":                     # install.sh: python3 -m faceup.distro opencv dlib …
    import sys
    print(hint(*sys.argv[1:]))
