"""faceup — make face unlock on Linux actually work."""
__version__ = "0.3.0"


def main(argv=None):
    """The CLI (kept lazy: faceup-auth's fast path imports this package without numpy/cv2)."""
    from .cli import main as _main
    return _main(argv)
