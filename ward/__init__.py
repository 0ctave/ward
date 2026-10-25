"""ward — make face unlock on Linux actually work."""
__version__ = "0.4.0"


def main(argv=None):
    """The CLI (kept lazy: ward-auth's fast path imports this package without numpy/cv2)."""
    from .cli import main as _main
    return _main(argv)
