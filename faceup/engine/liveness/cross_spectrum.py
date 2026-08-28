"""§7.1 Cross-spectrum consistency: the face must appear in BOTH the IR and the RGB frame,
at the same place (after the enrollment calibration). Screens emit ~no IR; IR-tuned prints
look wrong to the colour camera.
"""
from __future__ import annotations

from ..detect import Box
from . import ABSTAIN, FAIL, PASS, Vote

NAME = "cross"
MIN_IOU = 0.5


def apply_calib(box: Box, calib: dict, width: int, height: int) -> Box:
    """Map an RGB-frame box into IR-frame normalised space using (dx, dy, s)."""
    s = float(calib.get("s", 1.0))
    dx = float(calib.get("dx", 0.0)) * width
    dy = float(calib.get("dy", 0.0)) * height
    return Box(int(box.x * s + dx), int(box.y * s + dy), int(box.w * s), int(box.h * s))


def vote(ir_box: Box | None, rgb_box: Box | None, ir_dark: bool, rgb_dark: bool,
         calib: dict, ir_size: tuple[int, int], rgb_size: tuple[int, int]) -> Vote:
    if ir_dark and rgb_dark:
        return ABSTAIN(NAME, "both frames dark")
    if rgb_dark:
        return ABSTAIN(NAME, "rgb dark (night)")
    if ir_dark:
        return ABSTAIN(NAME, "ir dark (emitter?)")
    if ir_box is None and rgb_box is None:
        return ABSTAIN(NAME, "no face in either frame")
    if ir_box is not None and rgb_box is None:
        return FAIL(NAME, "face in IR only — IR-tuned print?")
    if rgb_box is not None and ir_box is None:
        return FAIL(NAME, "face in RGB only — screen replay?")
    # both present: compare positions in the IR frame's coordinate system
    iw, ih = ir_size
    rw, rh = rgb_size
    # normalise the RGB box to IR pixel space first (cameras may differ in resolution)
    nb = Box(int(rgb_box.x * iw / rw), int(rgb_box.y * ih / rh), int(rgb_box.w * iw / rw), int(rgb_box.h * ih / rh))
    mapped = apply_calib(nb, calib, iw, ih)
    iou = ir_box.iou(mapped)
    if iou >= MIN_IOU:
        return PASS(NAME, f"iou {iou:.2f}", iou)
    return FAIL(NAME, f"boxes disagree (iou {iou:.2f})", iou)
