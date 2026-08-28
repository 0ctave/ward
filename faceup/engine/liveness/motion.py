"""§7.4 Non-rigid motion: a real head moves like a 3D object, a photo/screen moves like a plane.

Between the first and last good frames of the window we track many points inside the
face (Shi-Tomasi corners + Lucas-Kanade optical flow), fit ONE global homography by
least squares, and look at the reprojection error. A plane is explained almost exactly;
a face is not (nose vs ears parallax, mouth/jaw deformation).
"""
from __future__ import annotations

import cv2
import numpy as np

from ..detect import Box
from . import ABSTAIN, FAIL, PASS, Vote

NAME = "motion"
MIN_DISPLACEMENT_PX = 1.5     # below this the user simply didn't move → abstain
PASS_RMS_PX = 0.8             # residual of a global homography ≥ this → non-rigid → PASS
FAIL_RMS_PX = 0.3             # ≤ this → moved like a plane → FAIL
MIN_POINTS = 25


def _track(g0: np.ndarray, g1: np.ndarray, box: Box):
    mask = np.zeros_like(g0)
    # inner face region: skip the box border where background leaks in
    mx, my = int(box.w * 0.12), int(box.h * 0.12)
    mask[max(0, box.y + my):box.y + box.h - my, max(0, box.x + mx):box.x + box.w - mx] = 255
    p0 = cv2.goodFeaturesToTrack(g0, maxCorners=200, qualityLevel=0.01, minDistance=4, mask=mask)
    if p0 is None or len(p0) < MIN_POINTS:
        return None, None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(15, 15), maxLevel=3)
    ok = st.reshape(-1) == 1
    return p0[ok].reshape(-1, 2), p1[ok].reshape(-1, 2)


def vote(first_gray: np.ndarray, last_gray: np.ndarray, box: Box) -> Vote:
    p0, p1 = _track(first_gray, last_gray, box)
    if p0 is None or len(p0) < MIN_POINTS:
        return ABSTAIN(NAME, "too few trackable points")
    disp = float(np.mean(np.linalg.norm(p1 - p0, axis=1)))
    if disp < MIN_DISPLACEMENT_PX:
        return ABSTAIN(NAME, f"still ({disp:.1f}px)", disp)
    H, _ = cv2.findHomography(p0, p1, method=0)   # plain least squares on purpose (no RANSAC)
    if H is None:
        return ABSTAIN(NAME, "no homography")
    proj = cv2.perspectiveTransform(p0.reshape(-1, 1, 2), H).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((proj - p1) ** 2, axis=1))))
    if rms >= PASS_RMS_PX:
        return PASS(NAME, f"non-rigid (rms {rms:.2f}px, moved {disp:.1f}px)", rms)
    if rms <= FAIL_RMS_PX:
        return FAIL(NAME, f"planar motion (rms {rms:.2f}px, moved {disp:.1f}px)", rms)
    return ABSTAIN(NAME, f"inconclusive (rms {rms:.2f}px)", rms)
