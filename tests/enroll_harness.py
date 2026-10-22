"""Headless driver for gui.guided_setup — the enrolment window as a unit test.

Everything around the step function is faked deterministically: a virtual clock, a screen that
calls step() at 30 fps and replays scripted keys, a camera that returns a synthetic frame, models
whose detector and landmarks are stubs, a pose estimator driven by a scripted "user", and an
in-process sampler that returns embeddings near the user's own. No GTK, no camera, no dlib.

    world = World()                       # a cooperative user: reaches each target after 0.6 s
    res, snaps = run_setup(world)         # guided_setup's return value + every snapshot it emitted
"""
from __future__ import annotations

import os
import sys
import tempfile
from contextlib import ExitStack
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("FACEUP_ETC", tempfile.mkdtemp(prefix="faceup-enroll-"))

from faceup import display as disp, gui  # noqa: E402
from faceup.engine import pose as posemod, store  # noqa: E402
from faceup.engine.capture import Frame  # noqa: E402
from faceup.engine.detect import Box  # noqa: E402

FPS = 15.0                 # virtual camera rate (real ~30; halves the test time, same logic)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class World:
    """The scripted user and room. Override the hooks to script a scenario.
    face(t) → whether a face is in view; target_reached_after → seconds a user needs to reach a
    new target; pose_for(target_disp, since) → the DISPLAY-space (yaw, pitch) the user holds."""

    def __init__(self, reach_s: float = 0.6, face_w: int = 120, light: str = "ok", base_seed: int = 1):
        self.reach_s, self.face_w, self.light = reach_s, face_w, light
        rng = np.random.default_rng(base_seed)
        self.base = rng.normal(size=128); self.base /= np.linalg.norm(self.base)
        self.rng = np.random.default_rng(base_seed + 1)
        self.target = (0.0, 0.0); self.target_since = 0.0
        self.keys_at: dict[float, list[str]] = {}       # virtual seconds from start → keys
        self.keys_after_done: dict[float, list[str]] = {}   # seconds after the FIRST done card → keys
        self.t_done = None
        self.no_frame: tuple[float, float] | None = None
        self.t0 = None

    def face(self, t: float) -> bool:
        return True

    def light_at(self, t: float) -> str:
        return self.light

    def frame_ok(self, t: float) -> bool:
        return not (self.no_frame and self.no_frame[0] <= t < self.no_frame[1])

    def pose(self, t: float):
        """Display pose: moves linearly from centre to the current target over reach_s."""
        k = min(1.0, (t - self.target_since) / max(self.reach_s, 1e-6))
        return (self.target[0] * k, self.target[1] * k)

    def embedding(self) -> np.ndarray:
        e = self.base + self.rng.normal(scale=0.012, size=128)
        return e / np.linalg.norm(e)


class FakeScreen:
    snapshot = True

    def __init__(self, world: World, clock: Clock, max_s: float):
        self.world, self.clock, self.max_s = world, clock, max_s
        self.snaps: list[dict] = []
        self._pending: list[str] = []

    def keys(self) -> list[str]:
        t = self.clock() - self.world.t0
        due = [k for k in list(self.world.keys_at) if k <= t]
        out = []
        for k in sorted(due):
            out += self.world.keys_at.pop(k)
        if self.world.t_done is not None:
            for k in sorted(d for d in list(self.world.keys_after_done) if t - self.world.t_done >= d):
                out += self.world.keys_after_done.pop(k)
        return out

    def run(self, step) -> None:
        end = self.clock() + self.max_s
        while self.clock() < end:
            self.clock.advance(1.0 / FPS)
            snap = step()
            if snap is None:
                return
            self.snaps.append({k: v for k, v in snap.items() if k != "frame"})
        self.timed_out = True


class FakeCap:
    def __init__(self, world: World, clock: Clock, spectra=("ir",)):
        self.world, self.clock = world, clock
        self.spectra = list(spectra)
        self.notes: list[str] = []
        self.lost = ""

    def read_one(self, spectrum, flush=0):
        t = self.clock() - self.world.t0
        if spectrum not in self.spectra or not self.world.frame_ok(t):
            return None
        g = np.full((180, 320), 90, np.uint8)
        g[50:150, 110:210] = 140                       # something with texture for measure()
        g[::7, :] = 60
        return Frame(spectrum, self.clock(), g, np.dstack([g, g, g]))

    def meter(self, spectrum, gray, box):
        world, t = self.world, self.clock() - self.world.t0
        return mock.Mock(state=lambda sp: world.light_at(t))


class FakeModels:
    max_height = 320
    sp68 = None
    detector_kind = "yunet"
    recogniser_kind = "dlib"

    def __init__(self, world: World, clock: Clock):
        self.world, self.clock = world, clock

    def detect(self, gray, near=None):
        t = self.clock() - self.world.t0
        if not self.world.face(t):
            return None
        w = self.world.face_w
        return Box(160 - w // 2, 90 - w // 2, w, w)

    def landmarks68(self, gray, box):
        return np.zeros((68, 2), np.float32)


class FakeSampler:
    """In-process stand-in for the spawn-process sampler: one job in flight, results next drain."""

    def __init__(self, world: World):
        self.world = world
        self._out: list = []
        self.alive = True

    @property
    def busy(self) -> bool:
        return bool(self._out) or not self.alive

    def submit(self, cell, tag, items) -> bool:
        if not self.alive:
            return False
        results = [(sp, q, self.world.embedding(), []) for sp, _rgb, _box, q, *_ in items]
        self._out.append((cell, tag, results))
        return True

    def drain(self) -> list:
        out, self._out = self._out, []
        return out

    def close(self, timeout: float = 3.0) -> list:
        return self.drain()


def run_setup(world: World, max_s: float = 240.0, condition: str = "", poses=None, st=None,
              sampler_cls=None, cap_spectra=("ir",), need=None, timeout=None):
    """Run guided_setup against `world`. Returns (result, snapshots, store)."""
    clock = Clock()
    world.t0 = clock()
    st = st or store.Store(user="harness", path=store.path_for("harness"))
    st.save = lambda: None
    screen = FakeScreen(world, clock, max_s)
    cap = FakeCap(world, clock, cap_spectra)
    models = FakeModels(world, clock)
    sgn = -1.0                                            # mirrored preview: display yaw = -camera yaw

    # the scripted user follows the window: when it shows a new target, the user starts moving
    real_step_snaps = screen.snaps

    def estimate(lm, w, h, shape3d=None, neutral=None):
        t = clock() - world.t0
        y, p = world.pose(t)
        return posemod.Pose(sgn * y, p, 0.0)              # camera convention in, display out of gui

    class Track(list):
        def append(self, s):
            super().append(s)
            if s.get("phase") == "run" and s.get("target") is not None and tuple(s["target"]) != world.target:
                world.target = tuple(s["target"]); world.target_since = clock() - world.t0
            if s.get("phase") == "intro":
                world.target = (0.0, 0.0)
            if s.get("phase") == "done" and world.t_done is None:
                world.t_done = clock() - world.t0

    screen.snaps = Track()
    cams = {"ir_dark_threshold": 25, "rgb_dark_threshold": 40}
    with ExitStack() as es:
        es.enter_context(mock.patch.object(gui.time, "monotonic", clock))
        es.enter_context(mock.patch.object(disp, "open_display", lambda *a, **k: screen))
        es.enter_context(mock.patch.object(gui, "_Sampler", sampler_cls or (lambda models: FakeSampler(world))))
        es.enter_context(mock.patch.object(posemod, "estimate", estimate))
        es.enter_context(mock.patch("faceup.engine.emitter.instructions_for", lambda *a, **k: []))
        es.enter_context(mock.patch.object(gui, "_ir_device", lambda cap: ""))
        res = gui.guided_setup(st, cap, models, cams, {"fusion_frames": 4}, condition=condition,
                               poses=poses, need=need, timeout=timeout)
    return res, list(screen.snaps), st
