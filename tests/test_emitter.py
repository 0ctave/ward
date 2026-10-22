"""Emitter control (roadmap L1): tool-file parsing, OFF-state discovery, toggle bench, shadow pair."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from faceup.engine import emitter

TOOL_FILE = """- status: start
  unit: 4
  selector: 2
  current:
    - 0
    - 32
  maximum:
    - 255
    - 255
  minimum:
    - 0
    - 0
- status: idle
  unit: 4
  selector: 3
  current:
    - 0
  maximum:
    - 255
  minimum:
    - 0
"""


class FakeXU:
    def __init__(self, cur: dict, default: dict | None = None):
        self.cur = dict(cur); self.default = default or {}; self.log = []

    def get(self, unit, selector, size, code=emitter.UVC_GET_CUR):
        if code == emitter.UVC_GET_CUR:
            return self.cur[(unit, selector)]
        if code == emitter.UVC_GET_DEF and (unit, selector) in self.default:
            return self.default[(unit, selector)]
        raise OSError("unsupported")

    def set(self, unit, selector, value):
        self.cur[(unit, selector)] = bytes(value); self.log.append((unit, selector, bytes(value)))

    def close(self):
        pass


class Parsing(unittest.TestCase):
    def test_parse_and_state(self):
        ins = emitter.parse_tool_file(TOOL_FILE)
        self.assertEqual(len(ins), 2)
        self.assertEqual((ins[0].unit, ins[0].selector, ins[0].on, ins[0].status), (4, 2, b"\x00\x20", "start"))
        self.assertEqual(ins[1].status, "idle")
        d = Path(tempfile.mkdtemp())
        self.assertEqual(emitter.tool_state(d), "absent")
        (d / "dev").write_text(TOOL_FILE.replace("status: start", "status: idle", 1))
        self.assertEqual(emitter.tool_state(d), "none")               # this machine's situation
        self.assertEqual(emitter.load_instructions(d), [])
        (d / "dev").write_text(TOOL_FILE)
        self.assertEqual(emitter.tool_state(d), "working")
        self.assertEqual(len(emitter.load_instructions(d)), 1)


class Switching(unittest.TestCase):
    def test_off_state_is_what_was_there_before(self):
        ins = [emitter.Instruction(4, 2, b"\x00\x20")]
        xu = FakeXU({(4, 2): b"\x00\x00"})
        em = emitter.Emitter("/dev/videoX", ins, xu=xu)
        self.assertTrue(em.available)
        self.assertTrue(em.on()); self.assertEqual(xu.cur[(4, 2)], b"\x00\x20")
        self.assertTrue(em.off()); self.assertEqual(xu.cur[(4, 2)], b"\x00\x00")

    def test_off_state_falls_back_to_default_then_minimum(self):
        ins = [emitter.Instruction(4, 2, b"\x00\x20", minimum=b"\x00\x00")]
        xu = FakeXU({(4, 2): b"\x00\x20"}, default={(4, 2): b"\x00\x05"})   # already on at boot
        em = emitter.Emitter("/dev/videoX", ins, xu=xu)
        self.assertTrue(em.available); em.off(); self.assertEqual(xu.cur[(4, 2)], b"\x00\x05")
        xu = FakeXU({(4, 2): b"\x00\x20"})
        em = emitter.Emitter("/dev/videoX", ins, xu=xu)
        self.assertTrue(em.available); em.off(); self.assertEqual(xu.cur[(4, 2)], b"\x00\x00")
        # no way to tell OFF: unavailable, never raises
        em = emitter.Emitter("/dev/videoX", [emitter.Instruction(4, 2, b"\x00\x20")], xu=FakeXU({(4, 2): b"\x00\x20"}))
        self.assertFalse(em.available); self.assertIn("OFF", em.error)

    def test_no_instruction_means_unavailable(self):
        em = emitter.Emitter("/dev/videoX", [], xu=FakeXU({}))
        self.assertFalse(em.available); self.assertIn("no working", em.error)


class Bench(unittest.TestCase):
    def _sim(self, delay_frames: int, on_level: float = 120.0, off_level: float = 30.0):
        """A camera whose brightness follows the LED state after `delay_frames` frames."""
        ins = [emitter.Instruction(4, 2, b"\x01")]
        xu = FakeXU({(4, 2): b"\x00"})
        em = emitter.Emitter("/dev/videoX", ins, xu=xu)
        history = []
        def read():
            history.append(xu.cur[(4, 2)] == b"\x01")
            state = history[-1 - delay_frames] if len(history) > delay_frames else history[0]
            return np.full((4, 4), on_level if state else off_level, np.uint8)
        return em, read, xu

    def test_fast_toggle_is_usable(self):
        em, read, xu = self._sim(delay_frames=1)
        r = emitter.toggle_bench(em, read, cycles=2, frames_per_state=6, fps=30)
        self.assertTrue(r.usable, r.reason); self.assertEqual(r.latency_ms, 33)
        self.assertAlmostEqual(r.delta, 90, delta=1)
        self.assertEqual(xu.cur[(4, 2)], b"\x01")                   # left ON

    def test_slow_toggle_is_rejected(self):
        em, read, _ = self._sim(delay_frames=5)
        r = emitter.toggle_bench(em, read, cycles=2, frames_per_state=8, fps=30)
        self.assertFalse(r.usable); self.assertGreater(r.latency_ms, emitter.SHADOW_MAX_LATENCY_MS)

    def test_no_brightness_change_is_reported(self):
        em, read, _ = self._sim(delay_frames=1, on_level=40, off_level=35)
        r = emitter.toggle_bench(em, read, cycles=2, frames_per_state=6)
        self.assertFalse(r.usable); self.assertIn("only", r.reason)

    def test_shadow_pair_leaves_emitter_on(self):
        em, read, xu = self._sim(delay_frames=0)
        off, on = emitter.shadow_pair(em, read, settle_frames=1)
        self.assertEqual((int(off.mean()), int(on.mean())), (30, 120)); self.assertEqual(xu.cur[(4, 2)], b"\x01")
        em2 = emitter.Emitter("/dev/videoX", [], xu=FakeXU({}))
        self.assertEqual(emitter.shadow_pair(em2, read), (None, None))


if __name__ == "__main__":
    unittest.main()
