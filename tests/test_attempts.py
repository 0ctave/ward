"""Attempt log parsing and summary (roadmap U5)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from faceup import attempts

LINES = """2026-09-03T12:48:00 user=octave result=accept ir=0.14 rgb=0.36 irw=11.14/5.66 votes=cross:PASS,motion:ABSTAIN learned=ir:C/up frames=1 t=1616ms
2026-09-03T14:46:37 user=octave result=camera frames=0 t=520ms detail='cannot open ir camera /dev/video2 (busy?)'
2026-09-03T16:08:44 user=octave result=accept ir=0.42 irw=34.20/5.68 votes=cross:ABSTAIN,motion:PASS frames=3 t=1600ms detail='via sudo'
2026-09-04T00:41:27 user=octave result=no-match ir=0.61 frames=9 t=2500ms detail='via gdm-password'
2026-09-04T00:42:27 user=octave result=spoof:cross frames=4 t=900ms detail='via sudo'
garbage line
"""


class Parse(unittest.TestCase):
    def test_fields(self):
        rows = [r for r in (attempts.parse(l) for l in LINES.splitlines()) if r]
        self.assertEqual(len(rows), 5)
        r = rows[0]
        self.assertTrue(r["accepted"]); self.assertEqual(r["ir"], 0.14); self.assertEqual(r["ms"], 1616)
        self.assertEqual(r["votes"], ["cross:PASS", "motion:ABSTAIN"]); self.assertEqual(r["learned"], "ir:C/up")
        self.assertEqual(rows[1]["reason"], "camera"); self.assertIn("busy", rows[1]["detail"])
        self.assertEqual(rows[2]["service"], "sudo"); self.assertEqual(rows[3]["service"], "gdm-password")
        self.assertEqual(rows[4]["reason"], "spoof")

    def test_summary(self):
        rows = [r for r in (attempts.parse(l) for l in LINES.splitlines()) if r]
        s = attempts.summary(rows)
        self.assertEqual((s["attempts"], s["accepted"]), (5, 2))
        self.assertEqual(s["reasons"], {"camera": 1, "no-match": 1, "spoof": 1})
        self.assertEqual(s["median_accept_ms"], 1608)
        self.assertEqual(s["services"]["sudo"], {"attempts": 2, "accepted": 1})
        self.assertEqual(s["learned"], 1); self.assertEqual(s["spoof"], 1)
        self.assertEqual(attempts.summary([])["attempts"], 0)



class DarkFlag(unittest.TestCase):
    def test_service_and_dark_parse_together(self):
        from faceup import attempts
        r = attempts.parse("2026-10-22T23:01:29 user=o result=accept ir=0.31 frames=3 t=1500ms detail='via sudo; ir-only (dark)'")
        self.assertEqual(r["service"], "sudo"); self.assertTrue(r["dark"])
        r = attempts.parse("2026-10-22T13:01:29 user=o result=accept ir=0.31 frames=3 t=1500ms detail='via gdm-password'")
        self.assertEqual(r["service"], "gdm-password"); self.assertFalse(r["dark"])

if __name__ == "__main__":
    unittest.main()
