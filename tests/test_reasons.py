import unittest

from ward import feedback, reasons


class Reasons(unittest.TestCase):
    def test_label_and_tip(self):
        self.assertEqual(reasons.label("no-match"), "face not recognised")
        self.assertEqual(reasons.label("totally-unknown"), "totally-unknown")   # fallback = the code
        self.assertTrue(reasons.tip("no-match"))
        self.assertEqual(reasons.tip("policy"), "")                             # nothing actionable
        self.assertEqual(reasons.tip("totally-unknown"), "")

    def test_covers_every_attempt_reason(self):
        """Drift guard: every during-attempt result (minus 'accept') has a Reliability-page label,
        so the app and `ward log --summary` can't disagree about a reason the engine can emit."""
        for r in feedback.RESULT:
            if r == "accept":
                continue
            self.assertIn(r, reasons.LABEL, f"reason {r!r} in feedback.RESULT has no reasons.LABEL")


if __name__ == "__main__":
    unittest.main()
