"""Update check: version comparison, best-effort fetch, caching, and the doctor cache-only path.
No network (urllib is mocked), no camera."""
import json
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("FACEUP_ETC", tempfile.mkdtemp(prefix="ward-test-"))
os.environ["WARD_CACHE"] = tempfile.mkdtemp(prefix="ward-cache-")

from ward import update


class Versions(unittest.TestCase):
    def test_parse_and_compare(self):
        self.assertEqual(update._parse("v0.3.1"), (0, 3, 1))
        self.assertEqual(update._parse("0.4.0-rc1"), (0, 4, 0))
        self.assertTrue(update.is_newer("v0.4.0", "0.3.0"))
        self.assertTrue(update.is_newer("0.3.10", "0.3.2"))      # numeric, not lexical
        self.assertFalse(update.is_newer("0.3.0", "0.3.0"))
        self.assertFalse(update.is_newer("v0.2.9", "0.3.0"))


class Fetch(unittest.TestCase):
    def test_fetch_is_best_effort(self):
        with mock.patch("urllib.request.urlopen", side_effect=OSError("offline")):
            self.assertIsNone(update.fetch_latest(timeout=0.1))

    def test_check_caches_a_tag(self):
        import io

        class Resp(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False
        body = io.BytesIO(json.dumps({"tag_name": "v9.9.9"}).encode())
        body.__enter__ = lambda: body; body.__exit__ = lambda *a: False
        with mock.patch("urllib.request.urlopen", return_value=body):
            rec = update.check()
        self.assertEqual(rec["latest"], "v9.9.9")
        self.assertTrue(update.CACHE.exists())
        self.assertTrue(update.is_newer(rec["latest"]))

    def test_cached_freshness(self):
        update.CACHE.parent.mkdir(parents=True, exist_ok=True)
        update.CACHE.write_text(json.dumps({"current": "0.3.0", "latest": "v1.0.0", "checked": int(time.time())}))
        self.assertEqual(update.cached()["latest"], "v1.0.0")
        update.CACHE.write_text(json.dumps({"latest": "v1.0.0", "checked": int(time.time()) - 999999}))
        self.assertIsNone(update.cached())                       # stale → ignored
        update.CACHE.write_text("not json")
        self.assertIsNone(update.cached())                       # corrupt → ignored, no raise


class Doctor(unittest.TestCase):
    def test_health_updates_section_is_offline(self):
        from ward import health
        with mock.patch.object(update, "cached", return_value={"latest": "v9.9.9", "checked": 0}):
            rep = health.collect(user="nobody")
        upd = next(s for s in rep["sections"] if s.key == "updates")
        self.assertTrue(any("newer Ward" in i.text for i in upd.items))
        self.assertTrue(any(i.fix == ["update"] for i in upd.items))

    def test_cmd_update_offline_warns(self):
        from ward.cli import cmd_update
        with mock.patch.object(update, "fetch_latest", return_value=None):
            self.assertEqual(cmd_update(None), 1)


if __name__ == "__main__":
    unittest.main()
