#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_ensemble as runner  # noqa: E402


class RunnerInvariantTests(unittest.TestCase):
    def test_completed_run_skip_requires_exact_identity_and_config(self):
        spec = runner.RunSpec(4101, "feedback", 4101037, "pool4101_feedback", False)
        metadata = {
            "status": "complete",
            "pool_seed": 4101,
            "policy": "feedback",
            "policy_seed": 4101037,
            "configuration_sha256": "locked-hash",
            "requested_cycles": 12,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pool4101_feedback_metadata.json"
            path.write_text(json.dumps(metadata), encoding="utf-8")
            self.assertTrue(
                runner.is_complete(Path(tmp), spec, "locked-hash", 12)
            )
            self.assertFalse(
                runner.is_complete(Path(tmp), spec, "different-hash", 12)
            )
            self.assertFalse(
                runner.is_complete(Path(tmp), spec, "locked-hash", 8)
            )


if __name__ == "__main__":
    unittest.main()
