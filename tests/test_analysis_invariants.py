#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import analyze_ensemble as analysis  # noqa: E402


def paired_frames(seeds=(4101, 4102), mismatched_signature=False):
    metric_rows = []
    quality_rows = []
    for seed in seeds:
        for index, policy in enumerate(analysis.POLICY_ORDER):
            signature = f"pool-{seed}"
            if mismatched_signature and seed == seeds[0] and policy == "feedback":
                signature = "different"
            quality_rows.append(
                {
                    "run_id": f"pool{seed}_{policy}",
                    "pool_seed": seed,
                    "policy": policy,
                    "configuration_sha256": "one-config",
                    "pool_signature_sha256": signature,
                    "mesh_cells": 2058,
                    "mesh_vertices": 512,
                }
            )
            for cycle, event_id in enumerate((-1, 0, 1), start=0):
                metric_rows.append(
                    {
                        "pool_seed": seed,
                        "policy": policy,
                        "cycle": cycle,
                        "event_id": event_id,
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(quality_rows)


class AnalysisInvariantTests(unittest.TestCase):
    def test_exact_sign_flip_known_case(self):
        self.assertAlmostEqual(
            analysis.exact_sign_flip_pvalue(np.ones(4)),
            2.0 / 16.0,
        )

    def test_pairing_accepts_complete_exact_blocks(self):
        metrics, quality = paired_frames()
        analysis.validate_pairing(metrics, quality, [4101, 4102])

    def test_pairing_rejects_missing_expected_seed(self):
        metrics, quality = paired_frames(seeds=(4101,))
        with self.assertRaisesRegex(RuntimeError, "Pool-seed set mismatch"):
            analysis.validate_pairing(metrics, quality, [4101, 4102])

    def test_pairing_rejects_different_exact_pool(self):
        metrics, quality = paired_frames(mismatched_signature=True)
        with self.assertRaisesRegex(RuntimeError, "exact same direction/force pool"):
            analysis.validate_pairing(metrics, quality, [4101, 4102])

    def test_pairing_rejects_wrong_expected_config(self):
        metrics, quality = paired_frames()
        with self.assertRaisesRegex(RuntimeError, "expected publication config"):
            analysis.validate_pairing(
                metrics, quality, [4101, 4102], expected_config_hash="different"
            )


if __name__ == "__main__":
    unittest.main()
