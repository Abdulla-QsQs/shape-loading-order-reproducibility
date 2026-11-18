#!/usr/bin/env python3
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matched_pool_study as study  # noqa: E402


class StudyInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(
            (ROOT / "config" / "publication_study.json").read_text(encoding="utf-8")
        )

    def test_balanced_pool_is_unit_length_and_zero_sum(self):
        pool = study.generate_balanced_load_pool(24, 1001, 1.0e6)
        directions = np.vstack([event.direction for event in pool])
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-14)
        np.testing.assert_allclose(directions.sum(axis=0), 0.0, atol=1e-14)

    def test_random_policy_consumes_every_event_once(self):
        pool = study.generate_balanced_load_pool(24, 1001, 1.0e6)
        state = study.GeometryState(
            center_of_mass=np.zeros(3),
            covariance=np.eye(3),
            normalized_shape_change=np.zeros((3, 3)),
            principal_values=np.ones(3),
            principal_axes=np.eye(3),
        )
        rng = np.random.default_rng(55)
        remaining = set(range(len(pool)))
        selected = []
        while remaining:
            result = study.select_random(
                sorted(remaining), pool, state, rng, self.config
            )
            selected.append(result.event_id)
            remaining.remove(result.event_id)
        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(set(selected), set(range(len(pool))))

    def test_feedback_is_uniform_for_reference_geometry(self):
        pool = study.generate_balanced_load_pool(24, 1001, 1.0e6)
        state = study.GeometryState(
            center_of_mass=np.zeros(3),
            covariance=np.eye(3),
            normalized_shape_change=np.zeros((3, 3)),
            principal_values=np.ones(3),
            principal_axes=np.eye(3),
        )
        candidates, probabilities, scores = study.feedback_probabilities(
            range(len(pool)), pool, state, self.config
        )
        self.assertEqual(len(candidates), len(pool))
        np.testing.assert_allclose(probabilities, 1.0 / len(pool), atol=1e-14)
        np.testing.assert_allclose(scores, 0.0, atol=1e-14)

    def test_feedback_responds_to_geometry_magnitude(self):
        pool = study.generate_balanced_load_pool(24, 1001, 1.0e6)
        weak = study.GeometryState(
            center_of_mass=np.array([1e-6, 0.0, 0.0]),
            covariance=np.eye(3),
            normalized_shape_change=np.diag([1e-5, -5e-6, -5e-6]),
            principal_values=np.ones(3),
            principal_axes=np.eye(3),
        )
        strong = study.GeometryState(
            center_of_mass=100.0 * weak.center_of_mass,
            covariance=np.eye(3),
            normalized_shape_change=100.0 * weak.normalized_shape_change,
            principal_values=np.ones(3),
            principal_axes=np.eye(3),
        )
        _, p_weak, _ = study.feedback_probabilities(range(len(pool)), pool, weak, self.config)
        _, p_strong, _ = study.feedback_probabilities(range(len(pool)), pool, strong, self.config)
        self.assertGreater(1.0 - study.normalized_entropy(p_strong), 1.0 - study.normalized_entropy(p_weak))

    def test_cap_areas_are_nonzero_and_disjoint(self):
        mesh = study.create_mapped_sphere(0.05, 3)
        markers, impact, support, total = study.mark_directional_caps(
            mesh, np.zeros(3), np.array([1.0, 0.0, 0.0]), 0.04
        )
        self.assertGreater(impact, 0.0)
        self.assertGreater(support, 0.0)
        self.assertGreater(total, impact + support)
        values = markers.array()
        self.assertGreater(np.count_nonzero(values == study.IMPACT_MARKER), 0)
        self.assertGreater(np.count_nonzero(values == study.SUPPORT_MARKER), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
