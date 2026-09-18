"""
KRNA / Benchmark Statistical Integration Tests
===============================================
Verifies the summarize_comparison helper and the MO-SKROA bounds-aware
scalarized gradient.
"""

from __future__ import annotations
import unittest
import numpy as np

from krna.benchmarks import summarize_comparison
from krna import MOSKROA
from krna.mo_benchmarks import zdt1


class TestSummarizeComparison(unittest.TestCase):

    def test_skroa_wins_significantly(self) -> None:
        skroa = np.array([0.1, 0.12, 0.11, 0.09, 0.1, 0.13, 0.1, 0.11, 0.1, 0.12])
        pso = np.array([5.0, 6.1, 4.8, 5.5, 5.2, 4.9, 5.3, 5.7, 5.1, 5.4])
        stats = summarize_comparison(skroa, pso)
        self.assertEqual(stats["verdict"], "SKROA significantly better")
        self.assertLess(stats["p_value"], 0.05)
        self.assertGreater(stats["effect_size_r"], 0.5)

    def test_no_significant_difference_for_identical_distributions(self) -> None:
        rng = np.random.default_rng(0)
        skroa = rng.normal(1.0, 0.5, size=20)
        pso = rng.normal(1.0, 0.5, size=20)
        stats = summarize_comparison(skroa, pso)
        self.assertEqual(stats["verdict"], "no significant difference")

    def test_pso_wins_significantly(self) -> None:
        skroa = np.array([5.0, 6.1, 4.8, 5.5, 5.2, 4.9, 5.3, 5.7, 5.1, 5.4])
        pso = np.array([0.1, 0.12, 0.11, 0.09, 0.1, 0.13, 0.1, 0.11, 0.1, 0.12])
        stats = summarize_comparison(skroa, pso)
        self.assertEqual(stats["verdict"], "PSO significantly better")

    def test_too_few_trials_is_inconclusive_not_fabricated(self) -> None:
        stats = summarize_comparison(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        self.assertEqual(stats["verdict"], "inconclusive")
        self.assertTrue(np.isnan(stats["p_value"]))


class TestMOSKROABoundsAwareGradient(unittest.TestCase):

    def test_probes_respect_bounds(self) -> None:
        """Every point handed to the MO evaluator must lie inside [0, 1]."""
        seen: list[np.ndarray] = []

        def recording_evaluator(X: np.ndarray) -> np.ndarray:
            seen.append(np.array(X, copy=True))
            return zdt1(X)

        opt = MOSKROA(
            evaluator=recording_evaluator,
            bounds=(0.0, 1.0),
            dim=5,
            n_agents=12,
            max_iters=15,
            seed=9,
        )
        opt.optimize()

        all_points = np.vstack(seen)
        self.assertTrue(
            np.all(all_points >= 0.0) and np.all(all_points <= 1.0),
            f"MO evaluator saw out-of-bounds points: min={all_points.min()}, max={all_points.max()}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
