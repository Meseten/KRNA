"""
KRNA / SKROA Engine Verification Suite
=======================================
Behavioral tests for the SKROA optimizer: convergence, reproducibility,
evaluator contract enforcement, gradient probing inside bounds, and
boundary/shape validation.
"""

from __future__ import annotations
import unittest
import numpy as np

from krna import SKROA
from krna.benchmarks import rastrigin, ackley, rosenbrock


class TestSKROAEngine(unittest.TestCase):

    def test_converges_on_quadratic_bowl(self) -> None:
        """SKROA must beat a random search baseline on a convex landscape."""
        optimizer = SKROA(
            evaluator=lambda X: np.sum(X**2, axis=1),
            bounds=(-10.0, 10.0),
            dim=5,
            n_agents=30,
            max_iters=200,
            seed=7,
        )
        res = optimizer.optimize()
        self.assertTrue(np.isfinite(res["g_best_fit"]))
        self.assertLess(res["g_best_fit"], 1.0, "Failed to converge on sphere function")

    def test_reproducible_with_same_seed(self) -> None:
        """Identical seeds must produce identical best-fitness trajectories."""

        def run() -> np.ndarray:
            opt = SKROA(
                evaluator=rastrigin,
                bounds=(-5.12, 5.12),
                dim=5,
                n_agents=20,
                max_iters=80,
                seed=123,
            )
            return opt.optimize()["convergence_curve"]

        np.testing.assert_array_equal(run(), run())

    def test_evaluator_shape_mismatch_rejected(self) -> None:
        """An evaluator returning the wrong shape must fail loudly at startup."""

        def bad_evaluator(X: np.ndarray) -> np.ndarray:
            return np.zeros(X.shape[0] + 1)  # wrong length

        opt = SKROA(
            evaluator=bad_evaluator,
            bounds=(-1.0, 1.0),
            dim=3,
            n_agents=10,
            max_iters=5,
            seed=1,
        )
        with self.assertRaises(ValueError):
            opt.optimize()

    def test_gradient_probes_respect_bounds(self) -> None:
        """Every point handed to the evaluator must lie inside the box bounds."""
        low, high = -1.0, 2.0
        seen: list[np.ndarray] = []

        def recording_evaluator(X: np.ndarray) -> np.ndarray:
            seen.append(np.array(X, copy=True))
            return np.sum(X**2, axis=1)

        opt = SKROA(
            evaluator=recording_evaluator,
            bounds=(low, high),
            dim=4,
            n_agents=15,
            max_iters=30,
            seed=3,
        )
        opt.optimize()

        all_points = np.vstack(seen)
        self.assertTrue(
            np.all(all_points >= low) and np.all(all_points <= high),
            f"Evaluator saw out-of-bounds points: min={all_points.min()}, max={all_points.max()}",
        )

    def test_constructor_rejects_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            SKROA(evaluator=rastrigin, bounds=(-1.0, 1.0), dim=0)
        with self.assertRaises(ValueError):
            SKROA(evaluator=rastrigin, bounds=(-1.0, 1.0), dim=3, n_agents=0)
        with self.assertRaises(ValueError):
            SKROA(evaluator=rastrigin, bounds=(1.0, -1.0), dim=3)  # high < low
        with self.assertRaises(ValueError):
            SKROA(evaluator=rastrigin, bounds=(-1.0, 1.0), dim=3, max_iters=0)

    def test_result_contract(self) -> None:
        opt = SKROA(
            evaluator=ackley,
            bounds=(-32.768, 32.768),
            dim=4,
            n_agents=10,
            max_iters=20,
            seed=11,
        )
        res = opt.optimize()
        for key in ("g_best_pos", "g_best_fit", "convergence_curve", "exec_time_ms", "total_aborts"):
            self.assertIn(key, res)
        self.assertEqual(res["g_best_pos"].shape, (4,))
        self.assertEqual(res["convergence_curve"].shape, (20,))
        self.assertFalse(np.any(np.isnan(res["convergence_curve"])))
        self.assertGreaterEqual(res["exec_time_ms"], 0.0)


class TestBenchmarkFunctions(unittest.TestCase):
    """Known-value checks for the canonical landscapes."""

    def test_rastrigin_global_optimum(self) -> None:
        self.assertAlmostEqual(float(rastrigin(np.zeros(10))), 0.0, places=9)

    def test_ackley_global_optimum(self) -> None:
        self.assertAlmostEqual(float(ackley(np.zeros(8))), 0.0, places=9)

    def test_rosenbrock_global_optimum(self) -> None:
        self.assertAlmostEqual(float(rosenbrock(np.ones(6))), 0.0, places=9)

    def test_1d_and_2d_inputs_agree(self) -> None:
        x = np.array([0.5, -1.25, 2.0])
        np.testing.assert_allclose(rastrigin(x), rastrigin(x[np.newaxis, :])[0])
        np.testing.assert_allclose(ackley(x), ackley(x[np.newaxis, :])[0])
        np.testing.assert_allclose(rosenbrock(x), rosenbrock(x[np.newaxis, :])[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
