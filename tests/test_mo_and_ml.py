"""
KRNA / MO-SKROA and ML Tuner Verification Suite
================================================
Tests for non-dominated sorting, archive integrity, ZDT benchmarks,
and the continuous-to-hyperparameter mapper.
"""

from __future__ import annotations
import unittest
import numpy as np

from krna import MOSKROA, Archive, get_non_dominated_mask
from krna.mo_benchmarks import zdt1, zdt2
from krna.ml_tuning import HyperparameterMapper


class TestNonDominatedMask(unittest.TestCase):

    def test_identifies_dominated_points(self) -> None:
        # Point 1 is dominated by point 0 (worse in both objectives)
        F = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [0.5, 3.0],  # trade-off: non-dominated
        ])
        mask = get_non_dominated_mask(F)
        self.assertTrue(mask[0])
        self.assertFalse(mask[1])
        self.assertTrue(mask[2])

    def test_all_equal_points_are_all_non_dominated(self) -> None:
        F = np.ones((4, 2))
        self.assertTrue(np.all(get_non_dominated_mask(F)))


class TestArchive(unittest.TestCase):

    def test_archive_grows_and_prunes(self) -> None:
        archive = Archive()
        archive.update(
            np.array([[0.0], [1.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
        )
        self.assertEqual(len(archive.positions), 2)

        # A dominated point must not enter the archive
        archive.update(np.array([[0.5]]), np.array([[1.0, 1.0]]))
        self.assertEqual(len(archive.positions), 2)

        # A dominating point evicts everything it beats
        archive.update(np.array([[9.0]]), np.array([[0.0, 0.0]]))
        self.assertEqual(len(archive.positions), 1)
        np.testing.assert_allclose(archive.fitness[0], [0.0, 0.0])

    def test_archive_shapes_stay_2d(self) -> None:
        archive = Archive()
        archive.update(np.zeros((3, 2)), np.zeros((3, 2)))
        self.assertEqual(archive.positions.shape, (3, 2))
        self.assertEqual(archive.fitness.shape, (3, 2))


class TestZDTBenchmarks(unittest.TestCase):

    def test_zdt1_known_front_values(self) -> None:
        # On the front (g=1): f2 = 1 - sqrt(f1); x_i = 0 for i >= 2
        X = np.array([[0.0, 0.0], [0.25, 0.0], [1.0, 0.0]])
        F = zdt1(X)
        np.testing.assert_allclose(F[:, 0], [0.0, 0.25, 1.0], atol=1e-12)
        np.testing.assert_allclose(F[:, 1], [1.0, 0.5, 0.0], atol=1e-12)

    def test_zdt2_known_front_values(self) -> None:
        # On the front (g=1): f2 = 1 - f1^2; x_i = 0 for i >= 2
        X = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
        F = zdt2(X)
        np.testing.assert_allclose(F[:, 1], [1.0, 0.75, 0.0], atol=1e-12)

    def test_zdt_rejects_dim_below_two(self) -> None:
        with self.assertRaises(ValueError):
            zdt1(np.zeros(1))
        with self.assertRaises(ValueError):
            zdt2(np.zeros(1))


class TestMOSKROAEngine(unittest.TestCase):

    def test_produces_finite_pareto_front(self) -> None:
        opt = MOSKROA(
            evaluator=zdt1,
            bounds=(0.0, 1.0),
            dim=8,
            n_agents=40,
            max_iters=30,
            seed=5,
        )
        res = opt.optimize()
        pf = res["pareto_front_fitness"]
        self.assertGreater(res["archive_size"], 0)
        self.assertEqual(pf.shape[1], 2)
        self.assertTrue(np.all(np.isfinite(pf)))
        self.assertTrue(np.all(pf >= 0.0))
        # Front must actually span objective 1
        self.assertGreater(pf[:, 0].max() - pf[:, 0].min(), 0.1)


class TestMLTunerEvaluator(unittest.TestCase):

    def test_evaluate_single_tolerates_estimators_without_random_state(self) -> None:
        """
        The tuner must not crash with TypeError for model classes that do not
        accept random_state. The dummy model lacks fit/predict, so CV scoring
        fails and the documented penalty fitness (1.0) is returned instead.
        """
        from krna.ml_tuning import SKROAMLTuner

        class NoRandomState:
            def __init__(self, C: float = 1.0):
                self.C = C

        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 1))
        y = (X[:, 0] > 0).astype(int)
        tuner = SKROAMLTuner(
            model_class=NoRandomState,
            param_space={"C": {"type": "log_float", "min": 1e-2, "max": 1e2}},
            X=X,
            y=y,
            cv_folds=2,
            n_agents=3,
            max_iters=1,
            seed=0,
        )
        self.assertEqual(tuner._evaluate_single(np.array([0.5])), 1.0)


class TestHyperparameterMapper(unittest.TestCase):

    def test_decode_linear_and_log_scales(self) -> None:
        mapper = HyperparameterMapper({
            "alpha": {"type": "float", "min": 0.0, "max": 1.0},
            "C": {"type": "log_float", "min": 1e-3, "max": 1e3},
        })
        params = mapper.decode(np.array([0.5, 0.5]))
        self.assertAlmostEqual(params["alpha"], 0.5)
        self.assertAlmostEqual(params["C"], 1.0, places=9)  # midpoint of log scale

    def test_decode_categorical_with_edge_value(self) -> None:
        mapper = HyperparameterMapper({
            "kernel": {"type": "categorical", "values": ["a", "b", "c"]},
        })
        self.assertEqual(mapper.decode(np.array([0.0]))["kernel"], "a")
        self.assertEqual(mapper.decode(np.array([0.99]))["kernel"], "c")
        self.assertEqual(mapper.decode(np.array([1.0]))["kernel"], "c")  # edge clamp

    def test_decode_int_respects_bounds(self) -> None:
        mapper = HyperparameterMapper({
            "degree": {"type": "int", "min": 2, "max": 5},
        })
        for v in np.linspace(0.0, 1.0, 11):
            d = mapper.decode(np.array([v]))["degree"]
            self.assertTrue(2 <= d <= 5)
        self.assertEqual(mapper.decode(np.array([0.0]))["degree"], 2)
        self.assertEqual(mapper.decode(np.array([1.0]))["degree"], 5)

    def test_invalid_space_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HyperparameterMapper({})  # empty space
        with self.assertRaises(ValueError):
            HyperparameterMapper({"C": {"type": "log_float", "min": 10.0, "max": 1.0}})
        with self.assertRaises(ValueError):
            HyperparameterMapper({"k": {"type": "categorical", "values": []}})
        with self.assertRaises(ValueError):
            HyperparameterMapper({"x": {"type": "banana"}})


if __name__ == "__main__":
    unittest.main(verbosity=2)
