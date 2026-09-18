"""
KRNA / Statistical Testing Verification Suite
==============================================
Validates the scipy-free Wilcoxon rank-sum implementation against known
analytical values, hand-computable cases, and (when available) scipy itself.
"""

from __future__ import annotations
import unittest
import numpy as np

from krna.stats import wilcoxon_rank_sum, bonferroni_correct, cohens_r


def _sample_with_noise(rng: np.random.Generator, n: int, loc: float) -> np.ndarray:
    return rng.normal(loc=loc, scale=1.0, size=n)


class TestWilcoxonRankSum(unittest.TestCase):

    def test_identical_distributions_not_significant(self) -> None:
        """Two samples from the same distribution should rarely reject H0."""
        rng = np.random.default_rng(0)
        rejections = 0
        for _ in range(50):
            x = _sample_with_noise(rng, 20, 0.0)
            y = _sample_with_noise(rng, 20, 0.0)
            if wilcoxon_rank_sum(x, y).is_significant:
                rejections += 1
        # alpha = 0.05: expect ~2.5 rejections in 50; tolerate up to 8
        self.assertLessEqual(rejections, 8)

    def test_shifted_distribution_detected(self) -> None:
        """A large location shift must be detected with tiny p-value."""
        rng = np.random.default_rng(1)
        x = _sample_with_noise(rng, 40, 0.0)
        y = _sample_with_noise(rng, 40, 5.0)
        result = wilcoxon_rank_sum(x, y)
        self.assertTrue(result.is_significant)
        self.assertLess(result.p_value, 1e-6)

    def test_ties_handled(self) -> None:
        """Heavy ties (plateau fitness values) must not crash or produce NaN."""
        x = np.array([1.0] * 10 + [2.0] * 5)
        y = np.array([2.0] * 10 + [3.0] * 5)
        result = wilcoxon_rank_sum(x, y)
        self.assertTrue(np.isfinite(result.p_value))
        self.assertGreaterEqual(result.p_value, 0.0)
        self.assertLessEqual(result.p_value, 1.0)

    def test_all_identical_values_gives_p_one(self) -> None:
        x = np.ones(10)
        y = np.ones(12)
        result = wilcoxon_rank_sum(x, y)
        self.assertEqual(result.p_value, 1.0)
        self.assertFalse(result.is_significant)

    def test_symmetry_in_sample_order(self) -> None:
        """U and p must be identical regardless of which sample comes first."""
        rng = np.random.default_rng(2)
        x = _sample_with_noise(rng, 15, 1.0)
        y = _sample_with_noise(rng, 15, 0.0)
        a = wilcoxon_rank_sum(x, y)
        b = wilcoxon_rank_sum(y, x)
        self.assertAlmostEqual(a.u_statistic, b.u_statistic)
        self.assertAlmostEqual(a.p_value, b.p_value)

    def test_hand_computed_small_case(self) -> None:
        """
        x = {1,2,3,4,5,6,7,8}, y = {9,...,16}: complete separation.
        All of x ranks below all of y -> U1 = 0, U = 0.
        """
        result = wilcoxon_rank_sum(np.arange(1, 9), np.arange(9, 17))
        self.assertEqual(result.u_statistic, 0.0)
        self.assertLess(result.p_value, 0.001)

    def test_minimum_sample_size_enforced(self) -> None:
        with self.assertRaises(ValueError):
            wilcoxon_rank_sum(np.ones(7), np.ones(10))
        with self.assertRaises(ValueError):
            wilcoxon_rank_sum([], np.ones(10))

    def test_matches_scipy_when_available(self) -> None:
        """Cross-check against scipy.stats.mannwhitneyu (asymptotic, ties)."""
        try:
            from scipy.stats import mannwhitneyu
        except ImportError:
            self.skipTest("scipy not installed")
        rng = np.random.default_rng(3)
        x = np.round(_sample_with_noise(rng, 25, 0.0), 1)  # rounding forces ties
        y = np.round(_sample_with_noise(rng, 25, 0.7), 1)
        ours = wilcoxon_rank_sum(x, y)
        ref = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
        self.assertAlmostEqual(ours.u_statistic, ref.statistic, places=6)
        self.assertAlmostEqual(ours.p_value, ref.pvalue, places=8)


class TestBonferroni(unittest.TestCase):

    def test_scales_and_caps(self) -> None:
        adjusted = bonferroni_correct([0.01, 0.04, 0.5], num_comparisons=3)
        np.testing.assert_allclose(adjusted, [0.03, 0.12, 1.0])

    def test_default_family_size(self) -> None:
        np.testing.assert_allclose(bonferroni_correct([0.2, 0.9]), [0.4, 1.0])

    def test_invalid_family_size(self) -> None:
        with self.assertRaises(ValueError):
            bonferroni_correct([0.1], num_comparisons=0)


class TestCohensR(unittest.TestCase):

    def test_small_effect_for_identical_distributions(self) -> None:
        """Same-distribution samples must show at most a small effect."""
        rng = np.random.default_rng(4)
        x = _sample_with_noise(rng, 20, 0.0)
        y = _sample_with_noise(rng, 20, 0.0)
        self.assertLess(cohens_r(x, y), 0.3)

    def test_large_effect_for_big_shift(self) -> None:
        rng = np.random.default_rng(5)
        x = _sample_with_noise(rng, 50, 0.0)
        y = _sample_with_noise(rng, 50, 4.0)
        r = cohens_r(x, y)
        self.assertGreater(r, 0.5)  # "large" threshold
        self.assertLessEqual(r, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
