"""
KRNA / Statistical Significance Testing
========================================
Non-parametric hypothesis testing for benchmark comparisons, implemented with
NumPy only (no SciPy dependency).

The centerpiece is the Wilcoxon rank-sum test (equivalent to the
Mann-Whitney U test), the standard tool in the metaheuristics literature for
comparing two algorithms over multiple independent runs on one problem
(Derrac et al., "A practical tutorial on the use of nonparametric statistical
tests as a methodology for comparing evolutionary and swarm intelligence
algorithms", Swarm and Evolutionary Computation, 2011).

Hypotheses
----------
H0: the two samples come from the same distribution (no difference between
the algorithms). H1 (two-sided): they come from different distributions.

Method
------
Normal approximation with tie correction, which is the accepted variant for
sample sizes >= ~8 per group (scipy.stats.mannwhitneyu uses the same
continuity-corrected approximation when ``method='asymptotic'``):

    U  = min(U1, U2), chosen so smaller U means stronger evidence
    mu = n1 * n2 / 2
    sigma = sqrt(n1 * n2 / 12 * ((n + 1) - sum(t^3 - t) / (n * (n - 1))))
    z  = (U - mu + 0.5) / sigma        # continuity correction
    p  = 2 * (1 - Phi(|z|))            # two-sided

where ``t`` are the tie-group sizes. Ties arise constantly on benchmark
landscapes because runs repeatedly hit the same plateau fitness values.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _normal_sf(z: float) -> float:
    """Standard-normal survival function 1 - Phi(z), via math.erfc."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _rank_data(values: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Ranks (1-based, ascending) with average ranks assigned within tie groups.

    Returns (ranks, tie_correction) where tie_correction is
    sum(t^3 - t) over tie groups of size t (0.0 when there are no ties).
    """
    order = np.argsort(values, kind="stable")
    sorted_vals = values[order]
    ranks_sorted = np.empty(len(values), dtype=np.float64)

    i = 0
    tie_correction = 0.0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # average of 1-based ranks i+1..j+1
        ranks_sorted[i : j + 1] = avg_rank
        group_size = j - i + 1
        if group_size > 1:
            tie_correction += group_size**3 - group_size
        i = j + 1

    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks, tie_correction


@dataclass(frozen=True)
class RankSumResult:
    """Outcome of a two-sided Wilcoxon rank-sum (Mann-Whitney U) test."""

    u_statistic: float
    p_value: float
    n_sample1: int
    n_sample2: int

    @property
    def is_significant(self) -> bool:
        """Significant at the conventional alpha = 0.05 threshold."""
        return self.p_value < 0.05


def wilcoxon_rank_sum(
    sample1: Sequence[float],
    sample2: Sequence[float],
    alpha: float = 0.05,
) -> RankSumResult:
    """
    Two-sided Wilcoxon rank-sum test (Mann-Whitney U) between two samples.

    Args:
        sample1: Fitness values of algorithm 1 across independent runs.
        sample2: Fitness values of algorithm 2 across independent runs.
        alpha: Significance threshold used by :attr:`RankSumResult.is_significant`.

    Returns:
        RankSumResult with the U statistic and two-sided p-value.

    Raises:
        ValueError: If either sample is empty, or the normal approximation
            is requested with fewer than 8 observations per group.
    """
    x = np.asarray(sample1, dtype=np.float64).ravel()
    y = np.asarray(sample2, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if x.size == 0 or y.size == 0:
        raise ValueError("Both samples must contain at least one finite value")
    if x.size < 8 or y.size < 8:
        raise ValueError(
            f"Normal approximation requires >= 8 observations per group, "
            f"got {x.size} and {y.size}"
        )

    n1, n2 = x.size, y.size
    combined = np.concatenate([x, y])
    ranks, tie_correction = _rank_data(combined)

    r1 = float(np.sum(ranks[:n1]))
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    n = n1 + n2
    mu = n1 * n2 / 2.0
    sigma_sq = (n1 * n2 / 12.0) * ((n + 1) - tie_correction / (n * (n - 1)))
    if sigma_sq <= 0.0:
        # Degenerate: all values identical across both samples.
        return RankSumResult(u_statistic=u, p_value=1.0, n_sample1=int(n1), n_sample2=int(n2))
    sigma = math.sqrt(sigma_sq)

    # Continuity correction toward the null mean, as in scipy's asymptotic MWU.
    z = (abs(u - mu) - 0.5) / sigma
    p_value = 2.0 * _normal_sf(z)
    p_value = float(min(1.0, max(0.0, p_value)))

    return RankSumResult(u_statistic=float(u), p_value=p_value, n_sample1=int(n1), n_sample2=int(n2))


def bonferroni_correct(p_values: Sequence[float], num_comparisons: int | None = None) -> np.ndarray:
    """
    Bonferroni correction controlling the family-wise error rate.

    Args:
        p_values: Raw p-values from m comparisons.
        num_comparisons: Family size m. Defaults to ``len(p_values)``.

    Returns:
        Adjusted p-values (capped at 1.0), in input order.
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = num_comparisons if num_comparisons is not None else p.size
    if m < 1:
        raise ValueError("num_comparisons must be >= 1")
    return np.minimum(p * m, 1.0)


def cohens_r(sample1: Sequence[float], sample2: Sequence[float]) -> float:
    """
    Effect size r = |Z| / sqrt(N) for a rank-sum comparison.

    Interpretation (Cohen): ~0.1 small, ~0.3 medium, ~0.5 large. Returned as
    a non-negative magnitude; direction of the difference should be judged
    from the sample medians.
    """
    x = np.asarray(sample1, dtype=np.float64).ravel()
    y = np.asarray(sample2, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        raise ValueError("Both samples must contain at least one finite value")

    result = wilcoxon_rank_sum(x, y)
    n = result.n_sample1 + result.n_sample2
    if result.p_value >= 1.0:
        return 0.0
    z = _z_from_two_sided_p(result.p_value)  # invert the two-sided p back to |z|
    return z / math.sqrt(n)


def _z_from_two_sided_p(p: float) -> float:
    """Solve 2 * (1 - Phi(z)) = p for z >= 0 by bisection."""
    lo, hi = 0.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if 0.5 * math.erfc(mid / math.sqrt(2.0)) > p / 2.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0
