"""
KRNA / Bounds-Aware Finite-Difference Gradients
================================================
Shared probing kernel for the gradient-guided exploitation phase of SKROA
(single-objective) and MO-SKROA (scalarized multi-objective).

Every perturbed probe point is guaranteed to stay inside the declared search
bounds: user objective functions may be undefined outside them (log of
non-positives, sqrt of negatives, division by a span term, ...). Probe
directions are chosen per coordinate — +h where there is room above, -h where
there is room below, and no probe where the coordinate cannot move by h in
either direction (gradient reported as 0 there).
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def bounds_aware_gradient(
    evaluator: Callable[[np.ndarray], np.ndarray],
    positions: np.ndarray,
    base_fitness: np.ndarray,
    low: float,
    high: float,
    h: float,
) -> np.ndarray:
    """
    Vectorized forward/backward finite-difference gradient for N agents.

    Args:
        evaluator: Callable mapping an (M, D) position matrix to (M,) scalar
            fitness values (already scalarized for multi-objective use).
        positions: Agent coordinates of shape (N, D).
        base_fitness: Scalar fitness at ``positions`` of shape (N,).
        low: Lower search bound (shared across dimensions).
        high: Upper search bound (shared across dimensions).
        h: Finite-difference step size.

    Returns:
        Gradient estimates of shape (N, D); entries with no feasible probe
        direction are 0.
    """
    n1, d = positions.shape
    if n1 == 0:
        return np.empty((0, d), dtype=np.float64)

    # Sign per (agent, coordinate): +1 probe up, -1 probe down, 0 no probe
    probe_sign = np.where(
        (high - positions) >= h,
        1.0,
        np.where((positions - low) >= h, -1.0, 0.0),
    )

    pos_expanded = np.tile(positions[:, np.newaxis, :], (1, d, 1))
    perturbation = np.eye(d) * h * probe_sign[:, np.newaxis, :]
    pos_perturbed = pos_expanded + perturbation

    flat_perturbed = pos_perturbed.reshape(n1 * d, d)
    flat_fitness = np.asarray(evaluator(flat_perturbed), dtype=np.float64)

    f_perturbed = flat_fitness.reshape(n1, d)
    denom = h * probe_sign
    grad = (f_perturbed - base_fitness[:, np.newaxis]) / np.where(denom == 0.0, 1.0, denom)
    grad[probe_sign == 0.0] = 0.0

    return grad
