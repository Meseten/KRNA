"""
KRNA / SKROA Novel Mathematical Operators & Kernels
==============================================================================
Provides vectorized HPC kernels for:
  1. Lévy Flight step generation via Mantegna's algorithm (State 0 Exploration).
  2. Sympodial Clamping anti-collision repulsion vector (Swarm diversity).
  3. Culm-Abortion dynamic memory pruning and trajectory reallocation.
"""

from __future__ import annotations
import numpy as np
from scipy.special import gamma


def levy_flight_step(
    shape: tuple[int, ...],
    beta: float = 1.5,
    scale: float = 1.0,
    rng: np.random.Generator | None = None
) -> np.ndarray:
    """
    Generates vectorized Lévy Flight step increments using Mantegna's algorithm.

    Formula:
        step = scale * (u / |v|^(1 / beta))
        u ~ N(0, sigma_u^2), v ~ N(0, 1)
        sigma_u = [ (Gamma(1+beta) * sin(pi*beta/2)) /
                    (Gamma((1+beta)/2) * beta * 2^((beta-1)/2)) ]^(1/beta)

    Args:
        shape: Output array shape, typically (N, D).
        beta: Stability parameter in range (0.0, 2.0]. Standard default is 1.5.
        scale: Multiplicative scaling factor for step magnitude.
        rng: NumPy random generator instance.

    Returns:
        Array of step increments with shape matching `shape`.
    """
    if not 0.0 < beta <= 2.0:
        raise ValueError(f"Lévy flight beta parameter must be in (0, 2], got {beta}")

    if rng is None:
        rng = np.random.default_rng()

    num = gamma(1.0 + beta) * np.sin(np.pi * beta / 2.0)
    den = gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0))
    sigma_u = (num / den) ** (1.0 / beta)

    u = rng.normal(loc=0.0, scale=sigma_u, size=shape)
    v = rng.normal(loc=0.0, scale=1.0, size=shape)

    # Avoid zero division in denominator
    step = scale * (u / (np.abs(v) ** (1.0 / beta) + 1e-12))
    return step


def apply_sympodial_clamping(
    positions: np.ndarray,
    fitness: np.ndarray,
    epsilon_clamp: float,
    eta: float = 0.1,
    delta: float = 1e-8,
    bounds: tuple[float, float] | None = None
) -> np.ndarray:
    """
    Applies vectorized Sympodial Clamping (repulsion vector) to prevent agent
    collision and premature convergence around local minima.

    For all pairs (i, j) where ||x_i - x_j||_2 < epsilon_clamp and f(x_i) > f(x_j)
    (i is weaker in minimization), agent i experiences repulsion:
        v_ij = eta * (x_i - x_j) / (||x_i - x_j||_2^2 + delta)

    Args:
        positions: Swarm coordinate matrix of shape (N, D).
        fitness: Scalar fitness array of shape (N,).
        epsilon_clamp: Minimum spacing threshold triggering repulsion.
        eta: Repulsion magnitude coefficient.
        delta: Conditioning scalar to prevent division by zero.
        bounds: Optional (min_bound, max_bound) tuple to clip displaced coordinates.

    Returns:
        Updated swarm coordinate matrix of shape (N, D).
    """
    positions_arr = np.asarray(positions, dtype=np.float64)
    fitness_arr = np.asarray(fitness, dtype=np.float64)
    n_agents, dim = positions_arr.shape

    if n_agents <= 1:
        return positions_arr.copy()

    # Shape (N, N, D): diff[i, j, :] = x_i - x_j
    diff = positions_arr[:, np.newaxis, :] - positions_arr[np.newaxis, :, :]

    # Shape (N, N): squared Euclidean distances
    dist_sq = np.sum(diff ** 2, axis=-1)
    dist = np.sqrt(dist_sq)

    # Collision condition: distance within threshold and non-self-referencing
    mask_distance = (dist < epsilon_clamp) & (dist > 0.0)

    # Weaker-agent condition: i has worse (higher) fitness than j
    mask_weaker = fitness_arr[:, np.newaxis] > fitness_arr[np.newaxis, :]

    # Composite active mask of shape (N, N)
    active_mask = mask_distance & mask_weaker

    # Compute scalar repulsion factor matrix of shape (N, N)
    repulsion_factors = np.where(active_mask, eta / (dist_sq + delta), 0.0)

    # Sum displacement contributions along column axis -> Shape (N, D)
    displacement = np.sum(diff * repulsion_factors[:, :, np.newaxis], axis=1)

    updated_positions = positions_arr + displacement

    if bounds is not None:
        low, high = bounds
        updated_positions = np.clip(updated_positions, low, high)

    return updated_positions


def apply_culm_abortion(
    positions: np.ndarray,
    current_fitness: np.ndarray,
    previous_fitness: np.ndarray,
    states: np.ndarray,
    stagnation_counters: np.ndarray,
    global_best_position: np.ndarray,
    tau_stagnation: float,
    max_stagnation_steps: int,
    bounds: tuple[float, float],
    dispersion_scale: float = 0.3,
    rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies dynamic memory pruning (Culm-Abortion) to agents stuck in local
    exploitation basins.

    If an agent is in State 1 (Vertical Shoot) and absolute improvement is below
    tau_stagnation for max_stagnation_steps consecutive evaluations, its thread
    is aborted: state resets to 0 (Rhizome Creep), counter resets to 0, and
    coordinates are reallocated around `global_best_position`.

    Args:
        positions: Current agent positions of shape (N, D).
        current_fitness: Latest evaluation fitness array of shape (N,).
        previous_fitness: Previous evaluation fitness array of shape (N,).
        states: Biphasic state indicator array of shape (N,) in {0, 1}.
        stagnation_counters: Counter array of shape (N,) tracking stalled iterations.
        global_best_position: Coordinates of known best solution of shape (D,).
        tau_stagnation: Minimum absolute gradient/fitness improvement required.
        max_stagnation_steps: Maximum allowed stalled evaluations before abortion.
        bounds: (min_bound, max_bound) tuple defining valid search space.
        dispersion_scale: Standard deviation ratio for Gaussian respawn around best.
        rng: NumPy random generator instance.

    Returns:
        Tuple of (updated_positions, updated_states, updated_counters, aborted_mask):
          - updated_positions: (N, D) array with aborted agents reallocated.
          - updated_states: (N,) state array with aborted agents reset to 0.
          - updated_counters: (N,) stagnation counters array.
          - aborted_mask: (N,) boolean mask indicating which agents were aborted.
    """
    if rng is None:
        rng = np.random.default_rng()

    pos_out = np.copy(positions)
    states_out = np.copy(states)
    counters_out = np.copy(stagnation_counters)

    # 1. Measure absolute fitness progress
    delta_f = np.abs(current_fitness - previous_fitness)

    # 2. Identify State 1 agents experiencing gradient stagnation
    stagnant_mask = (states_out == 1) & (delta_f < tau_stagnation)
    progress_mask = (states_out == 1) & (delta_f >= tau_stagnation)

    # 3. Update stagnation counters
    counters_out[stagnant_mask] += 1
    counters_out[progress_mask] = 0
    counters_out[states_out == 0] = 0  # State 0 agents do not accumulate exploitation stagnation

    # 4. Trigger Culm-Abortion for threshold breakers
    aborted_mask = (states_out == 1) & (counters_out >= max_stagnation_steps)
    num_aborted = int(np.sum(aborted_mask))

    if num_aborted > 0:
        low, high = bounds
        domain_range = high - low
        sigma = dispersion_scale * domain_range

        # Reallocate around global best with Gaussian dispersion
        respawn_offsets = rng.normal(loc=0.0, scale=sigma, size=(num_aborted, pos_out.shape[1]))
        new_coords = np.expand_dims(global_best_position, axis=0) + respawn_offsets

        pos_out[aborted_mask] = np.clip(new_coords, low, high)
        states_out[aborted_mask] = 0
        counters_out[aborted_mask] = 0

    return pos_out, states_out, counters_out, aborted_mask