"""
KRNA / SKROA Core Algorithm Engine
==============================================================================
Implements the Sympodial Kawayan Rhizome Optimization Algorithm (SKROA).
Features a biphasic state machine (Rhizome Creep vs. Vertical Shoot), vectorized
finite-difference gradient exploitation, Sympodial Clamping, and Culm-Abortion.
"""

from __future__ import annotations
import time
from typing import Callable
import numpy as np

from krna.operators import levy_flight_step, apply_sympodial_clamping, apply_culm_abortion


class SKROA:
    """
    Sympodial Kawayan Rhizome Optimization Algorithm (SKROA)
    """

    def __init__(
        self,
        evaluator: Callable[[np.ndarray], np.ndarray],
        bounds: tuple[float, float],
        dim: int,
        n_agents: int = 50,
        max_iters: int = 1000,
        delta_threshold: float | None = None,  # None triggers dynamic median threshold
        tau_stagnation: float = 1e-4,
        max_stagnation_steps: int = 10,
        epsilon_clamp: float = 1e-2,
        gamma_lr: float = 0.05,
        sigma_jitter: float = 1e-4,
        levy_scale: float = 0.01,
        seed: int = 42
    ):
        self.evaluator = evaluator
        self.bounds = bounds
        self.dim = dim
        self.n_agents = n_agents
        self.max_iters = max_iters
        
        self.delta_threshold = delta_threshold
        self.tau_stagnation = tau_stagnation
        self.max_stagnation_steps = max_stagnation_steps
        self.epsilon_clamp = epsilon_clamp
        self.gamma_lr = gamma_lr
        self.sigma_jitter = sigma_jitter
        self.levy_scale = levy_scale
        
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if n_agents < 1:
            raise ValueError(f"n_agents must be >= 1, got {n_agents}")
        if max_iters < 1:
            raise ValueError(f"max_iters must be >= 1, got {max_iters}")
        low, high = bounds
        if not (np.isfinite(low) and np.isfinite(high)) or not high > low:
            raise ValueError(f"bounds must be finite with high > low, got {bounds!r}")
        
        self.rng = np.random.default_rng(seed)

    def _compute_vectorized_gradient(
        self,
        positions: np.ndarray,
        base_fitness: np.ndarray,
        h: float | None = None
    ) -> np.ndarray:
        """
        Computes the forward finite-difference gradient for N_1 agents without
        Python loops over dimensions.

        Probe directions are chosen per coordinate so every perturbed point
        stays inside the declared bounds: +h where there is room above, -h
        where there is room below, and no probe where the coordinate cannot
        move by h in either direction (gradient reported as 0 there).
        """
        n1, d = positions.shape
        if n1 == 0:
            return np.empty((0, d), dtype=np.float64)
        if h is None:
            h = 1e-5 * (self.bounds[1] - self.bounds[0])

        low, high = self.bounds
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
        flat_fitness = self.evaluator(flat_perturbed)

        f_perturbed = np.asarray(flat_fitness, dtype=np.float64).reshape(n1, d)
        denom = h * probe_sign
        grad = (f_perturbed - base_fitness[:, np.newaxis]) / np.where(denom == 0.0, 1.0, denom)
        grad[probe_sign == 0.0] = 0.0
        
        return grad

    def optimize(self) -> dict:
        """
        Executes the SKROA optimization loop.
        """
        low, high = self.bounds
        domain_span = high - low
        v_max = 0.1 * domain_span  # Global velocity clamp to prevent explosion
        
        positions = self.rng.uniform(low, high, size=(self.n_agents, self.dim))
        states = np.zeros(self.n_agents, dtype=int)
        stagnation_counters = np.zeros(self.n_agents, dtype=int)
        
        current_fitness = np.asarray(self.evaluator(positions), dtype=np.float64)
        if current_fitness.shape != (self.n_agents,):
            raise ValueError(
                f"Evaluator must map ({self.n_agents}, {self.dim}) positions to "
                f"shape ({self.n_agents},) fitness; got shape {current_fitness.shape}"
            )
        previous_fitness = np.copy(current_fitness)
        
        best_idx = np.argmin(current_fitness)
        g_best_pos = positions[best_idx].copy()
        g_best_fit = current_fitness[best_idx]
        
        convergence_curve = np.zeros(self.max_iters)
        abort_counts = 0
        start_time = time.perf_counter()

        for it in range(self.max_iters):
            if it > 0:
                current_fitness = np.asarray(self.evaluator(positions), dtype=np.float64)
                min_idx = np.argmin(current_fitness)
                if current_fitness[min_idx] < g_best_fit:
                    g_best_fit = current_fitness[min_idx]
                    g_best_pos = positions[min_idx].copy()

            convergence_curve[it] = g_best_fit

            # 1. Biphasic State Transition (Dynamic Median ensures exactly 50% exploit)
            if self.delta_threshold is not None:
                current_threshold = self.delta_threshold
            else:
                current_threshold = np.median(current_fitness)
                
            states = np.where(current_fitness <= current_threshold, 1, 0)
            
            # 2. Dynamic Memory Pruning (Culm-Abortion)
            positions, states, stagnation_counters, aborted_mask = apply_culm_abortion(
                positions=positions,
                current_fitness=current_fitness,
                previous_fitness=previous_fitness,
                states=states,
                stagnation_counters=stagnation_counters,
                global_best_position=g_best_pos,
                tau_stagnation=self.tau_stagnation,
                max_stagnation_steps=self.max_stagnation_steps,
                bounds=self.bounds,
                rng=self.rng
            )
            abort_counts += np.sum(aborted_mask)
            
            new_positions = np.copy(positions)
            
            # 3. State 0: Rhizome Creep (Lévy Flight Exploration)
            s0_mask = (states == 0)
            if np.any(s0_mask):
                n0 = np.sum(s0_mask)
                # Scale Lévy steps relative to the search space bounds
                l_scale = self.levy_scale * domain_span
                step = levy_flight_step(shape=(n0, self.dim), scale=l_scale, rng=self.rng)
                new_positions[s0_mask] += step
                
            # 4. State 1: Vertical Shoot (Gradient-Guided Exploitation)
            s1_mask = (states == 1)
            if np.any(s1_mask):
                x_s1 = positions[s1_mask]
                f_s1 = current_fitness[s1_mask]
                
                grad = self._compute_vectorized_gradient(x_s1, f_s1)
                
                # Scale learning rate by bounds and strictly clip to prevent explosion
                raw_step = self.gamma_lr * domain_span * grad
                clipped_step = np.clip(raw_step, -v_max, v_max)
                
                jitter_scale = self.sigma_jitter * domain_span
                jitter = self.rng.normal(0.0, jitter_scale, size=x_s1.shape)
                
                new_positions[s1_mask] -= clipped_step - jitter

            # 5. Apply Sympodial Clamping (Anti-Collision Repulsion)
            new_positions = apply_sympodial_clamping(
                positions=new_positions,
                fitness=current_fitness,
                epsilon_clamp=self.epsilon_clamp,
                bounds=self.bounds
            )
            
            # 6. Finalize Iteration
            positions = np.clip(new_positions, low, high)
            previous_fitness = np.copy(current_fitness)

        exec_time = time.perf_counter() - start_time
        
        return {
            "g_best_pos": g_best_pos,
            "g_best_fit": g_best_fit,
            "convergence_curve": convergence_curve,
            "exec_time_ms": exec_time * 1000.0,
            "total_aborts": abort_counts
        }