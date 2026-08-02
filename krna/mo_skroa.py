"""
KRNA / MO-SKROA Engine
==============================================================================
Multi-Objective Sympodial Kawayan Rhizome Optimization Algorithm.
Maintains an external archive of non-dominated solutions to map the Pareto Front.
"""

from __future__ import annotations
import time
from typing import Callable
import numpy as np

from krna.operators import levy_flight_step, apply_sympodial_clamping


def get_non_dominated_mask(fitness_matrix: np.ndarray) -> np.ndarray:
    """
    Identifies non-dominated vectors in an (N, M) fitness matrix (minimization).
    Solution A dominates B if A <= B in all objectives, and A < B in at least one.
    """
    n_samples = fitness_matrix.shape[0]
    is_dominated = np.zeros(n_samples, dtype=bool)
    
    for i in range(n_samples):
        # difference = F_j - F_i
        diff = fitness_matrix - fitness_matrix[i]
        
        # Another point j is better or equal in ALL objectives
        better_or_equal = np.all(diff <= 0, axis=1)
        # Another point j is strictly better in AT LEAST ONE objective
        strictly_better = np.any(diff < 0, axis=1)
        
        if np.any(better_or_equal & strictly_better):
            is_dominated[i] = True
            
    return ~is_dominated


class Archive:
    """External storage for the Pareto optimal set."""
    def __init__(self):
        self.positions = np.empty((0, 0))
        self.fitness = np.empty((0, 0))

    def update(self, new_positions: np.ndarray, new_fitness: np.ndarray) -> None:
        if self.positions.size == 0:
            self.positions = np.copy(new_positions)
            self.fitness = np.copy(new_fitness)
        else:
            self.positions = np.vstack((self.positions, new_positions))
            self.fitness = np.vstack((self.fitness, new_fitness))
            
        # Filter down to only non-dominated solutions
        nd_mask = get_non_dominated_mask(self.fitness)
        self.positions = self.positions[nd_mask]
        self.fitness = self.fitness[nd_mask]


class MOSKROA:
    """Multi-Objective SKROA."""
    def __init__(
        self,
        evaluator: Callable[[np.ndarray], np.ndarray],
        bounds: tuple[float, float],
        dim: int,
        n_objectives: int = 2,
        n_agents: int = 100,
        max_iters: int = 200,
        epsilon_clamp: float = 0.05,
        gamma_lr: float = 0.05,
        levy_scale: float = 0.02,
        seed: int = 42
    ):
        self.evaluator = evaluator
        self.bounds = bounds
        self.dim = dim
        self.n_objectives = n_objectives
        self.n_agents = n_agents
        self.max_iters = max_iters
        
        self.epsilon_clamp = epsilon_clamp
        self.gamma_lr = gamma_lr
        self.levy_scale = levy_scale
        
        self.rng = np.random.default_rng(seed)
        self.archive = Archive()

    def _compute_scalarized_gradient(self, positions: np.ndarray, base_fitness: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """
        Computes gradients by randomly weighting the multiple objectives.
        This forces agents to exploit different parts of the Pareto front.
        """
        n1, d = positions.shape
        if n1 == 0:
            return np.empty((0, d))

        # Generate random weights for each agent, normalized to sum to 1
        weights = self.rng.random((n1, self.n_objectives))
        weights /= np.sum(weights, axis=1, keepdims=True)

        pos_expanded = np.tile(positions[:, np.newaxis, :], (1, d, 1))
        perturbation = np.eye(d) * h
        pos_perturbed = pos_expanded + perturbation[np.newaxis, :, :]

        flat_perturbed = pos_perturbed.reshape(n1 * d, d)
        flat_fitness = self.evaluator(flat_perturbed) # Shape (N1*D, M)

        f_perturbed = flat_fitness.reshape(n1, d, self.n_objectives)
        
        # Scalarize the perturbed fitness and base fitness
        scalar_base = np.sum(base_fitness * weights, axis=1) # (N1,)
        scalar_perturbed = np.sum(f_perturbed * weights[:, np.newaxis, :], axis=2) # (N1, D)

        grad = (scalar_perturbed - scalar_base[:, np.newaxis]) / h
        return grad

    def optimize(self) -> dict:
        low, high = self.bounds
        domain_span = high - low
        v_max = 0.1 * domain_span
        
        positions = self.rng.uniform(low, high, size=(self.n_agents, self.dim))
        
        start_time = time.perf_counter()

        for it in range(self.max_iters):
            current_fitness = self.evaluator(positions)
            
            # 1. Update Pareto Archive
            self.archive.update(positions, current_fitness)
            
            # 2. Biphasic State Determination
            # Non-dominated agents in current swarm enter State 1 (Exploitation)
            nd_mask = get_non_dominated_mask(current_fitness)
            states = np.where(nd_mask, 1, 0)
            
            new_positions = np.copy(positions)
            
            # 3. State 0: Rhizome Creep (Explore)
            s0_mask = (states == 0)
            if np.any(s0_mask):
                n0 = np.sum(s0_mask)
                step = levy_flight_step(shape=(n0, self.dim), scale=self.levy_scale * domain_span, rng=self.rng)
                new_positions[s0_mask] += step
                
            # 4. State 1: Vertical Shoot (Exploit Pareto Front)
            s1_mask = (states == 1)
            if np.any(s1_mask):
                x_s1 = positions[s1_mask]
                f_s1 = current_fitness[s1_mask]
                
                grad = self._compute_scalarized_gradient(x_s1, f_s1)
                
                raw_step = self.gamma_lr * domain_span * grad
                clipped_step = np.clip(raw_step, -v_max, v_max)
                new_positions[s1_mask] -= clipped_step

            # 5. Sympodial Clamping (Maintain Diversity in Decision Space)
            # For MOO, we pass a dummy scalar fitness to satisfy the operator's signature
            # (Repulse if crowded, treating all agents as equal rank to force spacing)
            dummy_fit = np.arange(self.n_agents) 
            new_positions = apply_sympodial_clamping(
                positions=new_positions,
                fitness=dummy_fit,
                epsilon_clamp=self.epsilon_clamp * domain_span,
                bounds=self.bounds
            )
            
            positions = np.clip(new_positions, low, high)

        exec_time = time.perf_counter() - start_time
        
        # Final update to ensure archive is perfect
        self.archive.update(positions, self.evaluator(positions))
        
        # Sort archive by Objective 1 for clean plotting
        sort_idx = np.argsort(self.archive.fitness[:, 0])
        
        return {
            "pareto_front_positions": self.archive.positions[sort_idx],
            "pareto_front_fitness": self.archive.fitness[sort_idx],
            "exec_time_ms": exec_time * 1000.0,
            "archive_size": len(self.archive.positions)
        }