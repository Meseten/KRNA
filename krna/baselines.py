"""
KRNA / Benchmarking Baselines
==============================================================================
Implements a highly vectorized, standard Particle Swarm Optimization (PSO)
algorithm for head-to-head performance comparison against SKROA.
"""

from __future__ import annotations
import time
from typing import Callable
import numpy as np


class PSO:
    """
    Standard Particle Swarm Optimization (PSO) with linearly decaying inertia weight.
    """

    def __init__(
        self,
        evaluator: Callable[[np.ndarray], np.ndarray],
        bounds: tuple[float, float],
        dim: int,
        n_agents: int = 50,
        max_iters: int = 1000,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 2.0,
        c2: float = 2.0,
        seed: int = 42
    ):
        self.evaluator = evaluator
        self.bounds = bounds
        self.dim = dim
        self.n_agents = n_agents
        self.max_iters = max_iters
        
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        
        self.rng = np.random.default_rng(seed)

    def optimize(self) -> dict:
        """
        Executes the PSO optimization loop.
        
        Returns:
            Dictionary containing telemetry, best coordinates, and elapsed time.
        """
        low, high = self.bounds
        v_max = 0.1 * (high - low)  # Standard velocity clamping
        
        # Initialize positions and velocities
        positions = self.rng.uniform(low, high, size=(self.n_agents, self.dim))
        velocities = self.rng.uniform(-v_max, v_max, size=(self.n_agents, self.dim))
        
        # Personal best tracking
        p_best_pos = np.copy(positions)
        p_best_fit = self.evaluator(positions)
        
        # Global best tracking
        best_idx = np.argmin(p_best_fit)
        g_best_pos = np.copy(p_best_pos[best_idx])
        g_best_fit = p_best_fit[best_idx]
        
        convergence_curve = np.zeros(self.max_iters)
        start_time = time.perf_counter()

        for it in range(self.max_iters):
            # Evaluate fitness of current positions
            current_fitness = self.evaluator(positions)
            
            # Update personal bests
            improve_mask = current_fitness < p_best_fit
            p_best_fit[improve_mask] = current_fitness[improve_mask]
            p_best_pos[improve_mask] = positions[improve_mask]
            
            # Update global best
            min_idx = np.argmin(p_best_fit)
            if p_best_fit[min_idx] < g_best_fit:
                g_best_fit = p_best_fit[min_idx]
                g_best_pos = np.copy(p_best_pos[min_idx])
                
            convergence_curve[it] = g_best_fit
            
            # Calculate dynamic inertia weight
            w = self.w_max - (self.w_max - self.w_min) * (it / self.max_iters)
            
            # Generate stochastic cognitive and social components
            r1 = self.rng.random((self.n_agents, self.dim))
            r2 = self.rng.random((self.n_agents, self.dim))
            
            cognitive = self.c1 * r1 * (p_best_pos - positions)
            social = self.c2 * r2 * (g_best_pos - positions)
            
            # Update velocities and clamp
            velocities = (w * velocities) + cognitive + social
            velocities = np.clip(velocities, -v_max, v_max)
            
            # Update positions and enforce boundaries
            positions = positions + velocities
            positions = np.clip(positions, low, high)

        exec_time = time.perf_counter() - start_time
        
        return {
            "g_best_pos": g_best_pos,
            "g_best_fit": g_best_fit,
            "convergence_curve": convergence_curve,
            "exec_time_ms": exec_time * 1000.0,
            "total_aborts": 0  # To match SKROA's dictionary signature
        }