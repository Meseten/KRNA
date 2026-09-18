"""
KRNA / Multi-Objective Benchmarks
==============================================================================
Provides vectorized NumPy implementations of the canonical ZDT test suite for
Multi-Objective Optimization (ZDT1, ZDT2).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np

@dataclass(frozen=True)
class MOBenchmarkFunction:
    """
    Specification for a Multi-Objective test landscape.
    """
    name: str
    evaluator: Callable[[np.ndarray], np.ndarray]  # Maps (N, D) -> (N, M)
    bounds: tuple[float, float]
    n_objectives: int

def _ensure_2d(x: np.ndarray) -> np.ndarray:
    return x[np.newaxis, :] if x.ndim == 1 else x

def zdt1(x: np.ndarray) -> np.ndarray:
    """
    ZDT1 Function: 2 Objectives, Convex Pareto Front.
    x_i in [0, 1] for all i.
    """
    arr = _ensure_2d(np.asarray(x, dtype=np.float64))
    n_samples, dim = arr.shape
    if dim < 2:
        raise ValueError("ZDT functions require dimensionality D >= 2.")
    
    # Objective 1: f1(x) = x_1
    f1 = arr[:, 0]
    
    # g(x) = 1 + 9 * sum(x_2 ... x_D) / (D - 1)
    g = 1.0 + 9.0 * np.sum(arr[:, 1:], axis=1) / (dim - 1.0)
    
    # Objective 2: f2(x) = g(x) * [1 - sqrt(f1(x) / g(x))]
    f2 = g * (1.0 - np.sqrt(f1 / g))
    
    return np.column_stack((f1, f2))

def zdt2(x: np.ndarray) -> np.ndarray:
    """
    ZDT2 Function: 2 Objectives, Non-Convex Pareto Front.
    x_i in [0, 1] for all i.
    """
    arr = _ensure_2d(np.asarray(x, dtype=np.float64))
    n_samples, dim = arr.shape
    if dim < 2:
        raise ValueError("ZDT functions require dimensionality D >= 2.")
    
    f1 = arr[:, 0]
    g = 1.0 + 9.0 * np.sum(arr[:, 1:], axis=1) / (dim - 1.0)
    
    # Objective 2 uses a squared term to create a concave/non-convex front
    f2 = g * (1.0 - (f1 / g)**2)
    
    return np.column_stack((f1, f2))

ZDT1_BENCHMARK = MOBenchmarkFunction(name="ZDT1", evaluator=zdt1, bounds=(0.0, 1.0), n_objectives=2)
ZDT2_BENCHMARK = MOBenchmarkFunction(name="ZDT2", evaluator=zdt2, bounds=(0.0, 1.0), n_objectives=2)