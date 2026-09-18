"""
KRNA - Kawayan Rhizome Network Algorithm (SKROA)
================================================
A biphasic continuous optimization framework: Rhizome Creep (Lévy-flight
exploration) alternating with Vertical Shoot (gradient-guided exploitation),
with Sympodial Clamping anti-collision and Culm-Abortion stagnation pruning.

Quick start (single-objective)::

    import numpy as np
    from krna import SKROA
    from krna.benchmarks import rastrigin

    optimizer = SKROA(
        evaluator=lambda X: rastrigin(X),  # (N, D) positions -> (N,) fitness
        bounds=(-5.12, 5.12),
        dim=10,
        n_agents=50,
        max_iters=500,
    )
    result = optimizer.optimize()
    print(result["g_best_fit"], result["g_best_pos"])

Optional extras (require scikit-learn)::

    from krna.ml_tuning import SKROAMLTuner, HyperparameterMapper  # pip install krna[ml]
    from krna.mo_skroa import MOSKROA                              # multi-objective
"""

from krna.skroa import SKROA
from krna.operators import levy_flight_step, apply_sympodial_clamping, apply_culm_abortion
from krna.baselines import PSO
from krna.mo_skroa import MOSKROA, Archive, get_non_dominated_mask

__version__ = "1.0.0"

__all__ = [
    "SKROA",
    "PSO",
    "MOSKROA",
    "Archive",
    "get_non_dominated_mask",
    "levy_flight_step",
    "apply_sympodial_clamping",
    "apply_culm_abortion",
    "__version__",
]
