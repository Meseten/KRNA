# SKROA — Sympodial Kawayan Rhizome Optimization Algorithm

**KRNA** (`krna` on PyPI) is a gradient-free, swarm-style framework for
continuous optimization, inspired by the growth habit of running bamboo
(kawayan): a biphasic strategy that alternates between wide *exploration*
(rhizome creep via Lévy flights) and focused *exploitation* (vertical shoot,
guided by finite-difference gradients), with two additional operators:

- **Sympodial Clamping** — agents that crowd together repel each other, which
  keeps the swarm spread out and delays premature convergence.
- **Culm-Abortion** — exploiting agents that stop improving for several
  consecutive steps are pruned and respawned near the current best, freeing
  their search budget.

The package also ships a multi-objective variant (**MO-SKROA**) that maps
Pareto fronts of the ZDT suite, a baseline PSO for comparison, benchmark
landscapes (Rastrigin, Ackley, Rosenbrock), a sensitivity-sweep harness, and
a hyperparameter tuner for scikit-learn estimators.

## Installation

```bash
pip install krna
```

Core features need only **NumPy**. Plotting (benchmark/sensitivity harnesses)
adds Matplotlib, and the ML tuner adds scikit-learn:

```bash
pip install "krna[plots]"   # + matplotlib
pip install "krna[ml]"      # + scikit-learn (implies plots)
```

## Quick start

```python
import numpy as np
from krna import SKROA
from krna.benchmarks import rastrigin

optimizer = SKROA(
    evaluator=rastrigin,       # maps (N, D) candidate matrix -> (N,) fitness
    bounds=(-5.12, 5.12),      # same box bounds on every dimension
    dim=10,
    n_agents=50,
    max_iters=500,
    seed=42,                   # fully reproducible runs
)
result = optimizer.optimize()

print("best fitness :", result["g_best_fit"])
print("best position:", result["g_best_pos"])
# result["convergence_curve"] is a per-iteration history of the best fitness
```

Your evaluator receives a `(n_agents, dim)` float array and must return a
`(n_agents,)` array of **fitness values to minimize**. Batch evaluation like
this keeps the whole swarm vectorized — don't loop over agents in Python if
your objective can be written with NumPy.

### Hyperparameters

| Parameter | Default | Role |
|---|---|---|
| `n_agents` | 50 | swarm size |
| `max_iters` | 1000 | iteration budget |
| `delta_threshold` | `None` | fitness cutoff for the exploit state; `None` uses the swarm median (keeps ~50% exploiting) |
| `tau_stagnation` | `1e-4` | minimum per-step improvement before an exploiting agent counts as stalled |
| `max_stagnation_steps` | 10 | stalled steps tolerated before Culm-Abortion respawns the agent |
| `epsilon_clamp` | `1e-2` | repulsion distance for Sympodial Clamping |
| `gamma_lr` | `0.05` | gradient-step learning rate (relative to the domain span) |
| `levy_scale` | `0.01` | Lévy-flight step scale (relative to the domain span) |
| `sigma_jitter` | `1e-4` | Gaussian jitter added to gradient steps |
| `seed` | 42 | RNG seed for reproducibility |

## Multi-objective optimization

```python
from krna import MOSKROA
from krna.mo_benchmarks import ZDT1_BENCHMARK

opt = MOSKROA(evaluator=ZDT1_BENCHMARK.evaluator, bounds=(0.0, 1.0), dim=30)
res = opt.optimize()  # res["pareto_front_fitness"] -> (A, 2) objective matrix
```

MO-SKROA maintains a non-dominated archive; exploiting agents descend along
randomly scalarized gradients so different agents cover different regions of
the front.

## Hyperparameter tuning for scikit-learn models

```python
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from krna.ml_tuning import SKROAMLTuner

data = load_breast_cancer()
space = {
    "C":     {"type": "log_float", "min": 1e-3, "max": 1e3},
    "gamma": {"type": "log_float", "min": 1e-4, "max": 1e1},
    "kernel": {"type": "categorical", "values": ["linear", "rbf", "sigmoid"]},
}
tuner = SKROAMLTuner(model_class=SVC, param_space=space,
                     X=data.data, y=data.target, seed=101)
print(tuner.tune())
```

Supported parameter types: `int`, `float`, `log_float`, `categorical`.

## Command line

```bash
krna benchmark      # SKROA vs PSO on Rastrigin/Ackley/Rosenbrock, CSV + plots
krna tune           # SKROA hyperparameter search on an SVM (breast cancer)
krna mo-benchmark   # MO-SKROA on ZDT1/ZDT2 Pareto fronts
```

Outputs land in `results/logs/` (CSV telemetry) and `results/plots/`
(convergence curves, 3D surfaces, Pareto fronts, sensitivity heatmaps).

## Statistical significance testing

Comparing two metaheuristics by eyeballing mean fitness is not publishable —
differences can be noise. The benchmark suite therefore reports, for every
landscape, a **two-sided Wilcoxon rank-sum test** (Mann-Whitney U) between the
SKROA and PSO final best-fitness samples, with tie correction and continuity
correction, plus **Cohen's r** effect size. Both are implemented in
`krna.stats` with NumPy only and are cross-checked against
`scipy.stats.mannwhitneyu` in the test suite.

Each `benchmark_metrics.csv` row for PSO carries `P_Value_RankSum`,
`Effect_Size_r`, and a plain-language `Statistical_Verdict`; the console
prints a `[STATS]` line per landscape. Use at least 30 independent runs
(`--trials 30`) for stable estimates.

You can also test your own samples:

```python
from krna import wilcoxon_rank_sum, cohens_r, bonferroni_correct

result = wilcoxon_rank_sum(fitness_a, fitness_b)  # >= 8 runs per sample
print(result.u_statistic, result.p_value, result.is_significant)
print("effect size r =", cohens_r(fitness_a, fitness_b))
adjusted = bonferroni_correct([result.p_value, 0.02])  # family-wise control
```

For multi-problem studies, adjust for multiple comparisons (Bonferroni is
provided; Holm or Friedman + Imany–Davenport are common stronger choices).

## Methodology and reproducibility

- **Minimization everywhere.** Evaluators map an `(n_agents, dim)` array of
  candidate positions to an `(n_agents,)` array of fitness values; lower is
  better.
- **Seeded runs.** All stochastic operators draw from a NumPy `Generator`
  seeded at construction. Two runs with the same seed produce bit-identical
  convergence curves (verified by test).
- **Bounds are a hard contract.** No point is ever handed to your evaluator
  outside `[low, high]` — gradient probes fold inward at boundaries
  (`krna.gradient`).
- **Exploration/exploitation split.** Agents below the fitness threshold
  (default: the swarm median, `delta_threshold=None`) exploit via
  finite-difference gradient descent; the rest explore via Lévy flights.
  Stalled exploiting agents are respawned near the best solution after
  `max_stagnation_steps` non-improving iterations (Culm-Abortion).
- **Fair benchmarking.** SKROA and PSO receive identical iteration budgets,
  swarm sizes, seeds, and evaluation functions per trial.

## Running the tests

```bash
python -m unittest discover -s tests -v
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, ground rules,
and pull-request checks. Notable changes are recorded in
[CHANGELOG.md](CHANGELOG.md).

## License

MIT — see [LICENSE](LICENSE).
