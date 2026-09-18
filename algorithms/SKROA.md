# SKROA — Algorithm Explainer

**SKROA** = **S**ympodial **K**awayan **R**hizome **O**ptimization **A**lgorithm.
It lives inside the **KRNA** package (**K**awayan **R**hizome **N**etwork
**A**lgorithm), where *kawayan* is Tagalog for **bamboo**. Source of record:
[`krna/skroa.py`](../krna/skroa.py), operators in
[`krna/operators.py`](../krna/operators.py).

> What it is: a gradient-free swarm optimizer for minimization of black-box
> functions `f: R^D -> R`. What it is not: a supervised learning model — it
> never trains on data; it *searches*.

## The bamboo metaphor (and what it actually maps to)

Running bamboo spreads through underground **rhizomes**: it creeps widely
(exploration), then sends up vertical **culms** (exploitation). A culm that
fails gets aborted and its resources go elsewhere (budget reallocation). Dense
groves thin themselves out (diversity preservation).

| Metaphor | Mechanism | Literature relative |
|---|---|---|
| Rhizome Creep | Lévy flights (Mantegna's algorithm, β = 1.5) | Cuckoo Search, Lévy-PSO |
| Vertical Shoot | Vectorized finite-difference gradient descent | Memetic / local-search hybrids |
| Sympodial Clamping | Pairwise repulsion of agents closer than `epsilon_clamp` | PSO repulsion variants |
| Culm-Abortion | Prune exploiting agents stalled ≥ `max_stagnation_steps`, respawn near best | Stagnation-restart schemes |

Honest framing: SKROA is a **hybrid swarm/memetic algorithm** with a biphasic
(median-split) exploration/exploitation policy. The metaphor names the
mechanisms; the mechanisms are what matter, and every one of them can be
switched off and measured (see [Ablation](#ablation)).

## The loop

```text
initialize N agents uniformly in [low, high]
evaluate fitness of all agents
for t in 1..max_iters:
    # 1. Biphasic split (median threshold by default)
    exploit_mask  = fitness <= median(fitness)     # state 1: Vertical Shoot
    explore_mask  = ~exploit_mask                  # state 0: Rhizome Creep

    # 2. Culm-Abortion (budget reallocation)
    for each exploiting agent with improvement < tau_stagnation:
        stagnation_counter += 1
        if counter >= max_stagnation_steps:
            respawn agent ~ N(g_best, (0.3 * span)^2); reset to state 0

    # 3. Move
    explorers  += levy_flight_step(scale = levy_scale * span)
    exploiters -= clip(gamma_lr * span * grad_f(x), ±0.1 * span) + jitter

    # 4. Sympodial Clamping (anti-collision repulsion)
    for each pair (i, j) with ||x_i - x_j|| < epsilon_clamp and f_i > f_j:
        x_i += eta * (x_i - x_j) / (||x_i - x_j||^2 + delta)

    # 5. Enforce bounds (hard contract), record best
```

Key invariants (enforced in code, verified by tests):

- **Hard bounds contract.** No candidate is ever evaluated outside
  `[low, high]`; gradient probes fold inward at the boundary
  ([`krna/gradient.py`](../krna/gradient.py)).
- **Reproducibility.** Every stochastic draw comes from one seeded
  `np.random.Generator`; same seed ⇒ bit-identical convergence curve.
- **Vectorization.** Gradients, Lévy steps, and repulsion are computed for the
  whole swarm as array ops — no per-agent Python loops.

## Hyperparameters

| Parameter | Default | Role |
|---|---|---|
| `n_agents` | 50 | swarm size |
| `max_iters` | 1000 | iteration budget |
| `delta_threshold` | `None` | exploit cutoff; `None` = dynamic swarm median |
| `tau_stagnation` | `1e-4` | min improvement counted as progress |
| `max_stagnation_steps` | 10 | stalled steps tolerated before Culm-Abortion |
| `epsilon_clamp` | `1e-2` | repulsion trigger distance |
| `gamma_lr` | 0.05 | gradient step size (× domain span) |
| `levy_scale` | 0.01 | Lévy step scale (× domain span) |
| `sigma_jitter` | 1e-4 | Gaussian jitter on gradient steps |
| `seed` | 42 | RNG seed |

## Ablation

Each operator can be disabled at construction to measure its contribution:

```python
SKROA(..., use_biphasic=False)       # whole swarm explores via Lévy flights
SKROA(..., use_clamping=False)       # no anti-collision repulsion
SKROA(..., use_culm_abortion=False)  # stalled exploiters are never pruned
```

Run the full study — all variants × all landscapes × N seeds, Wilcoxon-tested
against full SKROA — with:

```bash
krna ablation --trials 30
```

Results land in `results/logs/ablation_metrics.csv` and
`results/plots/ablation_convergence.png`. All variants share per-trial seeds,
so comparisons are paired at the seed level.

## Multi-objective and ML tuning

- **MO-SKROA** ([`krna/mo_skroa.py`](../krna/mo_skroa.py)) maintains a
  non-dominated archive; exploiting agents descend along randomly scalarized
  gradients so different agents cover different regions of the Pareto front.
- **SKROAMLTuner** ([`krna/ml_tuning.py`](../krna/ml_tuning.py)) maps SKROA's
  continuous `[0, 1]^D` coordinates onto int / float / log_float / categorical
  hyperparameters of any scikit-learn estimator and minimizes `1 - CV accuracy`.

## Positioning (read this before citing SKROA)

Closest relatives: **PSO** (same swarm skeleton — shipped as
[`krna/baselines.PSO`](../krna/baselines.py)), Differential Evolution,
CMA-ES, and the broader memetic-algorithm family. Bamboo-inspired
metaphor neighbors exist in the literature (e.g., BFGO, Bamboo Forest Growth
Optimization).

Our own published benchmark (`krna benchmark --trials 30`) shows **baseline
PSO beating SKROA on Rastrigin, Ackley, and Rosenbrock at D=10 with equal
iteration budgets** — we report this rather than hide it. Per iteration SKROA
also pays extra finite-difference evaluations that PSO does not; the benchmark
CSV records evaluation counts so budget-matched readings are possible.

Before claiming SKROA as a contribution in academic work, run:

1. `krna benchmark` against the full opponent panel (PSO, SciPy DE, CMA-ES);
2. `krna ablation` to isolate which operators earn their keep;
3. problem classes where exploration-heavy budgets plausibly pay off
   (rugged, noisy, or expensive objectives).

If it does not win anywhere, the honest positioning is a well-engineered,
reproducible, statistically transparent optimization framework — which is the
reason this package exists.
