# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-09-18

### Added

- PyPI publishing via GitHub Actions Trusted Publishing (`.github/workflows/publish.yml`): builds the sdist and wheel and uploads them on version tags (`v*`) using OIDC — no API tokens.
- CI workflow (`.github/workflows/tests.yml`) running the 50-test unittest suite on Python 3.11–3.13 for every push and pull request to `main`.
- README: random forest hyperparameter tuning example (`SKROAMLTuner` with `RandomForestClassifier`).

## [1.0.0] - 2026-09-18

### Added

- Biphasic continuous optimizer `SKROA` (Rhizome Creep exploration via Lévy
  flights, Vertical Shoot exploitation via finite-difference gradients) with
  Sympodial Clamping anti-collision and Culm-Abortion stagnation pruning.
- Multi-objective `MOSKROA` with a non-dominated archive, mapped on the ZDT1/ZDT2
  benchmarks; baseline `PSO` for comparison; Rastrigin/Ackley/Rosenbrock
  landscapes; a parameter sensitivity sweep harness; and an scikit-learn
  hyperparameter tuner (`SKROAMLTuner`).
- Statistical significance testing (`krna.stats`): two-sided Wilcoxon rank-sum
  test with tie correction and continuity correction (cross-checked against
  `scipy.stats.mannwhitneyu`), Cohen's r effect size, and Bonferroni
  correction — all NumPy-only. Benchmark telemetry CSV reports p-values,
  effect sizes, and per-pair verdicts.
- Shared bounds-aware gradient kernel (`krna.gradient`) used by both engines.
- Reproducibility via seeded `np.random.Generator` throughout.
- Public API exports from the `krna` package root; MIT license; PEP 621/639
  packaging with optional extras (`krna[plots]`, `krna[ml]`).
- 50 unit tests covering operators, engines, benchmarks, statistics, and the
  ML tuner.

### Fixed

- Culm-Abortion now counts fitness *regressions* as stagnation (previously an
  agent whose fitness worsened was treated as progressing and never pruned).
- Finite-difference gradient probes stay inside the declared bounds in both
  SKROA and MO-SKROA (shared `krna.gradient` kernel); objectives undefined
  outside the box no longer break exploitation.
- ML tuner no longer crashes with `TypeError` on estimators without a
  `random_state` parameter.
- Sensitivity heatmap no longer crashes on NaN / constant fitness matrices.
- Convergence plot grid sizes to the benchmark list and reports the true
  dimensionality instead of a hardcoded "D=10".
- ZDT benchmarks raise `ValueError` for dimensionality below 2 instead of
  silently dividing by zero.
- Constructors reject invalid configurations (dim/n_agents/max_iters below 1,
  inverted or non-finite bounds); evaluators returning wrong shapes fail
  loudly with an explanatory error.

### Changed

- Core package depends only on NumPy; Matplotlib and scikit-learn are optional
  extras. SciPy was removed as a runtime dependency.
- Repository hygiene: venv, results, build artifacts, and `__pycache__` are no
  longer tracked in git.
