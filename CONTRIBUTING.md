# Contributing to KRNA

Thanks for your interest in improving SKROA!

## Development setup

```bash
git clone https://github.com/Meseten/KRNA
cd KRNA
python3 -m venv venv && source venv/bin/activate
pip install -e ".[ml]"        # core + plotting + ML extras
pip install build twine       # release tooling
```

## Running the tests

```bash
python -m unittest discover -s tests -v
```

The suite runs in about a second and needs no network access. Tests that
cross-check `krna.stats` against scipy are skipped automatically when scipy
is not installed.

## Ground rules

- **Every behavior change ships with a test.** The statistical operators and
  engine invariants especially: if it computes a number, it needs a test with
  a known-answer or property check.
- **NumPy only in the core.** `krna.skroa`, `krna.operators`, `krna.gradient`,
  `krna.stats`, `krna.mo_skroa`, and `krna.baselines` must import nothing
  heavier than NumPy. Matplotlib / scikit-learn imports belong inside
  functions in `krna.benchmarks`, `krna.sensitivity`, `krna.cli`, and
  `krna.ml_tuning` only.
- **Keep evaluators pure.** Optimizers must never mutate the arrays they pass
  to user evaluators, and all probe points must stay inside the declared
  bounds (use `krna.gradient.bounds_aware_gradient`).
- **Reproducibility is a feature.** All stochastic behavior flows from the
  seeded `np.random.Generator`; never use module-level randomness.
- **Style**: PEP 8, PEP 257 docstrings, type hints on public functions.

## Before opening a pull request

1. `python -m unittest discover -s tests` passes.
2. New/changed public APIs are documented in `README.md` and `CHANGELOG.md`.
3. If you touched packaging, run `python -m build && twine check dist/*`.

## Reporting bugs

Open a GitHub issue with: your Python version, a minimal reproducing script,
the full traceback, and what you expected to happen instead.
