"""
KRNA / SKROA Benchmarking Objective Landscapes & Automated Telemetry Suite
==============================================================================
Provides vectorized NumPy implementations of canonical optimization test
functions (Rastrigin, Ackley, Rosenbrock) and an automated benchmarking engine
comparing SKROA against baseline PSO, SciPy's Differential Evolution, and
CMA-ES (when the `cma` package is installed).

Also provides an ablation-study harness (`execute_ablation_suite`) that reruns
SKROA with each operator disabled — biphasic split, Sympodial Clamping,
Culm-Abortion — to measure their individual contribution.

Logs structured metrics to results/logs/benchmark_metrics.csv (or
ablation_metrics.csv) and exports convergence curves and 3D surface topology
plots to results/plots/.
"""

from __future__ import annotations
import csv
import math
import os
import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable
import numpy as np

from krna.skroa import SKROA
from krna.baselines import PSO
from krna.stats import wilcoxon_rank_sum, cohens_r

try:  # Optional opponents — degrade gracefully when absent
    from scipy.optimize import differential_evolution
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - scipy is a runtime extra
    differential_evolution = None
    _HAS_SCIPY = False

try:
    import cma
    _HAS_CMA = True
except ImportError:  # pragma: no cover
    cma = None
    _HAS_CMA = False


@dataclass(frozen=True)
class BenchmarkFunction:
    """
    Immutable specification for a continuous test landscape.

    Attributes:
        name: Clean identifier for telemetry and logging.
        evaluator: Vectorized callable mapping (N, D) -> (N,).
        bounds: Tuple of (lower_bound, upper_bound) valid across all dimensions.
        global_optimum_coords: Function generating known optimum coords for dim D.
        global_optimum_fitness: Known minimum objective value.
    """
    name: str
    evaluator: Callable[[np.ndarray], np.ndarray]
    bounds: tuple[float, float]
    global_optimum_coords: Callable[[int], np.ndarray]
    global_optimum_fitness: float


def _ensure_2d(x: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Ensures input array is 2D (N, D). Returns formatted array and a boolean
    flag indicating whether the original input was 1D.
    """
    if x.ndim == 1:
        return x[np.newaxis, :], True
    elif x.ndim == 2:
        return x, False
    else:
        raise ValueError(f"Input array must be 1D or 2D, got shape {x.shape}")


def rastrigin(x: np.ndarray) -> np.ndarray:
    """
    Vectorized Rastrigin objective function.
    Formula: f(x) = 10 * D + sum(x_i^2 - 10 * cos(2 * pi * x_i))
    """
    arr, was_1d = _ensure_2d(np.asarray(x, dtype=np.float64))
    dim = arr.shape[1]
    omega = 2.0 * np.pi
    fitness = 10.0 * dim + np.sum(arr**2 - 10.0 * np.cos(omega * arr), axis=1)
    return fitness[0] if was_1d else fitness


def ackley(x: np.ndarray) -> np.ndarray:
    """
    Vectorized Ackley objective function.
    Formula: f(x) = -20*exp(-0.2*sqrt(mean(x_i^2))) - exp(mean(cos(2*pi*x_i))) + 20 + e
    """
    arr, was_1d = _ensure_2d(np.asarray(x, dtype=np.float64))
    dim = arr.shape[1]
    sum_sq = np.sum(arr**2, axis=1)
    sum_cos = np.sum(np.cos(2.0 * np.pi * arr), axis=1)

    term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / dim))
    term2 = -np.exp(sum_cos / dim)
    fitness = term1 + term2 + 20.0 + np.e
    return fitness[0] if was_1d else fitness


def rosenbrock(x: np.ndarray) -> np.ndarray:
    """
    Vectorized Rosenbrock objective function.
    Formula: f(x) = sum_{i=1}^{D-1} [100 * (x_{i+1} - x_i^2)^2 + (x_i - 1)^2]
    """
    arr, was_1d = _ensure_2d(np.asarray(x, dtype=np.float64))
    if arr.shape[1] < 2:
        raise ValueError("Rosenbrock function requires dimensionality D >= 2.")

    x_i = arr[:, :-1]
    x_next = arr[:, 1:]
    fitness = np.sum(100.0 * (x_next - x_i**2)**2 + (x_i - 1.0)**2, axis=1)
    return fitness[0] if was_1d else fitness


# ==============================================================================
# BENCHMARK REGISTRY
# ==============================================================================

RASTRIGIN_BENCHMARK = BenchmarkFunction(
    name="Rastrigin",
    evaluator=rastrigin,
    bounds=(-5.12, 5.12),
    global_optimum_coords=lambda dim: np.zeros(dim, dtype=np.float64),
    global_optimum_fitness=0.0
)

ACKLEY_BENCHMARK = BenchmarkFunction(
    name="Ackley",
    evaluator=ackley,
    bounds=(-32.768, 32.768),
    global_optimum_coords=lambda dim: np.zeros(dim, dtype=np.float64),
    global_optimum_fitness=0.0
)

ROSENBROCK_BENCHMARK = BenchmarkFunction(
    name="Rosenbrock",
    evaluator=rosenbrock,
    bounds=(-5.0, 10.0),
    global_optimum_coords=lambda dim: np.ones(dim, dtype=np.float64),
    global_optimum_fitness=0.0
)

_BENCHMARK_CATALOG: dict[str, BenchmarkFunction] = {
    "rastrigin": RASTRIGIN_BENCHMARK,
    "ackley": ACKLEY_BENCHMARK,
    "rosenbrock": ROSENBROCK_BENCHMARK,
}


def get_benchmark(name: str) -> BenchmarkFunction:
    """
    Retrieves a BenchmarkFunction specification by its canonical string name.
    """
    key = name.lower().strip()
    if key not in _BENCHMARK_CATALOG:
        valid_keys = ", ".join(sorted(_BENCHMARK_CATALOG.keys()))
        raise KeyError(f"Unknown benchmark '{name}'. Available: {valid_keys}")
    return _BENCHMARK_CATALOG[key]


# ==============================================================================
# OPTIMIZER DISPATCH (SKROA, PSO, SciPy DE, CMA-ES)
# ==============================================================================

def _make_eval_counting_evaluator(
    evaluator: Callable[[np.ndarray], np.ndarray],
    counter: dict,
) -> Callable[[np.ndarray], np.ndarray]:
    """Wraps an evaluator so every call batch increments a shared counter."""
    def counted(X: np.ndarray) -> np.ndarray:
        counter["evals"] += np.atleast_2d(X).shape[0]
        return evaluator(X)
    return counted


def get_available_optimizers() -> list[str]:
    """Names of opponents usable on this installation."""
    names = ["SKROA", "PSO"]
    if _HAS_SCIPY:
        names.append("SciPy-DE")
    if _HAS_CMA:
        names.append("CMA-ES")
    return names


def run_one(
    algo_name: str,
    evaluator: Callable[[np.ndarray], np.ndarray],
    bounds: tuple[float, float],
    dim: int,
    n_agents: int,
    max_iters: int,
    seed: int,
    skroa_kwargs: dict | None = None,
) -> dict:
    """
    Runs one optimization trial with the named algorithm and normalizes the
    result to the common telemetry signature:
        {g_best_pos, g_best_fit, convergence_curve, exec_time_ms, total_aborts}
    """
    counter = {"evals": 0}
    counted_eval = _make_eval_counting_evaluator(evaluator, counter)

    start_time = time.perf_counter()
    if algo_name == "SKROA":
        optimizer = SKROA(
            evaluator=counted_eval,
            bounds=bounds,
            dim=dim,
            n_agents=n_agents,
            max_iters=max_iters,
            seed=seed,
            **(skroa_kwargs or {}),
        )
        res = optimizer.optimize()
    elif algo_name == "PSO":
        optimizer = PSO(
            evaluator=counted_eval,
            bounds=bounds,
            dim=dim,
            n_agents=n_agents,
            max_iters=max_iters,
            seed=seed,
        )
        res = optimizer.optimize()
    elif algo_name == "SciPy-DE":
        if not _HAS_SCIPY:
            raise RuntimeError("SciPy is not installed; SciPy-DE unavailable")
        # scipy's popsize is a per-dimension multiplier: total population is
        # popsize * dim. Divide the swarm size across dimensions so DE fields
        # the same number of candidate solutions as the swarm algorithms.
        de_bounds = [(bounds[0], bounds[1])] * dim
        de_result = differential_evolution(
            lambda z: float(evaluator(np.asarray(z, dtype=np.float64)[np.newaxis, :])[0]),
            de_bounds,
            maxiter=max_iters,
            popsize=max(1, n_agents // dim),
            seed=seed,
            polish=False,
        )
        return {
            "g_best_pos": np.asarray(de_result.x, dtype=np.float64),
            "g_best_fit": float(de_result.fun),
            "convergence_curve": np.asarray([de_result.fun]),
            "exec_time_ms": (time.perf_counter() - start_time) * 1000.0,
            "total_aborts": 0,
            "total_evals": counter["evals"],
        }
    elif algo_name == "CMA-ES":
        if not _HAS_CMA:
            raise RuntimeError("The 'cma' package is not installed; CMA-ES unavailable")
        sigma0 = 0.25 * (bounds[1] - bounds[0])
        es = cma.CMAEvolutionStrategy(
            np.full(dim, (bounds[0] + bounds[1]) / 2.0),
            sigma0,
            {
                "bounds": [bounds[0], bounds[1]],
                "popsize": n_agents,
                "seed": seed + 1,  # cma rejects seed 0; shift to stay reproducible
                "verbose": -9,
            },
        )
        curve: list[float] = []
        solutions = es.ask()
        solutions_fit = evaluator(np.asarray(solutions, dtype=np.float64))
        counter["evals"] += len(solutions)
        es.tell(solutions, list(solutions_fit))
        curve.append(float(np.min(solutions_fit)))
        for _ in range(max_iters - 1):
            solutions = es.ask()
            solutions_fit = evaluator(np.asarray(solutions, dtype=np.float64))
            counter["evals"] += len(solutions)
            es.tell(solutions, list(solutions_fit))
            curve.append(float(np.min(solutions_fit)))
            if es.stop():
                break
        best_idx = int(np.argmin(solutions_fit))
        res = {
            "g_best_pos": np.asarray(solutions[best_idx], dtype=np.float64),
            "g_best_fit": float(np.min(solutions_fit)),
            "convergence_curve": np.asarray(curve),
            "exec_time_ms": (time.perf_counter() - start_time) * 1000.0,
            "total_aborts": 0,
        }
    else:
        raise KeyError(f"Unknown algorithm '{algo_name}'. Available: {get_available_optimizers()}")

    res["exec_time_ms"] = (time.perf_counter() - start_time) * 1000.0
    res["total_evals"] = counter["evals"]
    return res


# ==============================================================================
# TELEMETRY, MEMORY PROFILING & VISUALIZATION ENGINE
# ==============================================================================

_ALGO_COLORS = {
    "SKROA": "#1f77b4",
    "PSO": "#d62728",
    "SciPy-DE": "#2ca02c",
    "CMA-ES": "#ff7f0e",
}
_FALLBACK_COLORS = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def _algo_color(algo_name: str, index: int) -> str:
    if algo_name in _ALGO_COLORS:
        return _ALGO_COLORS[algo_name]
    return _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)]


def generate_3d_surface_plot(benchmark: BenchmarkFunction, output_dir: str) -> None:
    """
    Generates and saves a high-resolution 3D surface plot of a 2D slice
    of the objective function landscape to illustrate its topological challenges.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (Registers 3D projection)
    low, high = benchmark.bounds
    # Prevent excessive range visual clipping on Ackley
    plot_low = max(low, -10.0) if benchmark.name == "Ackley" else low
    plot_high = min(high, 10.0) if benchmark.name == "Ackley" else high

    grid_size = 80
    x = np.linspace(plot_low, plot_high, grid_size)
    y = np.linspace(plot_low, plot_high, grid_size)
    xx, yy = np.meshgrid(x, y)

    coords = np.column_stack([xx.ravel(), yy.ravel()])
    zz = benchmark.evaluator(coords).reshape(grid_size, grid_size)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        xx, yy, zz,
        cmap="viridis",
        edgecolor="none",
        alpha=0.88,
        antialiased=True
    )
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1, label="Objective Value $f(x)$")

    opt_coords = benchmark.global_optimum_coords(2)
    ax.scatter(
        opt_coords[0], opt_coords[1], benchmark.global_optimum_fitness,
        color="red", s=70, label="Global Minimum $\\mathbf{g}^*$", depthshade=False
    )

    ax.set_title(f"{benchmark.name} Function (D=2 Topology)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("$x_1$", labelpad=10)
    ax.set_ylabel("$x_2$", labelpad=10)
    ax.set_zlabel("$f(\\mathbf{x})$", labelpad=10)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"surface_3d_{benchmark.name.lower()}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_convergence_grid(
    benchmarks: list[BenchmarkFunction],
    curves_store: dict[str, dict[str, np.ndarray]],
    output_dir: str,
    dim: int = 10,
    title: str = "KRNA Algorithm Optimization: Convergence Trajectories",
    filename_stem: str = "convergence_comparison",
) -> None:
    """
    Exports a publication-grade subplot grid comparing the mean convergence
    trajectories of all algorithms across all tested landscapes on a log scale.

    curves_store maps benchmark.name -> {algorithm (or variant) name -> mean
    curve}, as collected by the suite drivers.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_panels = max(1, len(benchmarks))
    fig, axes = plt.subplots(1, n_panels, figsize=(6.0 * n_panels, 5.5), squeeze=False)
    axes = axes[0]
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

    for idx, bench in enumerate(benchmarks):
        ax = axes[idx]
        for algo_idx, (algo_name, curve) in enumerate(curves_store.get(bench.name, {}).items()):
            curve = np.asarray(curve, dtype=np.float64)
            finite = curve[np.isfinite(curve)]
            if finite.size == 0:
                continue
            iters = np.arange(1, finite.size + 1)
            ax.plot(
                iters, finite,
                label=algo_name,
                color=_algo_color(algo_name, algo_idx),
                linewidth=2.2 if algo_name == "SKROA" or algo_name == "Full SKROA" else 2.0,
                linestyle="-" if algo_name == "SKROA" or algo_name == "Full SKROA" else "--",
            )

        ax.set_title(f"{bench.name} Landscape ($D={dim}$)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Iteration ($t$)", fontsize=11)
        ax.set_ylabel("Best Objective Value $\\bar{f}(\\mathbf{g}^*)$", fontsize=11)
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", frameon=True)

    plt.tight_layout(rect=[0, 0.0, 1, 0.94])
    png_path = os.path.join(output_dir, f"{filename_stem}.png")
    pdf_path = os.path.join(output_dir, f"{filename_stem}.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize_comparison(
    focal_fits: np.ndarray,
    reference_fits: np.ndarray,
    focal_name: str = "SKROA",
    reference_name: str = "PSO",
) -> dict:
    """
    Head-to-head statistical comparison of two final best-fitness samples.

    Runs a two-sided Wilcoxon rank-sum test (Mann-Whitney U) — the standard
    tool in the metaheuristics literature for comparing two algorithms over
    independent runs — plus the r effect size.

    Returns:
        Dict with per-algorithm medians/means, the U statistic, two-sided
        p-value, effect size r, and a plain-language verdict. When the test
        is not applicable (fewer than 8 trials per algorithm), the statistics
        are NaN and the verdict is "inconclusive" — no fabricated
        significance.
    """
    focal = np.asarray(focal_fits, dtype=np.float64)
    reference = np.asarray(reference_fits, dtype=np.float64)
    out = {
        f"{focal_name.lower().replace('-', '_')}_median": float(np.median(focal)),
        f"{reference_name.lower().replace('-', '_')}_median": float(np.median(reference)),
        f"{focal_name.lower().replace('-', '_')}_mean": float(np.mean(focal)),
        f"{reference_name.lower().replace('-', '_')}_mean": float(np.mean(reference)),
        "u_statistic": float("nan"),
        "p_value": float("nan"),
        "effect_size_r": float("nan"),
        "verdict": "inconclusive",
    }
    try:
        test = wilcoxon_rank_sum(focal, reference)
        out["u_statistic"] = test.u_statistic
        out["p_value"] = test.p_value
        out["effect_size_r"] = cohens_r(focal, reference)
        if not test.is_significant:
            out["verdict"] = "no significant difference"
        elif np.median(focal) < np.median(reference):
            out["verdict"] = f"{focal_name} significantly better"
        else:
            out["verdict"] = f"{reference_name} significantly better"
    except ValueError:
        pass  # too few trials per algorithm: leave NaN / inconclusive
    return out


# ==============================================================================
# SUITE DRIVERS
# ==============================================================================

def _run_trials(
    algo_name: str,
    bench: BenchmarkFunction,
    dim: int,
    n_agents: int,
    max_iters: int,
    num_trials: int,
    skroa_kwargs: dict | None = None,
) -> dict:
    """Runs `num_trials` seeded trials of one algorithm on one landscape."""
    best_fits = np.zeros(num_trials)
    exec_times = np.zeros(num_trials)
    peak_memories = np.zeros(num_trials)
    aborts_counts = np.zeros(num_trials)
    eval_counts = np.zeros(num_trials)
    curves: list[np.ndarray] = []

    for trial in range(num_trials):
        seed = 1000 + trial
        tracemalloc.start()
        res = run_one(
            algo_name, bench.evaluator, bench.bounds, dim,
            n_agents, max_iters, seed, skroa_kwargs=skroa_kwargs,
        )
        exec_time = res["exec_time_ms"]
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        best_fits[trial] = res["g_best_fit"]
        exec_times[trial] = exec_time
        peak_memories[trial] = peak_bytes / 1024.0  # Convert bytes to KiB
        aborts_counts[trial] = res.get("total_aborts", 0)
        eval_counts[trial] = res.get("total_evals", 0)
        curves.append(np.asarray(res["convergence_curve"], dtype=np.float64))

    # Align curves of different lengths (DE/CMA may stop early) by padding with
    # each run's final value, then average across trials ignoring padding.
    curve_len = max(len(c) for c in curves)
    curves_matrix = np.full((num_trials, curve_len), np.nan)
    for trial, curve in enumerate(curves):
        curves_matrix[trial, :len(curve)] = curve
        fill = curve[np.isfinite(curve)]
        curves_matrix[trial, len(curve):] = fill[-1] if fill.size else np.nan

    return {
        "best_fits": best_fits,
        "mean_fit": float(np.mean(best_fits)),
        "std_fit": float(np.std(best_fits)),
        "mean_time": float(np.mean(exec_times)),
        "mean_peak_mem": float(np.mean(peak_memories)),
        "mean_aborts": float(np.mean(aborts_counts)),
        "mean_evals": float(np.mean(eval_counts)),
        "mean_curve": np.nanmean(curves_matrix, axis=0),
    }


def execute_benchmarking_suite(
    dim: int = 10,
    n_agents: int = 50,
    max_iters: int = 500,
    num_trials: int = 15,
    output_logs_dir: str = "results/logs",
    output_plots_dir: str = "results/plots",
    opponents: list[str] | None = None,
) -> None:
    """
    Executes automated head-to-head benchmarking comparing SKROA against the
    requested opponents (default: PSO, SciPy DE, and CMA-ES when installed)
    across Rastrigin, Ackley, and Rosenbrock landscapes.

    Requires >= 8 trials for the Wilcoxon rank-sum comparison.
    """
    os.makedirs(output_logs_dir, exist_ok=True)
    os.makedirs(output_plots_dir, exist_ok=True)

    available = get_available_optimizers()
    opponents = opponents or [n for n in available if n != "SKROA"]
    for name in opponents:
        if name not in available:
            raise RuntimeError(
                f"Opponent '{name}' unavailable. Installed set: {available}. "
                f"Install scipy and/or the 'cma' package for the full panel."
            )
    algo_names = ["SKROA"] + list(opponents)

    benchmarks = [RASTRIGIN_BENCHMARK, ACKLEY_BENCHMARK, ROSENBROCK_BENCHMARK]
    csv_path = os.path.join(output_logs_dir, "benchmark_metrics.csv")

    convergence_store: dict[str, dict[str, np.ndarray]] = {}

    print(f"[INFO] Initializing KRNA Benchmarking Suite (D={dim}, Agents={n_agents}, Iters={max_iters}, Trials={num_trials})")
    print(f"[INFO] Opponents: {', '.join(algo_names)}")
    print("=" * 90)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "Benchmark", "Algorithm", "Dimension", "Agents", "Max_Iters",
            "Mean_Best_Fitness", "Std_Best_Fitness", "Mean_Eval_Counts",
            "Mean_Exec_Time_ms", "Peak_Memory_KiB", "Mean_Aborts",
            "P_Value_RankSum", "Effect_Size_r", "Statistical_Verdict"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for bench in benchmarks:
            print(f"[INFO] Benchmarking Landscape: {bench.name}...")
            generate_3d_surface_plot(bench, output_plots_dir)
            convergence_store[bench.name] = {}

            fits_by_algo: dict[str, np.ndarray] = {}

            for algo_name in algo_names:
                stats_row = run_trials_and_summarize(
                    algo_name, bench, dim, n_agents, max_iters, num_trials,
                    fits_by_algo=fits_by_algo,
                )
                convergence_store[bench.name][algo_name] = stats_row["mean_curve"]

                verdict = stats_row["summary"]["verdict"] if "summary" in stats_row else "NA"
                p_str = (
                    f"{stats_row['summary']['p_value']:.6g}"
                    if "summary" in stats_row and np.isfinite(stats_row["summary"]["p_value"])
                    else "NA"
                )
                r_str = (
                    f"{stats_row['summary']['effect_size_r']:.3f}"
                    if "summary" in stats_row and np.isfinite(stats_row["summary"]["effect_size_r"])
                    else "NA"
                )

                writer.writerow({
                    "Benchmark": bench.name,
                    "Algorithm": algo_name,
                    "Dimension": dim,
                    "Agents": n_agents,
                    "Max_Iters": max_iters,
                    "Mean_Best_Fitness": f"{stats_row['mean_fit']:.8f}",
                    "Std_Best_Fitness": f"{stats_row['std_fit']:.8f}",
                    "Mean_Eval_Counts": f"{stats_row['mean_evals']:.1f}",
                    "Mean_Exec_Time_ms": f"{stats_row['mean_time']:.2f}",
                    "Peak_Memory_KiB": f"{stats_row['mean_peak_mem']:.2f}",
                    "Mean_Aborts": f"{stats_row['mean_aborts']:.1f}",
                    "P_Value_RankSum": p_str,
                    "Effect_Size_r": r_str,
                    "Statistical_Verdict": verdict,
                })

                print(
                    f"  -> [{algo_name:<8}] Mean Best Fit: {stats_row['mean_fit']:11.6f} ± {stats_row['std_fit']:<10.6f} | "
                    f"Evals: {stats_row['mean_evals']:8.0f} | Time: {stats_row['mean_time']:6.2f}ms | "
                    f"Peak Mem: {stats_row['mean_peak_mem']:6.1f}KiB | Aborts: {stats_row['mean_aborts']:4.1f}"
                )
                if "summary" in stats_row:
                    s = stats_row["summary"]
                    print(
                        f"  -> [STATS ] Wilcoxon rank-sum U={s['u_statistic']:.1f} | "
                        f"p={s['p_value']:.3e} | r={s['effect_size_r']:.2f} | {s['verdict']}"
                    )

    plot_convergence_grid(
        benchmarks, convergence_store, output_plots_dir, dim=dim,
        title="KRNA Algorithm Optimization: SKROA vs. Opponents Convergence Trajectories",
        filename_stem="convergence_comparison",
    )
    print("=" * 90)
    print(f"[SUCCESS] Telemetry metrics saved to     : {csv_path}")
    print(f"[SUCCESS] Convergence & 3D plots saved to: {output_plots_dir}/")


def run_trials_and_summarize(
    algo_name: str,
    bench: BenchmarkFunction,
    dim: int,
    n_agents: int,
    max_iters: int,
    num_trials: int,
    fits_by_algo: dict[str, np.ndarray],
    skroa_kwargs: dict | None = None,
) -> dict:
    """
    Runs all trials for one algorithm on one landscape, stores its fitness
    sample in `fits_by_algo`, and (for every non-SKROA algorithm) compares it
    against the stored SKROA sample with the Wilcoxon rank-sum test.
    """
    row = _run_trials(algo_name, bench, dim, n_agents, max_iters, num_trials, skroa_kwargs)
    fits_by_algo[algo_name] = row["best_fits"]

    if algo_name != "SKROA" and "SKROA" in fits_by_algo:
        row["summary"] = summarize_comparison(
            fits_by_algo["SKROA"], row["best_fits"],
            focal_name="SKROA", reference_name=algo_name,
        )
    return row


_ABLATION_VARIANTS: list[tuple[str, dict]] = [
    ("Full SKROA", {}),
    ("No Biphasic (all-explore)", {"use_biphasic": False}),
    ("No Clamping", {"use_clamping": False}),
    ("No Culm-Abortion", {"use_culm_abortion": False}),
]


def execute_ablation_suite(
    dim: int = 10,
    n_agents: int = 50,
    max_iters: int = 500,
    num_trials: int = 15,
    output_logs_dir: str = "results/logs",
    output_plots_dir: str = "results/plots",
) -> None:
    """
    Executes an ablation study: reruns SKROA on every landscape with each
    operator disabled in turn and tests each crippled variant against the full
    algorithm with the Wilcoxon rank-sum test (requires >= 8 trials).

    Variants:
        - No Biphasic: the swarm never splits; every agent explores via Lévy
          flights (measures the exploitation phase's contribution).
        - No Clamping: Sympodial Clamping repulsion is switched off (measures
          the anti-collision diversity mechanism).
        - No Culm-Abortion: stalled exploiting agents are never pruned
          (measures the budget-reallocation mechanism).

    All variants share the same per-trial seeds (1000 + trial), so differences
    between a variant and Full SKROA are paired at the seed level.
    """
    os.makedirs(output_logs_dir, exist_ok=True)
    os.makedirs(output_plots_dir, exist_ok=True)

    benchmarks = [RASTRIGIN_BENCHMARK, ACKLEY_BENCHMARK, ROSENBROCK_BENCHMARK]
    csv_path = os.path.join(output_logs_dir, "ablation_metrics.csv")

    convergence_store: dict[str, dict[str, np.ndarray]] = {}

    print(f"[INFO] Initializing KRNA Ablation Suite (D={dim}, Agents={n_agents}, Iters={max_iters}, Trials={num_trials})")
    print(f"[INFO] Variants: {', '.join(name for name, _ in _ABLATION_VARIANTS)}")
    print("=" * 90)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "Benchmark", "Variant", "Dimension", "Agents", "Max_Iters",
            "Mean_Best_Fitness", "Std_Best_Fitness", "Mean_Eval_Counts",
            "Mean_Exec_Time_ms", "P_Value_RankSum", "Effect_Size_r",
            "Statistical_Verdict"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for bench in benchmarks:
            print(f"[INFO] Ablation Landscape: {bench.name}...")
            convergence_store[bench.name] = {}
            fits_by_variant: dict[str, np.ndarray] = {}

            for variant_name, skroa_kwargs in _ABLATION_VARIANTS:
                row = _run_trials(
                    "SKROA", bench, dim, n_agents, max_iters, num_trials,
                    skroa_kwargs=skroa_kwargs,
                )
                fits_by_variant[variant_name] = row["best_fits"]
                convergence_store[bench.name][variant_name] = row["mean_curve"]

                if variant_name != "Full SKROA":
                    summary = summarize_comparison(
                        fits_by_variant["Full SKROA"], row["best_fits"],
                        focal_name="Full SKROA", reference_name=variant_name,
                    )
                    p_str = f"{summary['p_value']:.6g}" if np.isfinite(summary["p_value"]) else "NA"
                    r_str = f"{summary['effect_size_r']:.3f}" if np.isfinite(summary["effect_size_r"]) else "NA"
                    verdict = summary["verdict"]
                else:
                    p_str, r_str, verdict = "NA", "NA", "reference"

                writer.writerow({
                    "Benchmark": bench.name,
                    "Variant": variant_name,
                    "Dimension": dim,
                    "Agents": n_agents,
                    "Max_Iters": max_iters,
                    "Mean_Best_Fitness": f"{row['mean_fit']:.8f}",
                    "Std_Best_Fitness": f"{row['std_fit']:.8f}",
                    "Mean_Eval_Counts": f"{row['mean_evals']:.1f}",
                    "Mean_Exec_Time_ms": f"{row['mean_time']:.2f}",
                    "P_Value_RankSum": p_str,
                    "Effect_Size_r": r_str,
                    "Statistical_Verdict": verdict,
                })

                print(
                    f"  -> [{variant_name:<26}] Mean Best Fit: {row['mean_fit']:11.6f} ± {row['std_fit']:<10.6f} | "
                    f"Evals: {row['mean_evals']:8.0f} | Time: {row['mean_time']:6.2f}ms"
                )
                if variant_name != "Full SKROA":
                    print(
                        f"  -> [STATS ] vs Full SKROA: p={p_str} | r={r_str} | {verdict}"
                    )

    plot_convergence_grid(
        benchmarks, convergence_store, output_plots_dir, dim=dim,
        title="SKROA Ablation Study: Operator Contribution Across Landscapes",
        filename_stem="ablation_convergence",
    )
    print("=" * 90)
    print(f"[SUCCESS] Ablation metrics saved to     : {csv_path}")
    print(f"[SUCCESS] Ablation convergence plots    : {output_plots_dir}/ablation_convergence.png")


if __name__ == "__main__":
    execute_benchmarking_suite()
