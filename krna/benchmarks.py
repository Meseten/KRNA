"""
KRNA / SKROA Benchmarking Objective Landscapes & Automated Telemetry Suite
==============================================================================
Provides vectorized NumPy implementations of canonical optimization test
functions (Rastrigin, Ackley, Rosenbrock) and an automated benchmarking engine
comparing SKROA against baseline PSO.

Logs structured metrics to results/logs/benchmark_metrics.csv and exports
high-D convergence curves and 3D surface topology plots to results/plots/.
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
from krna.stats import wilcoxon_rank_sum, cohens_r, bonferroni_correct


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
# TELEMETRY, MEMORY PROFILING & VISUALIZATION ENGINE
# ==============================================================================

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


def plot_convergence_curves(
    benchmarks: list[BenchmarkFunction],
    skroa_curves: dict[str, np.ndarray],
    pso_curves: dict[str, np.ndarray],
    output_dir: str,
    dim: int = 10
) -> None:
    """
    Exports a publication-grade subplot grid comparing the mean convergence
    trajectories of SKROA and PSO across all tested landscapes on a log scale.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_panels = max(1, len(benchmarks))
    fig, axes = plt.subplots(1, n_panels, figsize=(6.0 * n_panels, 5.5), squeeze=False)
    axes = axes[0]
    fig.suptitle(
        "KRNA Algorithm Optimization: SKROA vs. Baseline PSO Convergence Trajectories",
        fontsize=15,
        fontweight="bold",
        y=0.98
    )

    for idx, bench in enumerate(benchmarks):
        ax = axes[idx]
        skroa_mean = skroa_curves[bench.name]
        pso_mean = pso_curves[bench.name]
        iters = np.arange(1, len(skroa_mean) + 1)

        ax.plot(iters, skroa_mean, label="SKROA (Biphasic + Pruning)", color="#1f77b4", linewidth=2.2)
        ax.plot(iters, pso_mean, label="PSO (Baseline)", color="#d62728", linewidth=2.0, linestyle="--")

        ax.set_title(f"{bench.name} Landscape ($D={dim}$)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Iteration ($t$)", fontsize=11)
        ax.set_ylabel("Best Objective Value $\\bar{f}(\\mathbf{g}^*)$", fontsize=11)
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", frameon=True)

    plt.tight_layout(rect=[0, 0.0, 1, 0.94])
    png_path = os.path.join(output_dir, "convergence_comparison.png")
    pdf_path = os.path.join(output_dir, "convergence_comparison.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize_comparison(skroa_fits: np.ndarray, pso_fits: np.ndarray) -> dict:
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
    skroa = np.asarray(skroa_fits, dtype=np.float64)
    pso = np.asarray(pso_fits, dtype=np.float64)
    out = {
        "skroa_median": float(np.median(skroa)),
        "pso_median": float(np.median(pso)),
        "skroa_mean": float(np.mean(skroa)),
        "pso_mean": float(np.mean(pso)),
        "u_statistic": float("nan"),
        "p_value": float("nan"),
        "effect_size_r": float("nan"),
        "verdict": "inconclusive",
    }
    try:
        test = wilcoxon_rank_sum(skroa, pso)
        out["u_statistic"] = test.u_statistic
        out["p_value"] = test.p_value
        out["effect_size_r"] = cohens_r(skroa, pso)
        if not test.is_significant:
            out["verdict"] = "no significant difference"
        elif out["skroa_median"] < out["pso_median"]:
            out["verdict"] = "SKROA significantly better"
        else:
            out["verdict"] = "PSO significantly better"
    except ValueError:
        pass  # too few trials per algorithm: leave NaN / inconclusive
    return out


def execute_benchmarking_suite(
    dim: int = 10,
    n_agents: int = 50,
    max_iters: int = 500,
    num_trials: int = 15,
    output_logs_dir: str = "results/logs",
    output_plots_dir: str = "results/plots"
) -> None:
    """
    Executes automated head-to-head benchmarking comparing SKROA against PSO
    across Rastrigin, Ackley, and Rosenbrock landscapes.
    """
    os.makedirs(output_logs_dir, exist_ok=True)
    os.makedirs(output_plots_dir, exist_ok=True)

    benchmarks = [RASTRIGIN_BENCHMARK, ACKLEY_BENCHMARK, ROSENBROCK_BENCHMARK]
    csv_path = os.path.join(output_logs_dir, "benchmark_metrics.csv")

    skroa_convergence_store: dict[str, np.ndarray] = {}
    pso_convergence_store: dict[str, np.ndarray] = {}

    print(f"[INFO] Initializing KRNA Benchmarking Suite (D={dim}, Agents={n_agents}, Iters={max_iters}, Trials={num_trials})")
    print("=" * 90)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "Benchmark", "Algorithm", "Dimension", "Agents", "Max_Iters",
            "Mean_Best_Fitness", "Std_Best_Fitness", "Mean_Exec_Time_ms",
            "Peak_Memory_KiB", "Mean_Aborts",
            "P_Value_RankSum", "Effect_Size_r", "Statistical_Verdict"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for bench in benchmarks:
            print(f"[INFO] Benchmarking Landscape: {bench.name}...")
            generate_3d_surface_plot(bench, output_plots_dir)

            # Both algorithms collect the same number of independent runs so a
            # rank-sum comparison is possible afterwards.
            skroa_best_fits = np.zeros(num_trials)
            pso_best_fits = np.zeros(num_trials)

            for algo_name in ["SKROA", "PSO"]:
                best_fits = skroa_best_fits if algo_name == "SKROA" else pso_best_fits
                exec_times = np.zeros(num_trials)
                peak_memories = np.zeros(num_trials)
                aborts_counts = np.zeros(num_trials)
                curves_matrix = np.zeros((num_trials, max_iters))

                for trial in range(num_trials):
                    seed = 1000 + trial
                    tracemalloc.start()
                    start_time = time.perf_counter()

                    if algo_name == "SKROA":
                        optimizer = SKROA(
                            evaluator=bench.evaluator,
                            bounds=bench.bounds,
                            dim=dim,
                            n_agents=n_agents,
                            max_iters=max_iters,
                            seed=seed
                        )
                    else:
                        optimizer = PSO(
                            evaluator=bench.evaluator,
                            bounds=bench.bounds,
                            dim=dim,
                            n_agents=n_agents,
                            max_iters=max_iters,
                            seed=seed
                        )

                    res = optimizer.optimize()
                    exec_time = (time.perf_counter() - start_time) * 1000.0
                    _, peak_bytes = tracemalloc.get_traced_memory()
                    tracemalloc.stop()

                    best_fits[trial] = res["g_best_fit"]
                    exec_times[trial] = exec_time
                    peak_memories[trial] = peak_bytes / 1024.0  # Convert bytes to KiB
                    aborts_counts[trial] = res.get("total_aborts", 0)
                    curves_matrix[trial, :] = res["convergence_curve"]

                mean_fit = float(np.mean(best_fits))
                std_fit = float(np.std(best_fits))
                mean_time = float(np.mean(exec_times))
                mean_peak_mem = float(np.mean(peak_memories))
                mean_aborts = float(np.mean(aborts_counts))
                mean_curve = np.mean(curves_matrix, axis=0)

                if algo_name == "SKROA":
                    skroa_convergence_store[bench.name] = mean_curve
                else:
                    pso_convergence_store[bench.name] = mean_curve

                if algo_name == "PSO":
                    stats = summarize_comparison(skroa_best_fits, pso_best_fits)
                    stat_row = {
                        "P_Value_RankSum": f"{stats['p_value']:.6g}" if np.isfinite(stats["p_value"]) else "NA",
                        "Effect_Size_r": f"{stats['effect_size_r']:.3f}" if np.isfinite(stats["effect_size_r"]) else "NA",
                        "Statistical_Verdict": stats["verdict"],
                    }
                else:
                    stat_row = {"P_Value_RankSum": "NA", "Effect_Size_r": "NA", "Statistical_Verdict": "NA"}

                writer.writerow({
                    "Benchmark": bench.name,
                    "Algorithm": algo_name,
                    "Dimension": dim,
                    "Agents": n_agents,
                    "Max_Iters": max_iters,
                    "Mean_Best_Fitness": f"{mean_fit:.8f}",
                    "Std_Best_Fitness": f"{std_fit:.8f}",
                    "Mean_Exec_Time_ms": f"{mean_time:.2f}",
                    "Peak_Memory_KiB": f"{mean_peak_mem:.2f}",
                    "Mean_Aborts": f"{mean_aborts:.1f}",
                    **stat_row
                })

                print(
                    f"  -> [{algo_name:<5}] Mean Best Fit: {mean_fit:11.6f} ± {std_fit:<10.6f} | "
                    f"Time: {mean_time:6.2f}ms | Peak Mem: {mean_peak_mem:6.1f}KiB | "
                    f"Aborts: {mean_aborts:4.1f}"
                )

                if algo_name == "PSO":
                    print(
                        f"  -> [STATS ] Wilcoxon rank-sum U={stats['u_statistic']:.1f} | "
                        f"p={stats['p_value']:.3e} | r={stats['effect_size_r']:.2f} | {stats['verdict']}"
                    )

    plot_convergence_curves(benchmarks, skroa_convergence_store, pso_convergence_store, output_plots_dir, dim=dim)
    print("=" * 90)
    print(f"[SUCCESS] Telemetry metrics saved to     : {csv_path}")
    print(f"[SUCCESS] Convergence & 3D plots saved to: {output_plots_dir}/")


if __name__ == "__main__":
    execute_benchmarking_suite()