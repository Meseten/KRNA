"""
KRNA / SKROA Parameter Sensitivity Analysis
==============================================================================
Executes a 2D grid search over the Culm-Abortion threshold (tau_stagnation)
and Sympodial Clamping distance (epsilon_clamp) to evaluate their non-linear
effects on global convergence (mean best fitness).

Outputs a parameter matrix to CSV and a color-mapped heatmap to the plots directory.
"""

from __future__ import annotations
import os
import csv
import time
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from krna.benchmarks import get_benchmark
from krna.skroa import SKROA


def execute_sensitivity_sweep(
    benchmark_name: str = "rastrigin",
    dim: int = 10,
    n_agents: int = 50,
    max_iters: int = 300,
    num_trials: int = 5,
    output_logs_dir: str = "results/logs",
    output_plots_dir: str = "results/plots"
) -> None:
    """
    Runs a grid search over SKROA's unique hyperparameters and generates telemetry.
    """
    os.makedirs(output_logs_dir, exist_ok=True)
    os.makedirs(output_plots_dir, exist_ok=True)
    
    benchmark = get_benchmark(benchmark_name)
    
    tau_levels = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    epsilon_levels = [0.0, 0.01, 0.1, 0.5, 1.0] 
    
    n_tau = len(tau_levels)
    n_eps = len(epsilon_levels)
    
    mean_fitness_matrix = np.zeros((n_tau, n_eps), dtype=np.float64)
    mean_aborts_matrix = np.zeros((n_tau, n_eps), dtype=np.float64)
    
    csv_path = os.path.join(output_logs_dir, f"sensitivity_{benchmark_name}.csv")
    
    print(f"[INFO] Initializing KRNA Sensitivity Analysis on {benchmark.name} (D={dim})")
    print(f"[INFO] Grid: {n_tau} Tau levels x {n_eps} Epsilon levels = {n_tau * n_eps} configurations")
    print("=" * 90)
    
    total_start_time = time.perf_counter()
    
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["Tau_Stagnation", "Epsilon_Clamp", "Mean_Best_Fitness", "Mean_Aborts", "Mean_Exec_Time_ms"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, tau in enumerate(tau_levels):
            for j, eps in enumerate(epsilon_levels):
                
                trial_fitness = np.zeros(num_trials)
                trial_aborts = np.zeros(num_trials)
                trial_times = np.zeros(num_trials)
                
                for trial in range(num_trials):
                    seed = 2000 + trial
                    start_trial = time.perf_counter()
                    
                    optimizer = SKROA(
                        evaluator=benchmark.evaluator,
                        bounds=benchmark.bounds,
                        dim=dim,
                        n_agents=n_agents,
                        max_iters=max_iters,
                        tau_stagnation=tau,
                        epsilon_clamp=eps,
                        seed=seed
                    )
                    
                    res = optimizer.optimize()
                    
                    trial_times[trial] = (time.perf_counter() - start_trial) * 1000.0
                    trial_fitness[trial] = res["g_best_fit"]
                    trial_aborts[trial] = res.get("total_aborts", 0)
                
                mean_fit = np.mean(trial_fitness)
                mean_aborts = np.mean(trial_aborts)
                mean_time = np.mean(trial_times)
                
                mean_fitness_matrix[i, j] = mean_fit
                mean_aborts_matrix[i, j] = mean_aborts
                
                writer.writerow({
                    "Tau_Stagnation": f"{tau:.1e}",
                    "Epsilon_Clamp": f"{eps:.2f}",
                    "Mean_Best_Fitness": f"{mean_fit:.6f}",
                    "Mean_Aborts": f"{mean_aborts:.1f}",
                    "Mean_Exec_Time_ms": f"{mean_time:.2f}"
                })
                
                print(f"  -> Config [Tau={tau:.1e}, Eps={eps:<4.2f}] | Mean Fit: {mean_fit:10.5f} | Aborts: {mean_aborts:4.1f}")

    total_time = time.perf_counter() - total_start_time
    print("=" * 90)
    print(f"[SUCCESS] Grid search completed in {total_time:.2f} seconds.")
    
    # ==============================================================================
    # HEATMAP VISUALIZATION (Fixed for identical value crashes)
    # ==============================================================================
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    vmin = float(np.min(mean_fitness_matrix))
    vmax = float(np.max(mean_fitness_matrix))
    
    # Strict fallback check to prevent Matplotlib LogNorm crash
    if np.isnan(vmin) or np.isnan(vmax) or vmin >= vmax:
        vmin = max(vmin * 0.9, 1e-8)
        vmax = max(vmax * 1.1, 1e-7)
    else:
        vmin = max(vmin, 1e-8)
        vmax = max(vmax, 1e-7)
    
    cax = ax.imshow(
        mean_fitness_matrix, 
        cmap="viridis_r",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        aspect="auto"
    )
    
    ax.set_xticks(np.arange(n_eps))
    ax.set_yticks(np.arange(n_tau))
    ax.set_xticklabels([f"{e:.2f}" for e in epsilon_levels])
    ax.set_yticklabels([f"{t:.1e}" for t in tau_levels])
    
    ax.set_xlabel("Sympodial Clamping Distance ($\\epsilon_{\\text{clamp}}$)", fontsize=11, labelpad=10)
    ax.set_ylabel("Culm-Abortion Threshold ($\\tau_{\\text{stagnation}}$)", fontsize=11, labelpad=10)
    ax.set_title(f"SKROA Sensitivity Heatmap: {benchmark.name} ($D={dim}$)\nMean Best Fitness (Lower is Better)", fontsize=13, fontweight="bold", pad=15)
    
    for i in range(n_tau):
        for j in range(n_eps):
            fit_val = mean_fitness_matrix[i, j]
            text_color = "black" if cax.norm(fit_val) > 0.5 else "white"
            annotation = f"{fit_val:.2f}\n({mean_aborts_matrix[i, j]:.0f} aborts)"
            ax.text(j, i, annotation, ha="center", va="center", color=text_color, fontsize=9)
            
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Best Fitness (Log Scale)", rotation=270, labelpad=20)
    
    plt.tight_layout()
    plot_path = os.path.join(output_plots_dir, f"sensitivity_heatmap_{benchmark_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"[SUCCESS] Heatmap saved to : {plot_path}")
    print(f"[SUCCESS] Raw data saved to: {csv_path}")

if __name__ == "__main__":
    execute_sensitivity_sweep(benchmark_name="rastrigin")