"""
KRNA / Command Line Interface
==============================================================================
Provides the global terminal commands for the Kawayan Rhizome Network Algorithm.
"""

import argparse
import sys
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="KRNA (SKROA) Optimization CLI Framework",
        prog="krna"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run SKROA vs PSO on standard mathematical landscapes")
    bench_parser.add_argument("--dim", type=int, default=10, help="Dimensionality (default: 10)")
    bench_parser.add_argument("--agents", type=int, default=50, help="Swarm size (default: 50)")
    bench_parser.add_argument("--iters", type=int, default=500, help="Max iterations (default: 500)")
    bench_parser.add_argument("--trials", type=int, default=15, help="Number of trials (default: 15)")

    # Command: tune
    tune_parser = subparsers.add_parser("tune", help="Run SKROA ML Hyperparameter Tuner on Breast Cancer SVM")
    tune_parser.add_argument("--agents", type=int, default=15, help="Swarm size (default: 15)")
    tune_parser.add_argument("--iters", type=int, default=20, help="Max iterations (default: 20)")

    # Command: mo-benchmark
    mo_parser = subparsers.add_parser("mo-benchmark", help="Run MO-SKROA on Multi-Objective Pareto fronts (ZDT1, ZDT2)")
    mo_parser.add_argument("--dim", type=int, default=10, help="Dimensionality (default: 10)")
    mo_parser.add_argument("--agents", type=int, default=100, help="Swarm size (default: 100)")
    mo_parser.add_argument("--iters", type=int, default=250, help="Max iterations (default: 250)")

    args = parser.parse_args()

    if args.command == "benchmark":
        from krna.benchmarks import execute_benchmarking_suite
        execute_benchmarking_suite(dim=args.dim, n_agents=args.agents, max_iters=args.iters, num_trials=args.trials)
        
    elif args.command == "tune":
        from krna.ml_tuning import SKROAMLTuner
        from sklearn.svm import SVC
        from sklearn.datasets import load_breast_cancer
        
        data = load_breast_cancer()
        X, y = data.data, data.target
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
        
        svm_param_space = {
            'C': {'type': 'log_float', 'min': 1e-3, 'max': 1e3},
            'gamma': {'type': 'log_float', 'min': 1e-4, 'max': 1e1},
            'kernel': {'type': 'categorical', 'values': ['linear', 'rbf', 'sigmoid']}
        }
        
        tuner = SKROAMLTuner(model_class=SVC, param_space=svm_param_space, X=X, y=y, n_agents=args.agents, max_iters=args.iters)
        results = tuner.tune()
        print(f"[SUCCESS] Best Accuracy: {results['best_accuracy_percent']:.2f}% | Hyperparams: {results['best_hyperparams']}")

    elif args.command == "mo-benchmark":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from krna.mo_benchmarks import ZDT1_BENCHMARK, ZDT2_BENCHMARK
        from krna.mo_skroa import MOSKROA
        
        os.makedirs("results/plots", exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for idx, bench in enumerate([ZDT1_BENCHMARK, ZDT2_BENCHMARK]):
            print(f"[INFO] Running MO-SKROA on {bench.name} (D={args.dim})...")
            optimizer = MOSKROA(
                evaluator=bench.evaluator, bounds=bench.bounds, dim=args.dim, 
                n_objectives=bench.n_objectives, n_agents=args.agents, max_iters=args.iters
            )
            res = optimizer.optimize()
            
            pf = res["pareto_front_fitness"]
            axes[idx].scatter(pf[:, 0], pf[:, 1], color="blue", alpha=0.7, edgecolor="k", label="MO-SKROA Discovered Front")
            axes[idx].set_title(f"{bench.name} Pareto Front", fontweight="bold")
            axes[idx].set_xlabel("Objective 1 ($f_1$)", fontsize=11)
            axes[idx].set_ylabel("Objective 2 ($f_2$)", fontsize=11)
            axes[idx].grid(True, linestyle=":", alpha=0.6)
            axes[idx].legend()
            
            print(f"  -> Discovered {res['archive_size']} optimal trade-off solutions in {res['exec_time_ms']:.2f}ms.")
            
        plt.tight_layout()
        plot_path = "results/plots/pareto_fronts.png"
        plt.savefig(plot_path, dpi=300)
        print(f"================================================================================")
        print(f"[SUCCESS] Pareto front visualization saved to: {plot_path}")
        
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()