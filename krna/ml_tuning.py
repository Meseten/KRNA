"""
KRNA / SKROA Applied Machine Learning Tuner
==============================================================================
Bridges the continuous SKROA optimization engine with scikit-learn models.
Maps continuous [0, 1] algorithmic coordinates to discrete, categorical, and
log-scaled Machine Learning hyperparameters to find the optimal model configuration.
"""

from __future__ import annotations
import time
from typing import Any, Callable
import numpy as np

# ML Stack
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from joblib import Parallel, delayed

from krna.skroa import SKROA


class HyperparameterMapper:
    """
    Translates SKROA's continuous [0.0, 1.0] search space into usable ML hyperparameters.
    """
    def __init__(self, param_space: dict[str, dict[str, Any]]):
        """
        param_space example:
        {
            'C': {'type': 'log_float', 'min': 1e-3, 'max': 1e3},
            'kernel': {'type': 'categorical', 'values': ['linear', 'rbf', 'poly']},
            'degree': {'type': 'int', 'min': 2, 'max': 5}
        }
        """
        self.param_space = param_space
        self.keys = list(param_space.keys())
        self.dim = len(self.keys)

    def decode(self, vector: np.ndarray) -> dict[str, Any]:
        """
        Maps a single continuous vector in [0.0, 1.0]^D to a parameter dictionary.
        """
        params = {}
        for i, key in enumerate(self.keys):
            val = np.clip(vector[i], 0.0, 1.0) # Ensure strict [0, 1] bounds
            space = self.param_space[key]
            ptype = space['type']
            
            if ptype == 'int':
                p_min, p_max = space['min'], space['max']
                # Round to nearest integer within bounds
                params[key] = int(np.round(p_min + val * (p_max - p_min)))
                
            elif ptype == 'float':
                p_min, p_max = space['min'], space['max']
                params[key] = float(p_min + val * (p_max - p_min))
                
            elif ptype == 'log_float':
                # Excellent for learning rates and regularization parameters
                log_min, log_max = np.log10(space['min']), np.log10(space['max'])
                log_val = log_min + val * (log_max - log_min)
                params[key] = float(10 ** log_val)
                
            elif ptype == 'categorical':
                choices = space['values']
                idx = int(np.floor(val * len(choices)))
                # Edge case if val == 1.0
                idx = min(idx, len(choices) - 1)
                params[key] = choices[idx]
                
            else:
                raise ValueError(f"Unknown parameter type: {ptype}")
                
        return params


class SKROAMLTuner:
    """
    Uses SKROA to optimize hyperparameters for a given scikit-learn model.
    """
    def __init__(
        self,
        model_class: Callable,
        param_space: dict[str, dict[str, Any]],
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 3,
        n_agents: int = 20,
        max_iters: int = 30,
        n_jobs: int = -1,
        seed: int = 42
    ):
        self.model_class = model_class
        self.mapper = HyperparameterMapper(param_space)
        self.X = X
        self.y = y
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        self.n_jobs = n_jobs
        
        # SKROA expects bounds. We fix the search space to [0.0, 1.0] for all dimensions.
        self.skroa_engine = SKROA(
            evaluator=self._vectorized_cv_evaluator,
            bounds=(0.0, 1.0),
            dim=self.mapper.dim,
            n_agents=n_agents,
            max_iters=max_iters,
            delta_threshold=None, # Use dynamic median threshold
            tau_stagnation=1e-3,  # Abort if accuracy doesn't improve by 0.1%
            max_stagnation_steps=3,
            epsilon_clamp=0.05,   # Keep agents 5% apart in coordinate space
            seed=seed
        )

    def _evaluate_single(self, coords: np.ndarray) -> float:
        """
        Decodes coordinates, trains the model via CV, and returns (1 - accuracy).
        """
        hyperparams = self.mapper.decode(coords)
        model = self.model_class(**hyperparams, random_state=42 if 'random_state' in self.model_class().get_params() else None)
        
        try:
            # Calculate cross-validation accuracy
            scores = cross_val_score(model, self.X, self.y, cv=self.cv, scoring='accuracy', n_jobs=1)
            mean_acc = np.mean(scores)
            return 1.0 - mean_acc  # Objective is minimization (0.0 = 100% accuracy)
        except Exception:
            # If invalid hyperparameter combination (e.g., math error in kernel), return worst score
            return 1.0

    def _vectorized_cv_evaluator(self, positions: np.ndarray) -> np.ndarray:
        """
        Evaluates the N x D positions matrix in parallel.
        """
        arr = np.atleast_2d(positions)
        n_samples = arr.shape[0]
        
        # Dispatch model training to multiple CPU cores
        fitness_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._evaluate_single)(arr[i]) for i in range(n_samples)
        )
        
        return np.array(fitness_list, dtype=np.float64)

    def tune(self) -> dict:
        """
        Executes the tuning process and returns the best model configuration.
        """
        print(f"[INFO] Initiating SKROA ML Tuner...")
        print(f"  -> Model: {self.model_class.__name__}")
        print(f"  -> Hyperparameter Dimensions: {self.mapper.dim}")
        print(f"  -> Swarm Size: {self.skroa_engine.n_agents} | Max Iters: {self.skroa_engine.max_iters}")
        
        start_time = time.perf_counter()
        res = self.skroa_engine.optimize()
        exec_time = time.perf_counter() - start_time
        
        best_hyperparams = self.mapper.decode(res["g_best_pos"])
        best_accuracy = (1.0 - res["g_best_fit"]) * 100.0
        
        return {
            "best_hyperparams": best_hyperparams,
            "best_accuracy_percent": best_accuracy,
            "convergence_curve_error": res["convergence_curve"],
            "total_aborts": res["total_aborts"],
            "exec_time_sec": exec_time
        }


if __name__ == "__main__":
    # 1. Load a real-world dataset (Breast Cancer Classification)
    print("[INFO] Loading Breast Cancer Dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Normalize features (Standardization is required for SVMs)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / (X_std + 1e-8)

    # 2. Define the highly non-linear Hyperparameter Space for an SVM
    svm_param_space = {
        'C': {'type': 'log_float', 'min': 1e-3, 'max': 1e3},
        'gamma': {'type': 'log_float', 'min': 1e-4, 'max': 1e1},
        'kernel': {'type': 'categorical', 'values': ['linear', 'rbf', 'sigmoid']}
    }

    # 3. Initialize SKROA Tuner
    # We use fewer agents and iterations here because training ML models takes time
    tuner = SKROAMLTuner(
        model_class=SVC,
        param_space=svm_param_space,
        X=X,
        y=y,
        cv_folds=5,
        n_agents=15,
        max_iters=20,
        n_jobs=-1,  # Use all available CPU cores
        seed=101
    )

    # 4. Execute Optimization
    print("=" * 80)
    results = tuner.tune()
    
    print("=" * 80)
    print(f"[SUCCESS] SKROA Hyperparameter Optimization Complete!")
    print(f"  -> Best Validation Accuracy: {results['best_accuracy_percent']:.2f}%")
    print(f"  -> Best Hyperparameters    : {results['best_hyperparams']}")
    print(f"  -> Memory Pruning (Aborts) : {results['total_aborts']} dead-end threads culled.")
    print(f"  -> Total Tuning Time       : {results['exec_time_sec']:.2f} seconds.")