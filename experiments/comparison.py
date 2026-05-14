"""GA vs CNN-GA comparison on SEEDS_10 - main paper experiment."""

import json
import os
from typing import Any, Callable, Dict, List

import matplotlib
matplotlib.use("Agg")

import numpy as np
from scipy.stats import ttest_rel

from algorithms import CNNGuidedGA, GeneticAlgorithm
from algorithms.mutations import random_relocate_mutation
from experiments import plots
from experiments.cases import CNN_PARAMS, HARD_CASE, M_SWAPS, SEEDS_10


LOGS_DIR = "logs"
PLOTS_DIR = "plots"


# ---------------------------------------------------------------------------
# Log serialization
# ---------------------------------------------------------------------------

def save_log(log: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def load_log(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_separator(title: str) -> None:
    line = "=" * 60
    print(f"\n{line}\n  {title}\n{line}")


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_ga(seed: int) -> Dict[str, Any]:
    ga = GeneticAlgorithm(
        **HARD_CASE,
        K_swaps=M_SWAPS,
        rng_seed=seed,
        mutation_op=random_relocate_mutation,
    )
    _, log = ga.run(verbose=True)
    log["algo"] = "ga"
    log["seed"] = seed
    return log


def run_cnn_ga(seed: int) -> Dict[str, Any]:
    cnn = CNNGuidedGA(**HARD_CASE, **CNN_PARAMS, rng_seed=seed)
    _, log = cnn.run(verbose=True)
    log["algo"] = "cnn_ga"
    log["seed"] = seed
    return log


def run_or_load(
    name: str, runner: Callable[[int], Dict[str, Any]],
    seeds: List[int] = SEEDS_10, log_dir: str = LOGS_DIR,
) -> List[Dict[str, Any]]:
    """Run all seeds for an algorithm or load cached logs from disk."""
    logs: List[Dict[str, Any]] = []
    total = len(seeds)
    for i, seed in enumerate(seeds):
        log_path = f"{log_dir}/run_{name}_{seed}.json"
        if os.path.exists(log_path):
            print_separator(
                f"{name.upper()} run {i + 1}/{total} | seed={seed} [from disk]"
            )
            log = load_log(log_path)
            print(f"  Loaded: {log_path} (best={log['best_fitness_per_gen'][-1]})")
        else:
            print_separator(f"{name.upper()} run {i + 1}/{total} | seed={seed}")
            log = runner(seed)
            save_log(log, log_path)
            print(f"  Log saved: {log_path}")
            print(f"  >> Best fitness: {log['best_fitness_per_gen'][-1]}")
        logs.append(log)
    return logs


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def report_finals(name: str, logs: List[Dict[str, Any]]) -> List[float]:
    if not logs:
        print(f"{name}: no data")
        return []
    finals = [log["best_fitness_per_gen"][-1] for log in logs]
    arr = np.array(finals, dtype=float)
    print(f"{name}: {finals}")
    print(
        f"  mean={arr.mean():.2f}, std={arr.std(ddof=1):.2f}, "
        f"median={np.median(arr):.1f}, min={arr.min():.0f}, max={arr.max():.0f}"
    )
    return finals


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    ga_logs = run_or_load("ga", run_ga)
    cnn_logs = run_or_load("cnn_ga", run_cnn_ga)

    print_separator("Final statistics (10 seeds)")
    ga_finals = report_finals("GA    ", ga_logs)
    cnn_finals = report_finals("CNN-GA", cnn_logs)

    if ga_finals and cnn_finals:
        t_stat, p_val = ttest_rel(ga_finals, cnn_finals)
        gain = 100 * (np.mean(ga_finals) - np.mean(cnn_finals)) / np.mean(ga_finals)
        print(
            f"\nPaired t-test GA vs CNN-GA (n={len(ga_finals)}): "
            f"t={t_stat:.3f}, p={p_val:.4g}, df={len(ga_finals) - 1}, "
            f"gain={gain:+.2f}%"
        )

    print_separator("Building plots")
    gens = HARD_CASE["generations"]
    n_acc = CNN_PARAMS["n_accumulate"]

    plots.plot_convergence(
        ga_logs, cnn_logs, gens, f"{PLOTS_DIR}/convergence.png",
    )
    plots.plot_mean_fitness(
        ga_logs, cnn_logs, gens, n_acc, f"{PLOTS_DIR}/mean_fitness.png",
    )
    plots.plot_cnn_loss(cnn_logs, f"{PLOTS_DIR}/cnn_loss.png")
    plots.plot_surrogate_r2(cnn_logs, f"{PLOTS_DIR}/cnn_surrogate_r2.png")
    plots.plot_final_boxplot(
        ga_logs, cnn_logs, f"{PLOTS_DIR}/final_fitness_boxplot.png",
    )

    print("\nComparison experiment finished successfully.")


if __name__ == "__main__":
    main()
