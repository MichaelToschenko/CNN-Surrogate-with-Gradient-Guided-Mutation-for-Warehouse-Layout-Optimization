import json
import os
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")

import numpy as np

from algorithms import CNNGuidedGA, GeneticAlgorithm
from algorithms.mutations import random_relocate_mutation
from experiments import plots
from experiments.cases import CNN_PARAMS, HARD_CASE, M_SWAPS, SEEDS


N_RUNS = len(SEEDS)


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
# Runs
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


def run_or_load(name: str, runner, log_dir: str) -> List[Dict[str, Any]]:
    """Run N seeds or load logs from disk."""
    logs: List[Dict[str, Any]] = []
    for i, seed in enumerate(SEEDS):
        log_path = f"{log_dir}/run_{name}_{seed}.json"
        if os.path.exists(log_path):
            print_separator(
                f"{name.upper()} run {i + 1}/{N_RUNS} | seed={seed} [from disk]"
            )
            log = load_log(log_path)
            print(f"  Loaded: {log_path} (best={log['best_fitness_per_gen'][-1]})")
        else:
            print_separator(f"{name.upper()} run {i + 1}/{N_RUNS} | seed={seed}")
            log = runner(seed)
            save_log(log, log_path)
            print(f"  Log saved: {log_path}")
            print(f"  >> Best fitness: {log['best_fitness_per_gen'][-1]}")
        logs.append(log)
    return logs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    ga_logs = run_or_load("ga", run_ga, "logs")
    cnn_logs = run_or_load("cnn_ga", run_cnn_ga, "logs")

    print_separator("Final statistics")
    for name, logs in [("GA     ", ga_logs), ("CNN-GA ", cnn_logs)]:
        if not logs:
            print(f"{name}: no data")
            continue
        finals = [log["best_fitness_per_gen"][-1] for log in logs]
        print(f"{name}: {finals}")
        print(f"         mean={np.mean(finals):.1f}, std={np.std(finals):.1f}")

    print_separator("Building plots")
    gens = HARD_CASE["generations"]
    n_acc = CNN_PARAMS["n_accumulate"]

    plots.plot_convergence(ga_logs, cnn_logs, gens, "plots/convergence.png")
    plots.plot_mean_fitness(ga_logs, cnn_logs, gens, n_acc, "plots/mean_fitness.png")
    plots.plot_cnn_loss(cnn_logs, "plots/cnn_loss.png")
    plots.plot_surrogate_r2(cnn_logs, "plots/cnn_surrogate_r2.png")
    plots.plot_final_boxplot(ga_logs, cnn_logs, "plots/final_fitness_boxplot.png")

    print("\nExperiment finished successfully.")


if __name__ == "__main__":
    main()
