"""
ICRAS 2026 revision: 10-seed runs + ablation study.

Runs four configurations on SEEDS_10:
  - GA              (baseline)
  - CNN-GA          (pre-screening + gradient-guided mutation)
  - CNN-GA-PreOnly  (pre-screening only)
  - CNN-GA-GradOnly (gradient-guided mutation only)

Caches logs in logs_10seeds/ (existing logs for the first 5 seeds of GA and
CNN-GA are copied over from logs/ to avoid re-running). Builds 10-seed
versions of the main plots and the ablation plots in plots_10seeds/.
"""

import json
import os
import shutil
from typing import Any, Callable, Dict, List

import matplotlib
matplotlib.use("Agg")

import numpy as np
from scipy.stats import ttest_rel

from algorithms import CNNGuidedGA, GeneticAlgorithm
from algorithms.mutations import random_relocate_mutation
from experiments import plots
from experiments.cases import CNN_PARAMS, HARD_CASE, M_SWAPS, SEEDS, SEEDS_10


LOGS_DIR = "logs_10seeds"
PLOTS_DIR = "plots_10seeds"
OLD_LOGS_DIR = "logs"


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


def run_cnn_ga_pre_only(seed: int) -> Dict[str, Any]:
    cnn = CNNGuidedGA(
        **HARD_CASE, **CNN_PARAMS,
        use_prescreening=True, use_grad_mutation=False,
        rng_seed=seed,
    )
    _, log = cnn.run(verbose=True)
    log["algo"] = "cnn_ga_pre_only"
    log["seed"] = seed
    return log


def run_cnn_ga_grad_only(seed: int) -> Dict[str, Any]:
    cnn = CNNGuidedGA(
        **HARD_CASE, **CNN_PARAMS,
        use_prescreening=False, use_grad_mutation=True,
        rng_seed=seed,
    )
    _, log = cnn.run(verbose=True)
    log["algo"] = "cnn_ga_grad_only"
    log["seed"] = seed
    return log


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def reuse_existing_logs() -> None:
    """Copy ga/cnn_ga logs for the original 5 seeds from logs/ to logs_10seeds/."""
    n_copied = 0
    for name in ("ga", "cnn_ga"):
        for seed in SEEDS:
            src = f"{OLD_LOGS_DIR}/run_{name}_{seed}.json"
            dst = f"{LOGS_DIR}/run_{name}_{seed}.json"
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                n_copied += 1
    if n_copied:
        print(f"Reused {n_copied} log file(s) from {OLD_LOGS_DIR}/ -> {LOGS_DIR}/")


def run_or_load(
    name: str, runner: Callable[[int], Dict[str, Any]], seeds: List[int],
) -> List[Dict[str, Any]]:
    """Run all seeds for an algorithm or load cached logs from disk."""
    logs: List[Dict[str, Any]] = []
    total = len(seeds)
    for i, seed in enumerate(seeds):
        log_path = f"{LOGS_DIR}/run_{name}_{seed}.json"
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
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    reuse_existing_logs()

    ga_logs = run_or_load("ga", run_ga, SEEDS_10)
    cnn_logs = run_or_load("cnn_ga", run_cnn_ga, SEEDS_10)
    pre_logs = run_or_load("cnn_ga_pre_only", run_cnn_ga_pre_only, SEEDS_10)
    grad_logs = run_or_load("cnn_ga_grad_only", run_cnn_ga_grad_only, SEEDS_10)

    print_separator("Final statistics (10 seeds)")
    bundles = [
        ("GA            ", ga_logs),
        ("CNN-GA        ", cnn_logs),
        ("CNN-GA-Pre    ", pre_logs),
        ("CNN-GA-Grad   ", grad_logs),
    ]
    finals_by_name: Dict[str, List[float]] = {}
    for name, logs in bundles:
        if not logs:
            print(f"{name}: no data")
            continue
        finals = [log["best_fitness_per_gen"][-1] for log in logs]
        finals_by_name[name.strip()] = finals
        print(f"{name}: {finals}")
        print(
            f"               mean={np.mean(finals):.2f}, "
            f"std={np.std(finals, ddof=1):.2f}, median={np.median(finals):.1f}"
        )

    if "GA" in finals_by_name and "CNN-GA" in finals_by_name:
        t_stat, p_val = ttest_rel(finals_by_name["GA"], finals_by_name["CNN-GA"])
        print(
            f"\nPaired t-test GA vs CNN-GA (n={len(finals_by_name['GA'])}): "
            f"t={t_stat:.3f}, p={p_val:.4f}"
        )

    print_separator("Building plots (10 seeds)")
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

    print_separator("Building ablation plots")
    ablation_series = [
        ("GA",              plots.COLOR_GA,        ga_logs),
        ("CNN-GA",          plots.COLOR_CNN,       cnn_logs),
        ("CNN-GA-PreOnly",  plots.COLOR_PRE_ONLY,  pre_logs),
        ("CNN-GA-GradOnly", plots.COLOR_GRAD_ONLY, grad_logs),
    ]
    plots.plot_convergence_multi(
        ablation_series, gens, f"{PLOTS_DIR}/convergence_ablation.png",
    )
    plots.plot_mean_fitness_multi(
        ablation_series, gens, n_acc, f"{PLOTS_DIR}/mean_fitness_ablation.png",
    )

    print("\nAblation experiment finished successfully.")


if __name__ == "__main__":
    main()
