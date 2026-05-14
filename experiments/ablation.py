"""
Ablation study extending comparison.py with two CNN-GA modifications:
  - CNN-GA-PreOnly  (pre-screening only, random relocation mutation)
  - CNN-GA-GradOnly (gradient-guided mutation only, no oversampling)

Reuses GA and CNN-GA logs produced by experiments.comparison (running this
script directly without prior comparison.py will trigger those runs first via
run_or_load). Writes only the two ablation-specific plots; the main figures
are produced by comparison.py.
"""

import os
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")

import numpy as np
from scipy.stats import ttest_rel

from algorithms import CNNGuidedGA
from experiments import plots
from experiments.cases import CNN_PARAMS, HARD_CASE
from experiments.comparison import (
    LOGS_DIR,
    PLOTS_DIR,
    print_separator,
    report_finals,
    run_cnn_ga,
    run_ga,
    run_or_load,
)


# ---------------------------------------------------------------------------
# Ablation runners
# ---------------------------------------------------------------------------

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
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    ga_logs = run_or_load("ga", run_ga)
    cnn_logs = run_or_load("cnn_ga", run_cnn_ga)
    pre_logs = run_or_load("cnn_ga_pre_only", run_cnn_ga_pre_only)
    grad_logs = run_or_load("cnn_ga_grad_only", run_cnn_ga_grad_only)

    print_separator("Final statistics (10 seeds, all configurations)")
    ga_finals = report_finals("GA              ", ga_logs)
    cnn_finals = report_finals("CNN-GA          ", cnn_logs)
    pre_finals = report_finals("CNN-GA-PreOnly  ", pre_logs)
    grad_finals = report_finals("CNN-GA-GradOnly ", grad_logs)

    print()
    if ga_finals:
        for name, other in [
            ("CNN-GA         ", cnn_finals),
            ("CNN-GA-PreOnly ", pre_finals),
            ("CNN-GA-GradOnly", grad_finals),
        ]:
            if not other:
                continue
            t, p = ttest_rel(ga_finals, other)
            diff = np.mean(ga_finals) - np.mean(other)
            print(
                f"Paired t-test GA vs {name}: "
                f"t={t:>7.3f}, p={p:.4g}, mean diff (GA - X) = {diff:+.2f}"
            )

    if cnn_finals and grad_finals:
        t, p = ttest_rel(cnn_finals, grad_finals)
        print(
            f"Paired t-test CNN-GA vs CNN-GA-GradOnly: "
            f"t={t:>7.3f}, p={p:.4g}"
        )

    print_separator("Building ablation plots")
    gens = HARD_CASE["generations"]
    n_acc = CNN_PARAMS["n_accumulate"]
    series = [
        ("GA",              plots.COLOR_GA,        ga_logs),
        ("CNN-GA",          plots.COLOR_CNN,       cnn_logs),
        ("CNN-GA-PreOnly",  plots.COLOR_PRE_ONLY,  pre_logs),
        ("CNN-GA-GradOnly", plots.COLOR_GRAD_ONLY, grad_logs),
    ]
    plots.plot_convergence_multi(
        series, gens, f"{PLOTS_DIR}/convergence_ablation.png",
    )
    plots.plot_mean_fitness_multi(
        series, gens, n_acc, f"{PLOTS_DIR}/mean_fitness_ablation.png",
    )

    print("\nAblation experiment finished successfully.")


if __name__ == "__main__":
    main()
