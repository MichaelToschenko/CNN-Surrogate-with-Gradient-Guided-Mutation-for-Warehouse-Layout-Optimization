"""All plotting functions for GA / CNN-GA experiments and visualization."""

from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from constants import ENTRY, EXIT, ROAD, SAVE, CONFIG_CELL_TO_IDX, CONFIG_CMAP, CONFIG_COLORS
from individual import Individual


COLOR_GA = "steelblue"
COLOR_CNN = "tomato"
COLOR_PRE_ONLY = "seagreen"
COLOR_GRAD_ONLY = "goldenrod"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_mean_ci(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and 95% CI by row. NaNs are ignored."""
    from scipy.stats import t as t_dist
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0, ddof=1)
    n_samples = np.sum(~np.isnan(data), axis=0).clip(min=2)
    t_crit = t_dist.ppf(0.975, df=n_samples - 1)
    ci = t_crit * std / np.sqrt(n_samples)
    return mean, ci


# ---------------------------------------------------------------------------
# Single-run plots
# ---------------------------------------------------------------------------

def plot_fitness_history(
    log: Dict, save_path: Optional[str] = None,
) -> None:
    """Best-overall and mean-population fitness curves over generations."""
    best = log["best_fitness_per_gen"]
    avg = log["mean_fitness_per_gen"]
    gens = list(range(len(best)))

    plt.figure(figsize=(9, 5))
    plt.plot(gens, avg, marker="o", label="Mean")
    plt.plot(gens, best, marker="s", label="Best (overall)")
    plt.title("Fitness over generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_metrics_history(
    log: Dict, save_path: Optional[str] = None,
) -> None:
    """Spatial-metric curves over generations."""
    gens = list(range(len(log["avg_ss_dist"])))
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 3, 1)
    plt.plot(gens, log["avg_ss_dist"], marker="o")
    plt.title("Average Save-Save distance")
    plt.xlabel("Generation")
    plt.ylabel("Manhattan distance")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(gens, log["avg_local_density"], marker="o")
    plt.title("Average objects within R=3 of Save")
    plt.xlabel("Generation")
    plt.ylabel("Object count")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(gens, log["avg_s_to_ex"], marker="o")
    plt.title("Average distance Save -> nearest Entry/Exit")
    plt.xlabel("Generation")
    plt.ylabel("Manhattan distance")
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Comparative plots (multiple runs with CI)
# ---------------------------------------------------------------------------

def _plot_with_ci(
    ax, gens, datasets: List[Tuple[str, str, List[Dict], str]],
) -> None:
    """Helper: plot series with CI bands on a single axis."""
    for label, color, logs, key in datasets:
        if not logs:
            continue
        data = np.array([log[key] for log in logs], dtype=float)
        mean, ci = compute_mean_ci(data)
        ax.plot(gens, mean, color=color, lw=2, label=label)
        ax.fill_between(gens, mean - ci, mean + ci, alpha=0.2, color=color)


def plot_convergence(
    ga_logs: List[Dict], cnn_logs: List[Dict],
    generations: int, save_path: str,
) -> None:
    """Convergence (best fitness) of the two algorithms with CI."""
    gens = list(range(generations + 1))
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_with_ci(ax, gens, [
        ("GA", COLOR_GA, ga_logs, "best_fitness_per_gen"),
        ("CNN-GA", COLOR_CNN, cnn_logs, "best_fitness_per_gen"),
    ])
    ax.set_xlabel("Generation", fontsize=13)
    ax.set_ylabel("Best Fitness (simulation steps)", fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_convergence_multi(
    series: List[Tuple[str, str, List[Dict]]],
    generations: int, save_path: str,
) -> None:
    """Convergence (best fitness) with CI for N algorithms. series: (label, color, logs)."""
    gens = list(range(generations + 1))
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_with_ci(
        ax, gens,
        [(label, color, logs, "best_fitness_per_gen") for label, color, logs in series],
    )
    ax.set_xlabel("Generation", fontsize=13)
    ax.set_ylabel("Best Fitness (simulation steps)", fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_mean_fitness(
    ga_logs: List[Dict], cnn_logs: List[Dict],
    generations: int, n_accumulate: int, save_path: str,
) -> None:
    """Mean population fitness over generations with CI."""
    gens = list(range(generations + 1))
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_with_ci(ax, gens, [
        ("GA", COLOR_GA, ga_logs, "mean_fitness_per_gen"),
        ("CNN-GA", COLOR_CNN, cnn_logs, "mean_fitness_per_gen"),
    ])
    ax.axvline(
        x=n_accumulate, color="gray", linestyle="--", lw=1.5, alpha=0.8,
        label=f"Accumulation end (gen. {n_accumulate})",
    )
    ax.set_xlabel("Generation", fontsize=13)
    ax.set_ylabel("Mean Population Fitness (simulation steps)", fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_mean_fitness_multi(
    series: List[Tuple[str, str, List[Dict]]],
    generations: int, n_accumulate: int, save_path: str,
) -> None:
    """Mean population fitness with CI for N algorithms. series: (label, color, logs)."""
    gens = list(range(generations + 1))
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_with_ci(
        ax, gens,
        [(label, color, logs, "mean_fitness_per_gen") for label, color, logs in series],
    )
    ax.axvline(
        x=n_accumulate, color="gray", linestyle="--", lw=1.5, alpha=0.8,
        label=f"Accumulation end (gen. {n_accumulate})",
    )
    ax.set_xlabel("Generation", fontsize=13)
    ax.set_ylabel("Mean Population Fitness (simulation steps)", fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_cnn_loss(cnn_logs: List[Dict], save_path: str) -> None:
    """CNN-GA surrogate loss (MSE) per epoch with CI."""
    all_losses = [log.get("cnn_loss_per_epoch", []) for log in cnn_logs]
    max_epochs = max((len(l) for l in all_losses), default=0)
    if max_epochs == 0:
        print("No CNN loss data - skipping plot.")
        return

    loss_arr = np.full((len(cnn_logs), max_epochs), np.nan)
    for i, losses in enumerate(all_losses):
        loss_arr[i, : len(losses)] = losses

    epochs = list(range(1, max_epochs + 1))
    mean, ci = compute_mean_ci(loss_arr)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, mean, color=COLOR_CNN, lw=2, label="CNN-GA loss (MSE)")
    floor = max(np.nanmin(mean) * 0.5, 1e-6)
    lower = np.clip(mean - ci, a_min=floor, a_max=None)
    ax.fill_between(epochs, lower, mean + ci, alpha=0.25, color=COLOR_CNN)
    ax.set_yscale("log")
    ax.set_ylim(bottom=floor)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Loss (MSE, normalized, log scale)", fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_surrogate_r2(cnn_logs: List[Dict], save_path: str) -> None:
    """CNN-GA surrogate R^2 per training / fine-tuning."""
    all_r2 = [log.get("surrogate_r2_per_train", []) for log in cnn_logs]
    max_len = max((len(r) for r in all_r2), default=0)
    if max_len == 0:
        print("No CNN-GA surrogate R^2 data - skipping plot.")
        return

    r2_arr = np.full((len(cnn_logs), max_len), np.nan)
    for i, r2s in enumerate(all_r2):
        r2_arr[i, : len(r2s)] = r2s

    trains = list(range(1, max_len + 1))
    mean, ci = compute_mean_ci(r2_arr)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(trains, mean, color=COLOR_CNN, lw=2, marker="o", label="R^2")
    ax.fill_between(trains, mean - ci, mean + ci, alpha=0.25, color=COLOR_CNN)
    ax.set_xlabel("Training / Fine-tuning Index", fontsize=13)
    ax.set_ylabel("R^2", fontsize=13)
    ax.set_ylim(-0.1, 1.0)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_final_boxplot(
    ga_logs: List[Dict], cnn_logs: List[Dict], save_path: str,
) -> None:
    """Boxplot of final fitness for both algorithms with individual points."""
    all_data, labels, colors = [], [], []
    for name, color, logs in [
        ("GA", COLOR_GA, ga_logs),
        ("CNN-GA", COLOR_CNN, cnn_logs),
    ]:
        if not logs:
            continue
        all_data.append([log["best_fitness_per_gen"][-1] for log in logs])
        labels.append(name)
        colors.append(color)

    if not all_data:
        print("No data for boxplot - skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(all_data, labels=labels, patch_artist=True, widths=0.4)
    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c)
        box.set_alpha(0.4)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    for x_pos, (vals, c) in enumerate(zip(all_data, colors), start=1):
        ax.scatter(
            np.full(len(vals), x_pos), vals,
            zorder=5, color=c, edgecolors="black", s=60, alpha=0.9,
        )

    ax.set_ylabel("Final Fitness (simulation steps)", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Best-configuration evolution visualization
# ---------------------------------------------------------------------------

def plot_evolution(
    snapshots: List[Individual],
    gen_numbers: List[int],
    save_path: str,
    title: str = "CNN-GA: Evolution of Best Warehouse Layout",
    n_rows: int = 1,
) -> None:
    n_panels = len(snapshots)
    n_cols = (n_panels + n_rows - 1) // n_rows

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.8 * n_cols, 4.5 * n_rows),
    )
    if n_panels == 1:
        axes_flat = [axes]
    else:
        axes_flat = np.atleast_1d(axes).flatten().tolist()

    for ax, ind, gen in zip(axes_flat, snapshots, gen_numbers):
        numeric = np.vectorize(CONFIG_CELL_TO_IDX.get)(ind.grid)
        ax.pcolormesh(
            numeric, cmap=CONFIG_CMAP, vmin=0, vmax=3,
            edgecolors="#cccccc", linewidth=0.3,
        )
        ax.set_xlim(0, ind.n)
        ax.set_ylim(ind.m, 0)
        ax.set_aspect("equal")
        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"Generation {gen}", fontsize=12, pad=6)
        ax.set_xlabel(f"Fitness: {ind.fitness}", fontsize=10, labelpad=4)

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    legend_patches = [
        mpatches.Patch(color=CONFIG_COLORS[ROAD], label="Road"),
        mpatches.Patch(color=CONFIG_COLORS[ENTRY], label="Entry"),
        mpatches.Patch(color=CONFIG_COLORS[EXIT], label="Exit"),
        mpatches.Patch(color=CONFIG_COLORS[SAVE], label="Storage"),
    ]
    fig.legend(
        handles=legend_patches, loc="lower center",
        ncol=4, fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(title, fontsize=13, y=1.0)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08 if n_rows > 1 else 0.12)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
