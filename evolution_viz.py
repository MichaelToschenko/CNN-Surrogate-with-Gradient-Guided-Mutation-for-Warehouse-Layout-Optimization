"""
Визуализация эволюции лучшей конфигурации склада — CNN-GA.

Кейс 15×15, параметры как в comparison_experiment.
20 поколений, снэпшот каждые 5 → gen 0, 5, 10, 15, 20.

Результат: plots/evolution_cnn_ga.png

Запуск:
    python evolution_viz.py

При повторном запуске загружает снэпшоты из logs/ (пересчёт не нужен).
"""

import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from constants import (
    ENTRY, EXIT, ROAD, SAVE,
    CONFIG_CELL_TO_IDX, CONFIG_CMAP, CONFIG_COLORS,
)
from genetic_algorithm_cnn import CNNGuidedGA
from individual import Individual


# ===========================================================================
# Параметры
# ===========================================================================

SNAPSHOT_INTERVAL = 5
GENERATIONS       = 20
SEED              = 42

VIZ_CASE = dict(
    m=15, n=15, entries=5, exits=5, saves=30,
    num_robots=12, containers_to_process=160, prob=0.35,
    pop_size=50, generations=GENERATIONS, pm=0.3,
    k_tournament=3, elitism_count=3,
)


# ===========================================================================
# CNN-GA со сбором снэпшотов
# ===========================================================================

class SnapshotCNNGA(CNNGuidedGA):
    """CNNGuidedGA, сохраняющий лучшую особь каждые SNAPSHOT_INTERVAL поколений."""

    def run(self, verbose: bool = True):
        """Возвращает (best_overall, snapshots, gen_numbers)."""
        population = self._init_population()
        snapshots:   List[Individual] = []
        gen_numbers: List[int]        = []

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            if verbose:
                print("CNN-GA: оценка начальной популяции...")
            self._evaluate_batch(population, executor)
            self._record_to_buffer(population)

            if self.warmstart_extra > 0:
                if verbose:
                    print(f"  Warm-start: {self.warmstart_extra} доп. конфигураций...")
                warmstart = [
                    Individual.random_individual(
                        self.m, self.n, self.entries, self.exits, self.saves,
                        rng=self.rng,
                    )
                    for _ in range(self.warmstart_extra)
                ]
                self._evaluate_batch(warmstart, executor)
                self._record_to_buffer(warmstart)

            best_overall = min(population, key=lambda ind: ind.fitness)
            snapshots.append(best_overall.copy())
            gen_numbers.append(0)
            if verbose:
                print(f"  Gen 0: best={best_overall.fitness}, buf={len(self.buffer)}")

            for gen in range(1, self.generations + 1):
                in_accumulation = gen <= self.n_accumulate
                use_surrogate   = not in_accumulation and self.surrogate_trained

                sorted_pop = sorted(population, key=lambda ind: ind.fitness)
                elites   = [ind.copy() for ind in sorted_pop[:self.elitism_count]]
                n_needed = self.pop_size - self.elitism_count
                n_generate = n_needed * self.oversample_factor if use_surrogate else n_needed

                all_children: List[Individual] = []
                for _ in range(n_generate):
                    p1 = self._tournament_select(population)
                    p2 = self._tournament_select(population)
                    child = self._crossover(p1, p2)
                    if self.rng.random() < self.pm:
                        self._mutate_smart(child)
                    all_children.append(child)

                if use_surrogate and len(all_children) > n_needed:
                    predicted = self._surrogate_predict_batch(all_children)
                    top_idx   = np.argsort(predicted)[:n_needed]
                    children  = [all_children[i] for i in top_idx]
                else:
                    children = all_children

                self._evaluate_batch(children, executor)
                self._record_to_buffer(children)
                population = elites + children

                best_in_gen = min(population, key=lambda ind: ind.fitness)
                if best_in_gen.fitness < best_overall.fitness:
                    best_overall = best_in_gen.copy()

                if gen % SNAPSHOT_INTERVAL == 0:
                    snapshots.append(best_overall.copy())
                    gen_numbers.append(gen)

                # Обучение / дообучение суррогата
                if gen == self.n_accumulate:
                    if verbose:
                        print(f"  [Обучение суррогата] buffer={len(self.buffer)}...")
                    losses, r2 = self._train_surrogate()
                    if losses:
                        if r2 >= self.min_r2:
                            self.surrogate_trained = True
                        if verbose:
                            print(f"  Суррогат: R²={r2:.3f}"
                                  f"  {'(активен)' if self.surrogate_trained else '(слабый)'}")
                elif (not in_accumulation
                      and (gen - self.n_accumulate) % self.retrain_interval == 0):
                    losses, r2 = self._train_surrogate()
                    if losses:
                        if r2 >= self.min_r2 and not self.surrogate_trained:
                            self.surrogate_trained = True
                        if verbose:
                            print(f"  Дообучение gen {gen}: R²={r2:.3f}")

                if verbose:
                    phase = "surr" if use_surrogate else "rand"
                    print(f"  gen {gen:3d}: best={best_overall.fitness} [{phase}]")

        return best_overall, snapshots, gen_numbers


# ===========================================================================
# Сериализация
# ===========================================================================

def save_snapshots(snapshots: List[Individual],
                   gen_numbers: List[int], path: str) -> None:
    data = [
        {"gen": g, "fitness": ind.fitness, "grid": ind.grid.tolist()}
        for g, ind in zip(gen_numbers, snapshots)
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_snapshots(path: str) -> Tuple[List[Individual], List[int]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    inds = [
        Individual(np.array(d["grid"], dtype=object), d["fitness"])
        for d in data
    ]
    return inds, [d["gen"] for d in data]


# ===========================================================================
# Визуализация
# ===========================================================================

def plot_evolution(
    snapshots:   List[Individual],
    gen_numbers: List[int],
    save_path:   str,
) -> None:
    n_cols = len(snapshots)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.8 * n_cols, 4.5))
    if n_cols == 1:
        axes = [axes]

    for ax, ind, gen in zip(axes, snapshots, gen_numbers):
        numeric = np.vectorize(CONFIG_CELL_TO_IDX.get)(ind.grid)
        ax.pcolormesh(
            numeric, cmap=CONFIG_CMAP, vmin=0, vmax=3,
            edgecolors="#cccccc", linewidth=0.3,
        )
        ax.set_xlim(0, ind.n)
        ax.set_ylim(ind.m, 0)
        ax.set_aspect("equal")
        ax.tick_params(bottom=False, left=False,
                       labelbottom=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"Generation {gen}", fontsize=12, pad=6)
        ax.set_xlabel(f"Fitness: {ind.fitness}", fontsize=10, labelpad=4)

    legend_patches = [
        mpatches.Patch(color=CONFIG_COLORS[ROAD],  label="Road"),
        mpatches.Patch(color=CONFIG_COLORS[ENTRY], label="Entry"),
        mpatches.Patch(color=CONFIG_COLORS[EXIT],  label="Exit"),
        mpatches.Patch(color=CONFIG_COLORS[SAVE],  label="Storage"),
    ]
    fig.legend(
        handles=legend_patches, loc="lower center",
        ncol=4, fontsize=10, frameon=True,
        bbox_to_anchor=(0.5, -0.08),
    )

    fig.suptitle(
        "CNN-GA: Evolution of Best Warehouse Layout  (15×15)",
        fontsize=13, y=1.02,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Сохранён: {save_path}")


# ===========================================================================
# Точка входа
# ===========================================================================

def main() -> None:
    os.makedirs("logs",  exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    log_path = "logs/evolution_cnn_ga.json"

    if os.path.exists(log_path):
        snapshots, gen_numbers = load_snapshots(log_path)
        print(f"Загружено {len(snapshots)} снэпшотов из {log_path}")
    else:
        print("=== CNN-GA прогон ===")
        cnn = SnapshotCNNGA(
            **VIZ_CASE, K_swaps=4, m_swaps=4,
            n_accumulate=5, retrain_interval=3, n_train_epochs=30,
            max_buffer_size=10_000, oversample_factor=3,
            warmstart_extra=50, rng_seed=SEED,
        )
        _, snapshots, gen_numbers = cnn.run(verbose=True)
        save_snapshots(snapshots, gen_numbers, log_path)
        print(f"Логи сохранены: {log_path}")

    plot_evolution(snapshots, gen_numbers, "plots/evolution_cnn_ga.png")
    print("Готово.")


if __name__ == "__main__":
    main()
