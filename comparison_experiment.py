"""
Честное сравнение GA vs CNN-GA на сложном кейсе 15×15.

Запуск:
    python comparison_experiment.py

Результаты:
    logs/run_ga_<seed>.json       — логи каждого прогона GA
    logs/run_cnn_ga_<seed>.json   — логи каждого прогона CNN-GA
    plots/convergence.png
    plots/mutation_success_rate.png
    plots/cnn_loss.png
    plots/final_fitness_boxplot.png
"""

import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

# Headless backend — обязательно ДО импорта pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

# Локальные импорты
from constants import ROAD
from genetic_algorithm import GeneticAlgorithm
from genetic_algorithm_cnn import CNNGuidedGA
from individual import Individual
from simulation import evaluate_configuration


# ===========================================================================
# Параметры эксперимента
# ===========================================================================

M_SWAPS = 4        # число свапов: K_swaps GA = m_swaps CNN-GA (честное сравнение)
N_ACCUMULATE = 20  # поколений для накопления данных в CNN-GA

HARD_CASE: Dict[str, Any] = dict(
    m=15, n=15,
    entries=5, exits=5, saves=30,
    num_robots=12,
    containers_to_process=160,
    prob=0.35,
    pop_size=50,
    generations=80,
    pm=0.1,
    k_tournament = 5,
    elitism_count = 5,
)

N_RUNS = 5
SEEDS = [42, 123, 456, 789, 1337]


# ===========================================================================
# TrackingGA — GeneticAlgorithm + отслеживание доли успешных мутаций
# ===========================================================================

class TrackingGA(GeneticAlgorithm):
    """
    GeneticAlgorithm с отслеживанием доли успешных мутаций.

    Чтобы определить успешность мутации, каждый мутируемый ребёнок
    оценивается ДО и ПОСЛЕ мутации. Это не влияет на решения алгоритма
    (только на логирование), поэтому сравнение с CNN-GA остаётся честным.
    """

    def run(self, verbose: bool = True) -> Tuple[Individual, Dict[str, Any]]:
        """
        Возвращает (best_individual, log_dict) — формат совместим с CNNGuidedGA.
        """
        population = self._init_population()

        best_fitness_per_gen: List[int] = []
        mean_fitness_per_gen: List[float] = []
        successful_mutation_rate_per_gen: List[Optional[float]] = []

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:

            # ----------------------------------------------------------------
            # Поколение 0
            # ----------------------------------------------------------------
            if verbose:
                print("GA: оценка начальной популяции...")
            self._evaluate_batch(population, executor)

            best_overall = min(population, key=lambda ind: ind.fitness)
            avg_f0 = sum(ind.fitness for ind in population) / len(population)

            best_fitness_per_gen.append(best_overall.fitness)
            mean_fitness_per_gen.append(avg_f0)
            successful_mutation_rate_per_gen.append(None)

            if verbose:
                print(f"  Поколение 0: best={best_overall.fitness}, avg={avg_f0:.1f}")

            # ----------------------------------------------------------------
            # Основной цикл
            # ----------------------------------------------------------------
            for gen in range(1, self.generations + 1):
                sorted_pop = sorted(population, key=lambda ind: ind.fitness)
                elites = [ind.copy() for ind in sorted_pop[: self.elitism_count]]

                children: List[Individual] = []
                to_mutate_idx: List[int] = []

                while len(elites) + len(children) < self.pop_size:
                    p1 = self._tournament_select(population)
                    p2 = self._tournament_select(population)
                    child = self._crossover(p1, p2)
                    if self.rng.random() < self.pm:
                        to_mutate_idx.append(len(children))
                    children.append(child)

                # Разделяем детей на мутируемых и нет
                to_mutate_set = set(to_mutate_idx)
                mutated   = [children[i] for i in to_mutate_idx]
                untouched = [children[i] for i in range(len(children))
                             if i not in to_mutate_set]

                self._evaluate_batch(untouched, executor)
                self._evaluate_batch(mutated, executor)

                # Per-swap цикл: один своп → оценка → метка
                n_successful = 0
                n_swap_records = 0

                for _ in range(self.K_swaps):
                    pre_fitnesses = [c.fitness for c in mutated]

                    for child in mutated:
                        coords = [tuple(p)
                                  for p in np.argwhere(child.grid != ROAD)]
                        if len(coords) < 2:
                            continue
                        pos_a = coords[self.rng.randrange(len(coords))]
                        type_a = child.grid[pos_a]
                        candidates = [c for c in coords
                                      if child.grid[c] != type_a]
                        if not candidates:
                            continue
                        pos_b = candidates[self.rng.randrange(len(candidates))]
                        child.grid[pos_a], child.grid[pos_b] = (
                            child.grid[pos_b], child.grid[pos_a],
                        )

                    self._evaluate_batch(mutated, executor)

                    for child, pre_fit in zip(mutated, pre_fitnesses):
                        n_successful += 1 if child.fitness < pre_fit else 0
                        n_swap_records += 1

                mut_rate: Optional[float] = (
                    n_successful / n_swap_records if n_swap_records > 0 else None
                )
                successful_mutation_rate_per_gen.append(mut_rate)

                population = elites + children

                best_in_gen = min(population, key=lambda ind: ind.fitness)
                if best_in_gen.fitness < best_overall.fitness:
                    best_overall = best_in_gen.copy()

                avg_fitness = sum(ind.fitness for ind in population) / len(population)
                best_fitness_per_gen.append(best_overall.fitness)
                mean_fitness_per_gen.append(avg_fitness)

                if verbose:
                    mut_str = (
                        f"mut_ok={mut_rate:.1%}" if mut_rate is not None else "no mut"
                    )
                    print(
                        f"  GA gen {gen:3d}: "
                        f"best_overall={best_overall.fitness:7d}, "
                        f"avg={avg_fitness:8.1f}, "
                        f"{mut_str}"
                    )

        log: Dict[str, Any] = {
            "best_fitness_per_gen": best_fitness_per_gen,
            "mean_fitness_per_gen": mean_fitness_per_gen,
            "successful_mutation_rate_per_gen": successful_mutation_rate_per_gen,
            "cnn_loss_per_epoch": [],
        }
        return best_overall, log


# ===========================================================================
# Статистика и CI
# ===========================================================================

def compute_mean_ci(
    data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    mean и 95% CI по строкам (каждая строка — один прогон).
    nan-значения игнорируются.
    """
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0, ddof=1)
    ci = 1.96 * std / np.sqrt(np.sum(~np.isnan(data), axis=0).clip(min=1))
    return mean, ci


def rates_to_array(
    logs: List[Dict], key: str, skip_first: int = 1
) -> np.ndarray:
    """Конвертирует список логов в 2D np.ndarray (runs × gens), None → nan."""
    rows = []
    for log in logs:
        row = [
            x if x is not None else np.nan
            for x in log[key][skip_first:]
        ]
        rows.append(row)
    return np.array(rows, dtype=float)


# ===========================================================================
# Графики
# ===========================================================================

def plot_convergence(
    ga_logs: List[Dict],
    cnn_logs: List[Dict],
    generations: int,
    save_path: str,
) -> None:
    """Сходимость (лучший fitness) обоих алгоритмов с CI."""
    gens = list(range(generations + 1))

    ga_data = np.array([log["best_fitness_per_gen"] for log in ga_logs], dtype=float)
    cnn_data = np.array([log["best_fitness_per_gen"] for log in cnn_logs], dtype=float)

    ga_mean, ga_ci = compute_mean_ci(ga_data)
    cnn_mean, cnn_ci = compute_mean_ci(cnn_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gens, ga_mean, color="steelblue", lw=2, label="GA")
    ax.fill_between(
        gens, ga_mean - ga_ci, ga_mean + ga_ci, alpha=0.25, color="steelblue"
    )
    ax.plot(gens, cnn_mean, color="tomato", lw=2, label="CNN-GA")
    ax.fill_between(
        gens, cnn_mean - cnn_ci, cnn_mean + cnn_ci, alpha=0.25, color="tomato"
    )

    ax.set_xlabel("Поколение", fontsize=13)
    ax.set_ylabel("Лучший fitness (шагов симуляции)", fontsize=13)
    ax.set_title(
        f"Сходимость GA vs CNN-GA\n"
        f"15×15, K_swaps={M_SWAPS}, pm={HARD_CASE['pm']}, "
        f"mean ± 95% CI ({N_RUNS} прогонов)",
        fontsize=13,
    )
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Сохранён: {save_path}")


def plot_mutation_success_rate(
    ga_logs: List[Dict],
    cnn_logs: List[Dict],
    generations: int,
    n_accumulate: int,
    save_path: str,
) -> None:
    """Доля успешных мутаций с CI для обоих алгоритмов."""
    gens = list(range(1, generations + 1))

    ga_data = rates_to_array(ga_logs, "successful_mutation_rate_per_gen")
    cnn_data = rates_to_array(cnn_logs, "successful_mutation_rate_per_gen")

    ga_mean, ga_ci = compute_mean_ci(ga_data)
    cnn_mean, cnn_ci = compute_mean_ci(cnn_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gens, ga_mean, color="steelblue", lw=2, label="GA")
    ax.fill_between(
        gens, ga_mean - ga_ci, ga_mean + ga_ci, alpha=0.25, color="steelblue"
    )
    ax.plot(gens, cnn_mean, color="tomato", lw=2, label="CNN-GA")
    ax.fill_between(
        gens, cnn_mean - cnn_ci, cnn_mean + cnn_ci, alpha=0.25, color="tomato"
    )
    ax.axvline(
        x=n_accumulate,
        color="gray",
        linestyle="--",
        lw=1.5,
        alpha=0.8,
        label=f"Конец накопления (ген. {n_accumulate})",
    )

    ax.set_xlabel("Поколение", fontsize=13)
    ax.set_ylabel("Доля успешных мутаций", fontsize=13)
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Доля успешных мутаций: GA vs CNN-GA\nmean ± 95% CI ({N_RUNS} прогонов)",
        fontsize=13,
    )
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Сохранён: {save_path}")


def plot_cnn_loss(
    cnn_logs: List[Dict],
    save_path: str,
) -> None:
    """Loss CNN по эпохам с CI."""
    all_losses = [log["cnn_loss_per_epoch"] for log in cnn_logs]

    if not any(all_losses):
        print("Нет данных о loss CNN — пропуск графика.")
        return

    max_epochs = max((len(l) for l in all_losses), default=0)
    if max_epochs == 0:
        print("Нет данных о loss CNN — пропуск графика.")
        return

    loss_arr = np.full((len(cnn_logs), max_epochs), np.nan)
    for i, losses in enumerate(all_losses):
        loss_arr[i, : len(losses)] = losses

    epochs = list(range(1, max_epochs + 1))
    mean, ci = compute_mean_ci(loss_arr)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, mean, color="seagreen", lw=2, label="CNN loss")
    ax.fill_between(
        epochs, mean - ci, mean + ci, alpha=0.25, color="seagreen"
    )

    ax.set_xlabel("Эпоха", fontsize=13)
    ax.set_ylabel("Loss (BCE × 2)", fontsize=13)
    ax.set_title(
        f"Loss CNN по эпохам\nmean ± 95% CI ({N_RUNS} прогонов)",
        fontsize=13,
    )
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Сохранён: {save_path}")


def plot_final_boxplot(
    ga_logs: List[Dict],
    cnn_logs: List[Dict],
    save_path: str,
) -> None:
    """Боксплот финального fitness с индивидуальными точками."""
    ga_finals = [log["best_fitness_per_gen"][-1] for log in ga_logs]
    cnn_finals = [log["best_fitness_per_gen"][-1] for log in cnn_logs]

    fig, ax = plt.subplots(figsize=(7, 6))
    bp = ax.boxplot(
        [ga_finals, cnn_finals],
        labels=["GA", "CNN-GA"],
        patch_artist=True,
        widths=0.4,
    )
    bp["boxes"][0].set_facecolor("#AED6F1")
    bp["boxes"][1].set_facecolor("#F1948A")
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    # Индивидуальные точки
    for x, vals in [(1, ga_finals), (2, cnn_finals)]:
        jitter = np.zeros(len(vals))
        ax.scatter(
            np.full(len(vals), x) + jitter,
            vals,
            zorder=5,
            color="black",
            s=50,
            alpha=0.8,
        )

    ax.set_ylabel("Финальный fitness (шагов симуляции)", fontsize=13)
    ax.set_title(
        f"Финальный fitness: GA vs CNN-GA ({N_RUNS} прогонов)", fontsize=13
    )
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Сохранён: {save_path}")


# ===========================================================================
# Вспомогательные функции
# ===========================================================================

def save_log(log: Dict[str, Any], path: str) -> None:
    """Сохранить лог в JSON. None → null, float('nan') не допускается."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def load_log(path: str) -> Dict[str, Any]:
    """Загрузить лог из JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_separator(title: str) -> None:
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")


# ===========================================================================
# Точка входа
# ===========================================================================

def main() -> None:
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    ga_logs: List[Dict] = []
    cnn_logs: List[Dict] = []

    # -----------------------------------------------------------------------
    # Прогоны GA (загружаем с диска, если уже запущены)
    # -----------------------------------------------------------------------
    for run_idx, seed in enumerate(SEEDS):
        log_path = f"logs/run_ga_{seed}.json"
        if os.path.exists(log_path):
            print_separator(f"GA прогон {run_idx + 1}/{N_RUNS}  |  seed={seed}  [загружен с диска]")
            log = load_log(log_path)
            print(f"  Загружен: {log_path}  (best={log['best_fitness_per_gen'][-1]})")
        else:
            print_separator(f"GA прогон {run_idx + 1}/{N_RUNS}  |  seed={seed}")
            ga = TrackingGA(**HARD_CASE, K_swaps=M_SWAPS, rng_seed=seed)
            best, log = ga.run(verbose=True)
            log["algo"] = "ga"
            log["seed"] = seed
            save_log(log, log_path)
            print(f"Лог сохранён: {log_path}")
            print(f">> GA прогон {run_idx + 1} завершён: best fitness = {best.fitness}")

        ga_logs.append(log)

    # -----------------------------------------------------------------------
    # Прогоны CNN-GA
    # -----------------------------------------------------------------------
    for run_idx, seed in enumerate(SEEDS):
        print_separator(f"CNN-GA прогон {run_idx + 1}/{N_RUNS}  |  seed={seed}")

        cnn_ga = CNNGuidedGA(
            **HARD_CASE,
            K_swaps=M_SWAPS,
            m_swaps=M_SWAPS,
            n_accumulate=N_ACCUMULATE,
            retrain_interval=10,
            n_train_epochs=30,
            max_buffer_size=1000,
            rng_seed=seed,
        )
        best, log = cnn_ga.run(verbose=True)
        log["algo"] = "cnn_ga"
        log["seed"] = seed

        log_path = f"logs/run_cnn_ga_{seed}.json"
        save_log(log, log_path)
        print(f"Лог сохранён: {log_path}")
        print(f">> CNN-GA прогон {run_idx + 1} завершён: best fitness = {best.fitness}")

        cnn_logs.append(log)

    # -----------------------------------------------------------------------
    # Итоговая статистика в терминале
    # -----------------------------------------------------------------------
    print_separator("Итоговая статистика")
    ga_finals = [log["best_fitness_per_gen"][-1] for log in ga_logs]
    cnn_finals = [log["best_fitness_per_gen"][-1] for log in cnn_logs]

    print(f"GA      : {ga_finals}")
    print(f"         mean={np.mean(ga_finals):.1f}, std={np.std(ga_finals):.1f}")
    print(f"CNN-GA  : {cnn_finals}")
    print(f"         mean={np.mean(cnn_finals):.1f}, std={np.std(cnn_finals):.1f}")

    # -----------------------------------------------------------------------
    # Графики
    # -----------------------------------------------------------------------
    print_separator("Построение графиков")
    gens = HARD_CASE["generations"]

    plot_convergence(ga_logs, cnn_logs, gens, "plots/convergence.png")
    plot_mutation_success_rate(
        ga_logs, cnn_logs, gens, N_ACCUMULATE, "plots/mutation_success_rate.png"
    )
    plot_cnn_loss(cnn_logs, "plots/cnn_loss.png")
    plot_final_boxplot(ga_logs, cnn_logs, "plots/final_fitness_boxplot.png")

    print("\nЭксперимент завершён успешно.")


if __name__ == "__main__":
    main()
