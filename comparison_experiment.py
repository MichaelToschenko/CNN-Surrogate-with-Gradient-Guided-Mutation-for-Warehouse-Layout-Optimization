"""
Честное сравнение GA vs CNN-GA на сложном кейсе 15×15.

Оба алгоритма используют один оператор мутации — relocation
(перемещение не-дорожной ячейки на позицию ROAD).
Разница — только в CNN-guidance:
  - GA:     случайный выбор src и dst
  - CNN-GA: градиент суррогата для выбора + oversampling с фильтрацией

Запуск:
    python comparison_experiment.py

Результаты:
    logs/run_ga_<seed>.json        — логи каждого прогона GA
    logs/run_cnn_ga_<seed>.json    — логи каждого прогона CNN-GA
    plots/convergence.png
    plots/mutation_success_rate.png
    plots/cnn_loss.png
    plots/cnn_surrogate_r2.png
    plots/final_fitness_boxplot.png
"""

import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Tuple

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


# ===========================================================================
# Параметры эксперимента
# ===========================================================================

M_SWAPS = 4        # число relocations: K_swaps GA = m_swaps CNN-GA

HARD_CASE: Dict[str, Any] = dict(
    m=15, n=15,
    entries=5, exits=5, saves=30,
    num_robots=12,
    containers_to_process=160,
    prob=0.35,
    pop_size=50,
    generations=80,
    pm=0.3,
    k_tournament=3,
    elitism_count=3,
)

N_RUNS = 5
SEEDS = [42, 123, 456, 789, 1337]

# Цвета алгоритмов
COLOR_GA = "steelblue"
COLOR_CNN = "tomato"


# ===========================================================================
# TrackingGA — GeneticAlgorithm + отслеживание доли успешных мутаций
# ===========================================================================

class TrackingGA(GeneticAlgorithm):
    """
    GeneticAlgorithm с отслеживанием доли успешных мутаций.

    Мутация — случайный relocation: перемещение случайной не-дорожной ячейки
    на случайную позицию ROAD (тот же оператор, что и CNN-GA, но без
    градиентного направления). Это обеспечивает честное сравнение:
    оба алгоритма работают в одном пространстве мутаций (~7 400 ходов).

    Чтобы определить успешность мутации, каждый мутируемый ребёнок
    оценивается ДО и ПОСЛЕ мутации. Это не влияет на решения алгоритма
    (только на логирование), поэтому сравнение остаётся честным.
    """

    @staticmethod
    def _random_relocate(grid: np.ndarray, rng: random.Random) -> bool:
        """Один случайный relocation: не-дорожная ячейка → позиция ROAD.
        Возвращает True, если relocation был выполнен."""
        non_road = [tuple(p) for p in np.argwhere(grid != ROAD)]
        road_pos = [tuple(p) for p in np.argwhere(grid == ROAD)]
        if not non_road or not road_pos:
            return False
        src = non_road[rng.randrange(len(non_road))]
        dst = road_pos[rng.randrange(len(road_pos))]
        grid[dst] = grid[src]
        grid[src] = ROAD
        return True

    def run(self, verbose: bool = True) -> Tuple[Individual, Dict[str, Any]]:
        population = self._init_population()

        best_fitness_per_gen: List[int] = []
        mean_fitness_per_gen: List[float] = []

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:

            # Поколение 0
            if verbose:
                print("GA: оценка начальной популяции...")
            self._evaluate_batch(population, executor)

            best_overall = min(population, key=lambda ind: ind.fitness)
            avg_f0 = sum(ind.fitness for ind in population) / len(population)

            best_fitness_per_gen.append(best_overall.fitness)
            mean_fitness_per_gen.append(avg_f0)

            if verbose:
                print(f"  Поколение 0: best={best_overall.fitness}, avg={avg_f0:.1f}")

            for gen in range(1, self.generations + 1):
                sorted_pop = sorted(population, key=lambda ind: ind.fitness)
                elites = [ind.copy() for ind in sorted_pop[: self.elitism_count]]

                children: List[Individual] = []

                while len(elites) + len(children) < self.pop_size:
                    p1 = self._tournament_select(population)
                    p2 = self._tournament_select(population)
                    child = self._crossover(p1, p2)
                    if self.rng.random() < self.pm:
                        for _ in range(self.K_swaps):
                            self._random_relocate(child.grid, self.rng)
                    children.append(child)

                self._evaluate_batch(children, executor)

                population = elites + children

                best_in_gen = min(population, key=lambda ind: ind.fitness)
                if best_in_gen.fitness < best_overall.fitness:
                    best_overall = best_in_gen.copy()

                avg_fitness = sum(ind.fitness for ind in population) / len(population)
                best_fitness_per_gen.append(best_overall.fitness)
                mean_fitness_per_gen.append(avg_fitness)

                if verbose:
                    print(
                        f"  GA gen {gen:3d}: "
                        f"best_overall={best_overall.fitness:7d}, "
                        f"avg={avg_fitness:8.1f}"
                    )

        log: Dict[str, Any] = {
            "best_fitness_per_gen": best_fitness_per_gen,
            "mean_fitness_per_gen": mean_fitness_per_gen,
        }
        return best_overall, log


# ===========================================================================
# Статистика и CI
# ===========================================================================

def compute_mean_ci(
    data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """mean и 95% CI по строкам. nan игнорируются."""
    from scipy.stats import t as t_dist
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0, ddof=1)
    n_samples = np.sum(~np.isnan(data), axis=0).clip(min=2)
    t_crit = t_dist.ppf(0.975, df=n_samples - 1)
    ci = t_crit * std / np.sqrt(n_samples)
    return mean, ci


# ===========================================================================
# Графики
# ===========================================================================

def plot_convergence(
    ga_logs: List[Dict],
    cnn_logs: List[Dict],
    generations: int,
    save_path: str,
) -> None:
    """Сходимость (лучший fitness) двух алгоритмов с CI."""
    gens = list(range(generations + 1))

    datasets = [
        ("GA", COLOR_GA, ga_logs),
        ("CNN-GA", COLOR_CNN, cnn_logs),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, color, logs in datasets:
        if not logs:
            continue
        data = np.array(
            [log["best_fitness_per_gen"] for log in logs], dtype=float,
        )
        mean, ci = compute_mean_ci(data)
        ax.plot(gens, mean, color=color, lw=2, label=label)
        ax.fill_between(gens, mean - ci, mean + ci, alpha=0.2, color=color)

    ax.set_xlabel("Generation", fontsize=13)
    ax.set_ylabel("Best Fitness (simulation steps)", fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Сохранён: {save_path}")


def plot_mean_fitness(
    ga_logs: List[Dict],
    cnn_logs: List[Dict],
    generations: int,
    n_accumulate: int,
    save_path: str,
) -> None:
    """Средний fitness популяции по поколениям с CI."""
    gens = list(range(generations + 1))

    datasets = [
        ("GA", COLOR_GA, ga_logs),
        ("CNN-GA", COLOR_CNN, cnn_logs),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, color, logs in datasets:
        if not logs:
            continue
        data = np.array(
            [log["mean_fitness_per_gen"] for log in logs], dtype=float,
        )
        mean, ci = compute_mean_ci(data)
        ax.plot(gens, mean, color=color, lw=2, label=label)
        ax.fill_between(gens, mean - ci, mean + ci, alpha=0.2, color=color)

    ax.axvline(
        x=n_accumulate,
        color="gray",
        linestyle="--",
        lw=1.5,
        alpha=0.8,
        label=f"Accumulation end (gen. {n_accumulate})",
    )

    ax.set_xlabel("Generation", fontsize=13)
    ax.set_ylabel("Mean Population Fitness (simulation steps)", fontsize=13)
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
    """Loss суррогата CNN-GA (MSE) по эпохам с CI."""
    all_losses = [log.get("cnn_loss_per_epoch", []) for log in cnn_logs]

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
    ax.plot(epochs, mean, color=COLOR_CNN, lw=2, label="CNN-GA loss (MSE)")
    ax.fill_between(
        epochs, mean - ci, mean + ci, alpha=0.25, color=COLOR_CNN,
    )

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Loss (MSE, normalized)", fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Сохранён: {save_path}")


def plot_surrogate_r2(
    cnn_logs: List[Dict],
    save_path: str,
) -> None:
    """R² суррогата CNN-GA по обучениям/дообучениям."""
    all_r2 = [log.get("surrogate_r2_per_train", []) for log in cnn_logs]

    if not any(all_r2):
        print("Нет данных R² суррогата CNN-GA — пропуск графика.")
        return

    max_len = max((len(r) for r in all_r2), default=0)
    if max_len == 0:
        print("Нет данных R² суррогата CNN-GA — пропуск графика.")
        return

    r2_arr = np.full((len(cnn_logs), max_len), np.nan)
    for i, r2s in enumerate(all_r2):
        r2_arr[i, : len(r2s)] = r2s

    trains = list(range(1, max_len + 1))
    mean, ci = compute_mean_ci(r2_arr)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(trains, mean, color=COLOR_CNN, lw=2, marker="o", label="R²")
    ax.fill_between(
        trains, mean - ci, mean + ci, alpha=0.25, color=COLOR_CNN,
    )

    ax.set_xlabel("Training / Fine-tuning Index", fontsize=13)
    ax.set_ylabel("R²", fontsize=13)
    ax.set_ylim(-0.1, 1.0)
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
    """Боксплот финального fitness двух алгоритмов с точками."""
    all_data = []
    labels = []
    colors = []

    for name, color, logs in [
        ("GA", COLOR_GA, ga_logs),
        ("CNN-GA", COLOR_CNN, cnn_logs),
    ]:
        if not logs:
            continue
        finals = [log["best_fitness_per_gen"][-1] for log in logs]
        all_data.append(finals)
        labels.append(name)
        colors.append(color)

    if not all_data:
        print("Нет данных для боксплота — пропуск.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(
        all_data,
        labels=labels,
        patch_artist=True,
        widths=0.4,
    )
    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c)
        box.set_alpha(0.4)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    # Индивидуальные точки
    for x_pos, (vals, c) in enumerate(zip(all_data, colors), start=1):
        ax.scatter(
            np.full(len(vals), x_pos),
            vals,
            zorder=5,
            color=c,
            edgecolors="black",
            s=60,
            alpha=0.9,
        )

    ax.set_ylabel("Final Fitness (simulation steps)", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Сохранён: {save_path}")


# ===========================================================================
# Вспомогательные функции
# ===========================================================================

def save_log(log: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def load_log(path: str) -> Dict[str, Any]:
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

    # -------------------------------------------------------------------
    # Прогоны GA
    # -------------------------------------------------------------------
    for run_idx, seed in enumerate(SEEDS):
        log_path = f"logs/run_ga_{seed}.json"
        if os.path.exists(log_path):
            print_separator(
                f"GA прогон {run_idx + 1}/{N_RUNS}  |  seed={seed}  "
                f"[загружен с диска]"
            )
            log = load_log(log_path)
            print(
                f"  Загружен: {log_path}  "
                f"(best={log['best_fitness_per_gen'][-1]})"
            )
        else:
            print_separator(f"GA прогон {run_idx + 1}/{N_RUNS}  |  seed={seed}")
            ga = TrackingGA(**HARD_CASE, K_swaps=M_SWAPS, rng_seed=seed)
            best, log = ga.run(verbose=True)
            log["algo"] = "ga"
            log["seed"] = seed
            save_log(log, log_path)
            print(f"Лог сохранён: {log_path}")
            print(
                f">> GA прогон {run_idx + 1} завершён: "
                f"best fitness = {best.fitness}"
            )
        ga_logs.append(log)

    # -------------------------------------------------------------------
    # Прогоны CNN-GA
    # -------------------------------------------------------------------
    for run_idx, seed in enumerate(SEEDS):
        log_path = f"logs/run_cnn_ga_{seed}.json"
        if os.path.exists(log_path):
            print_separator(
                f"CNN-GA прогон {run_idx + 1}/{N_RUNS}  |  seed={seed}  "
                f"[загружен с диска]"
            )
            log = load_log(log_path)
            print(
                f"  Загружен: {log_path}  "
                f"(best={log['best_fitness_per_gen'][-1]})"
            )
        else:
            print_separator(
                f"CNN-GA прогон {run_idx + 1}/{N_RUNS}  |  seed={seed}"
            )
            cnn_ga = CNNGuidedGA(
                **HARD_CASE,
                K_swaps=M_SWAPS,
                m_swaps=M_SWAPS,
                n_accumulate=5,
                retrain_interval=3,
                n_train_epochs=30,
                max_buffer_size=10_000,
                oversample_factor=3,
                warmstart_extra=50,
                rng_seed=seed,
            )
            best, log = cnn_ga.run(verbose=True)
            log["algo"] = "cnn_ga"
            log["seed"] = seed
            save_log(log, log_path)
            print(f"Лог сохранён: {log_path}")
            print(
                f">> CNN-GA прогон {run_idx + 1} завершён: "
                f"best fitness = {best.fitness}"
            )
        cnn_logs.append(log)

    # -------------------------------------------------------------------
    # Итоговая статистика в терминале
    # -------------------------------------------------------------------
    print_separator("Итоговая статистика")

    for name, logs in [
        ("GA     ", ga_logs),
        ("CNN-GA ", cnn_logs),
    ]:
        if not logs:
            print(f"{name}: нет данных")
            continue
        finals = [log["best_fitness_per_gen"][-1] for log in logs]
        print(f"{name}: {finals}")
        print(
            f"         mean={np.mean(finals):.1f}, std={np.std(finals):.1f}"
        )

    # -------------------------------------------------------------------
    # Графики
    # -------------------------------------------------------------------
    print_separator("Построение графиков")
    gens = HARD_CASE["generations"]

    plot_convergence(
        ga_logs, cnn_logs, gens, "plots/convergence.png",
    )
    plot_mean_fitness(
        ga_logs, cnn_logs, gens, 5,
        "plots/mean_fitness.png",
    )
    plot_cnn_loss(cnn_logs, "plots/cnn_loss.png")
    plot_surrogate_r2(cnn_logs, "plots/cnn_surrogate_r2.png")
    plot_final_boxplot(
        ga_logs, cnn_logs, "plots/final_fitness_boxplot.png",
    )

    print("\nЭксперимент завершён успешно.")


if __name__ == "__main__":
    main()
