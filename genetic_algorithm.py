import os
import random
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from constants import ROAD
from individual import Individual
from metrics import (
    avg_save_to_save_distance,
    avg_objects_in_radius,
    avg_save_to_nearest_entry_exit,
)
from simulation import evaluate_configuration


# -- Генетический алгоритм -------------------------------------------------

class GeneticAlgorithm:
    """Эволюционирует конфигурации сетки склада для минимизации количества шагов симуляции."""

    def __init__(
        self,
        m: int,
        n: int,
        entries: int,
        exits: int,
        saves: int,
        num_robots: int,
        containers_to_process: int,
        prob: float,
        pop_size: int = 50,
        generations: int = 50,
        k_tournament: int = 3,
        elitism_count: int = 2,
        pm: float = 0.1,
        K_swaps: int = 5,
        rng_seed: int = None,
        evaluate_fn: Callable = None,
        n_jobs: int = -1,
    ):
        self.m, self.n = m, n
        self.entries, self.exits, self.saves = entries, exits, saves
        self.num_robots = num_robots
        self.containers_to_process = containers_to_process
        self.prob = prob

        self.pop_size = pop_size
        self.generations = generations
        self.k_tournament = k_tournament
        self.elitism_count = elitism_count
        self.pm = pm
        self.K_swaps = K_swaps

        self._evaluate_fn = evaluate_fn or evaluate_configuration
        self.n_jobs = os.cpu_count() if n_jobs == -1 else n_jobs

        self.rng = random.Random(rng_seed)

    # -- Популяция ----------------------------------------------------------

    def _init_population(self) -> List[Individual]:
        """Создать начальную случайную популяцию."""
        return [Individual.random_individual(self.m, self.n,
                                             self.entries, self.exits, self.saves,
                                             rng=self.rng)
                for _ in range(self.pop_size)]

    # -- Отбор --------------------------------------------------------------

    def _tournament_select(self, population: List[Individual]) -> Individual:
        """Турнирная селекция с размером турнира k_tournament."""
        contenders = self.rng.sample(population, self.k_tournament)
        return min(contenders,
                   key=lambda ind: ind.fitness if ind.fitness is not None else float("inf"))

    # -- Скрещивание --------------------------------------------------------

    def _crossover(self, p1: Individual, p2: Individual) -> Individual:
        """Одноточечное горизонтальное или вертикальное скрещивание."""
        child_grid = np.empty((self.m, self.n), dtype=object)
        if self.rng.random() < 0.5:
            k = self.rng.randint(1, self.n - 1)
            child_grid[:, :k] = p1.grid[:, :k]
            child_grid[:, k:] = p2.grid[:, k:]
        else:
            k = self.rng.randint(1, self.m - 1)
            child_grid[:k, :] = p1.grid[:k, :]
            child_grid[k:, :] = p2.grid[k:, :]
        child = Individual(child_grid)
        child.correct_counts(self.entries, self.exits, self.saves, rng=self.rng)
        return child

    # -- Мутация ------------------------------------------------------------

    def _mutate(self, individual: Individual):
        """K случайных перестановок среди не-дорожных ячеек."""
        coords = [tuple(p) for p in np.argwhere(individual.grid != ROAD)]
        if len(coords) < 2:
            return
        for _ in range(self.K_swaps):
            (ia, ja), (ib, jb) = self.rng.sample(coords, 2)
            if individual.grid[ia, ja] != individual.grid[ib, jb]:
                individual.grid[ia, ja], individual.grid[ib, jb] = \
                    individual.grid[ib, jb], individual.grid[ia, ja]

    # -- Оценка пригодности -------------------------------------------------

    def _evaluate_batch(self, individuals: List[Individual], executor):
        """Параллельная оценка пригодности для списка особей."""
        args_list = []
        for ind in individuals:
            entries_arr, exits_arr, saves_arr = ind.to_position_lists()
            args_list.append((
                self.m, self.n, self.num_robots,
                entries_arr, exits_arr, saves_arr,
                self.containers_to_process, self.prob,
            ))
        metrics = list(executor.map(self._evaluate_fn, args_list))
        for ind, metric in zip(individuals, metrics):
            ind.fitness = metric

    # -- Вспомогательные метрики --------------------------------------------

    def _record_metrics(self, best: Individual, metrics_history: dict):
        """Записать пространственные метрики лучшей особи."""
        metrics_history["avg_ss_dist"].append(avg_save_to_save_distance(best))
        metrics_history["avg_local_density"].append(avg_objects_in_radius(best, R=3))
        metrics_history["avg_s_to_ex"].append(avg_save_to_nearest_entry_exit(best))

    # -- Основной цикл эволюции ---------------------------------------------

    def run(self, verbose: bool = True) -> Tuple[Individual, list]:
        """Запустить генетический алгоритм. Возвращает лучшую особь и историю."""
        population = self._init_population()
        metrics_history = {"avg_ss_dist": [], "avg_local_density": [], "avg_s_to_ex": []}

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            if verbose:
                print("Оценка начальной популяции...")
            self._evaluate_batch(population, executor)

            best_overall = min(population, key=lambda ind: ind.fitness)
            avg_f0 = sum(ind.fitness for ind in population) / len(population)

            history = [{
                "generation": 0,
                "best_individual": best_overall,
                "avg_fitness": avg_f0,
                "best_overall_fitness": best_overall.fitness,
            }]
            self._record_metrics(best_overall, metrics_history)

            if verbose:
                print(f"Поколение 0: best = {best_overall.fitness}, avg = {avg_f0:.2f}")

            for gen in range(1, self.generations + 1):
                sorted_pop = sorted(population, key=lambda ind: ind.fitness)
                elites = [ind.copy() for ind in sorted_pop[:self.elitism_count]]

                children = []
                while len(elites) + len(children) < self.pop_size:
                    p1 = self._tournament_select(population)
                    p2 = self._tournament_select(population)
                    child = self._crossover(p1, p2)
                    if self.rng.random() < self.pm:
                        self._mutate(child)
                    children.append(child)

                self._evaluate_batch(children, executor)
                population = elites + children

                best_in_gen = min(population, key=lambda ind: ind.fitness)
                if best_in_gen.fitness < best_overall.fitness:
                    best_overall = best_in_gen.copy()

                avg_fitness = sum(ind.fitness for ind in population) / len(population)
                self._record_metrics(best_in_gen, metrics_history)

                history.append({
                    "generation": gen,
                    "best_individual": best_in_gen,
                    "avg_fitness": avg_fitness,
                    "best_overall_fitness": best_overall.fitness,
                })

                if verbose:
                    print(f"Поколение {gen}: "
                          f"best = {best_in_gen.fitness}, "
                          f"avg = {avg_fitness:.2f}, "
                          f"best_overall = {best_overall.fitness}")

        self._plot_fitness(history)
        self._plot_metrics(metrics_history)
        return best_overall, history

    # -- Графики ------------------------------------------------------------

    @staticmethod
    def _plot_fitness(history: list):
        """График средней пригодности по поколениям."""
        generations = [h["generation"] for h in history]
        avg_vals    = [h["avg_fitness"]  for h in history]
        plt.figure(figsize=(9, 5))
        plt.plot(generations, avg_vals, marker="o")
        plt.title("Средняя пригодность по поколениям")
        plt.xlabel("Поколение")
        plt.ylabel("Средняя пригодность (fitness)")
        plt.grid(True)
        plt.show()

    @staticmethod
    def _plot_metrics(metrics_history: dict):
        """Графики пространственных метрик по поколениям."""
        gens = list(range(len(metrics_history["avg_ss_dist"])))
        plt.figure(figsize=(16, 4))

        plt.subplot(1, 3, 1)
        plt.plot(gens, metrics_history["avg_ss_dist"], marker="o")
        plt.title("Среднее расстояние Save-Save")
        plt.xlabel("Поколение")
        plt.ylabel("Манхэттенское расстояние")
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(gens, metrics_history["avg_local_density"], marker="o")
        plt.title("Среднее число объектов в радиусе R=3 от Save")
        plt.xlabel("Поколение")
        plt.ylabel("Количество объектов")
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(gens, metrics_history["avg_s_to_ex"], marker="o")
        plt.title("Среднее расстояние Save -> ближайший Entry/Exit")
        plt.xlabel("Поколение")
        plt.ylabel("Манхэттенское расстояние")
        plt.grid(True)

        plt.tight_layout()
        plt.show()
