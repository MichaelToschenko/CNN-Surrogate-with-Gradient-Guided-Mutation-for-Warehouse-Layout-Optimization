import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from storage_fitness import evaluate_configuration, Simulator

# ── Метки типов ячеек ──────────────────────────────────────────────────────────
ROAD  = "Road"
ENTRY = "Entry"
EXIT  = "Exit"
SAVE  = "Save"

_NON_ROAD    = [ENTRY, EXIT, SAVE]
_CELL_CHARS  = {ROAD: ".", ENTRY: "E", EXIT: "X", SAVE: "S"}


# ── Вспомогательные метрики ────────────────────────────────────────────────────

def manhattan(a: tuple, b: tuple) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def avg_save_to_save_distance(ind: "Individual") -> float:
    """Среднее попарное манхэттенское расстояние между всеми ячейками хранения."""
    _, _, saves = ind.to_position_lists()
    if len(saves) < 2:
        return 0.0
    dists = [manhattan(saves[i], saves[j])
             for i in range(len(saves))
             for j in range(i + 1, len(saves))]
    return float(np.mean(dists))


def avg_objects_in_radius(ind: "Individual", R: int = 3) -> float:
    """Среднее количество нe-дорожных объектов в манхэттенском радиусе R от каждой ячейки хранения."""
    entries, exits, saves = ind.to_position_lists()
    objects = entries + exits + saves
    if not saves:
        return 0.0
    counts = [
        sum(1 for o in objects if o != s and manhattan(s, o) <= R)
        for s in saves
    ]
    return float(np.mean(counts))


def avg_save_to_nearest_entry_exit(ind: "Individual") -> float:
    """Среднее манхэттенское расстояние от каждой ячейки хранения до ближайшего входа или выхода."""
    entries, exits, saves = ind.to_position_lists()
    targets = entries + exits
    if not saves or not targets:
        return 0.0
    return float(np.mean([min(manhattan(s, t) for t in targets) for s in saves]))


# ── Особь ──────────────────────────────────────────────────────────────────────

class Individual:
    """Сетка m×n из меток типов ячеек, представляющая одну конфигурацию склада."""

    def __init__(self, grid_array: np.ndarray, fitness: int = None):
        self.grid = grid_array.copy()
        self.m, self.n = self.grid.shape
        self.fitness = fitness

    # ── Создание ──────────────────────────────────────────────────────────────

    @classmethod
    def random_individual(cls, m: int, n: int,
                          entries: int, exits: int, saves: int) -> "Individual":
        """Создать случайную корректную особь с заданным количеством ячеек каждого типа."""
        assert entries + exits + saves <= m * n, \
            "Сумма типов ячеек превышает размер матрицы"
        arr = np.full((m, n), ROAD, dtype=object)
        positions = random.sample([(i, j) for i in range(m) for j in range(n)],
                                  entries + exits + saves)
        idx = 0
        for label, count in zip([ENTRY, EXIT, SAVE], [entries, exits, saves]):
            for _ in range(count):
                arr[positions[idx]] = label
                idx += 1
        return cls(arr)

    def copy(self) -> "Individual":
        return Individual(self.grid.copy(), self.fitness)

    # ── Преобразование ────────────────────────────────────────────────────────

    def to_position_lists(self) -> Tuple[List[tuple], List[tuple], List[tuple]]:
        """Вернуть (entries, exits, saves) как списки кортежей (строка, столбец)."""
        entries, exits, saves = [], [], []
        for i in range(self.m):
            for j in range(self.n):
                cell = self.grid[i, j]
                if cell == ENTRY:
                    entries.append((i, j))
                elif cell == EXIT:
                    exits.append((i, j))
                elif cell == SAVE:
                    saves.append((i, j))
        return entries, exits, saves

    def count_types(self) -> dict:
        cnt = Counter(self.grid.flatten())
        return {k: cnt.get(k, 0) for k in [ROAD, ENTRY, EXIT, SAVE]}

    # ── Восстановление после скрещивания ──────────────────────────────────────

    def correct_counts(self, desired_entries: int, desired_exits: int,
                       desired_saves: int, rng=random):
        """Восстановить точное количество типов ячеек после скрещивания.

        Избыточные не-дорожные ячейки заменяются на дорогу; дефицит
        восполняется случайными дорожными ячейками.
        """
        desired = {ENTRY: desired_entries, EXIT: desired_exits, SAVE: desired_saves}
        cur = self.count_types()

        # убрать избыток не-дорожных ячеек
        for t in _NON_ROAD:
            excess = cur[t] - desired[t]
            if excess > 0:
                positions = [(i, j) for i in range(self.m) for j in range(self.n)
                             if self.grid[i, j] == t]
                for i, j in rng.sample(positions, excess):
                    self.grid[i, j] = ROAD
                cur[t] -= excess
                cur[ROAD] = cur.get(ROAD, 0) + excess

        # заполнить дефицит не-дорожных ячеек из дорожных позиций
        for t in _NON_ROAD:
            deficit = desired[t] - cur[t]
            if deficit > 0:
                road_positions = [(i, j) for i in range(self.m) for j in range(self.n)
                                  if self.grid[i, j] == ROAD]
                assert len(road_positions) >= deficit, \
                    "Не хватает дорожных позиций для коррекции дефицита"
                for i, j in rng.sample(road_positions, deficit):
                    self.grid[i, j] = t
                cur[t] += deficit
                cur[ROAD] -= deficit

    # ── Отображение ───────────────────────────────────────────────────────────

    def __str__(self) -> str:
        rows = ["".join(_CELL_CHARS.get(self.grid[i, j], "?")
                        for j in range(self.n))
                for i in range(self.m)]
        return "\n".join(rows)


# ── Генетический алгоритм ──────────────────────────────────────────────────────

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
        simulator_class=None,
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

        self.simulator_class = simulator_class
        self.n_jobs = os.cpu_count() if n_jobs == -1 else n_jobs

        self.rng = random.Random(rng_seed)
        np.random.seed(rng_seed)

    # ── Популяция ─────────────────────────────────────────────────────────────

    def _init_population(self) -> List[Individual]:
        return [Individual.random_individual(self.m, self.n,
                                             self.entries, self.exits, self.saves)
                for _ in range(self.pop_size)]

    # ── Отбор ─────────────────────────────────────────────────────────────────

    def _tournament_select(self, population: List[Individual]) -> Individual:
        contenders = self.rng.sample(population, self.k_tournament)
        return min(contenders,
                   key=lambda ind: ind.fitness if ind.fitness is not None else float('inf')
                   ).copy()

    # ── Скрещивание ───────────────────────────────────────────────────────────

    def _crossover(self, p1: Individual, p2: Individual) -> Individual:
        """Одноточечное горизонтальное или вертикальное скрещивание."""
        child_grid = np.empty((self.m, self.n), dtype=object)
        if self.rng.random() < 0.5:
            # вертикальный разрез
            k = self.rng.randint(1, self.n - 1)
            child_grid[:, :k] = p1.grid[:, :k]
            child_grid[:, k:] = p2.grid[:, k:]
        else:
            # горизонтальный разрез
            k = self.rng.randint(1, self.m - 1)
            child_grid[:k, :] = p1.grid[:k, :]
            child_grid[k:, :] = p2.grid[k:, :]
        child = Individual(child_grid)
        child.correct_counts(self.entries, self.exits, self.saves, rng=self.rng)
        return child

    # ── Мутация ───────────────────────────────────────────────────────────────

    def _mutate(self, individual: Individual):
        """K случайных перестановок среди не-дорожных ячеек."""
        coords = [(i, j) for i in range(self.m) for j in range(self.n)
                  if individual.grid[i, j] != ROAD]
        if len(coords) < 2:
            return
        for _ in range(self.K_swaps):
            (ia, ja), (ib, jb) = self.rng.sample(coords, 2)
            if individual.grid[ia, ja] != individual.grid[ib, jb]:
                individual.grid[ia, ja], individual.grid[ib, jb] = \
                    individual.grid[ib, jb], individual.grid[ia, ja]

    # ── Оценка пригодности ────────────────────────────────────────────────────

    def _evaluate_batch(self, individuals: List[Individual]):
        """Параллельная оценка пригодности для списка особей."""
        args_list = []
        for ind in individuals:
            entries_arr, exits_arr, saves_arr = ind.to_position_lists()
            args_list.append((
                self.m, self.n, self.num_robots,
                entries_arr, exits_arr, saves_arr,
                self.containers_to_process, self.prob
            ))
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            metrics = list(executor.map(evaluate_configuration, args_list))
        for ind, metric in zip(individuals, metrics):
            ind.fitness = metric

    # ── Вспомогательные метрики ───────────────────────────────────────────────

    def _record_metrics(self, best: Individual, metrics_history: dict):
        metrics_history["avg_ss_dist"].append(avg_save_to_save_distance(best))
        metrics_history["avg_local_density"].append(avg_objects_in_radius(best, R=3))
        metrics_history["avg_s_to_ex"].append(avg_save_to_nearest_entry_exit(best))

    # ── Основной цикл эволюции ────────────────────────────────────────────────

    def run(self, verbose: bool = True) -> Tuple[Individual, list]:
        """Запустить генетический алгоритм.

        Возвращает:
            best_overall: особь с наименьшим значением пригодности.
            history:      список словарей со статистикой по каждому поколению.
        """
        population = self._init_population()
        metrics_history = {"avg_ss_dist": [], "avg_local_density": [], "avg_s_to_ex": []}

        if verbose:
            print("Оценка начальной популяции...")
        self._evaluate_batch(population)

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

            self._evaluate_batch(children)
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

    # ── Отрисовка графиков ────────────────────────────────────────────────────

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
        """Графики трёх вспомогательных пространственных метрик по поколениям."""
        gens = list(range(len(metrics_history["avg_ss_dist"])))
        plt.figure(figsize=(16, 4))

        plt.subplot(1, 3, 1)
        plt.plot(gens, metrics_history["avg_ss_dist"], marker="o")
        plt.title("Среднее расстояние Save–Save")
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
        plt.title("Среднее расстояние Save → ближайший Entry/Exit")
        plt.xlabel("Поколение")
        plt.ylabel("Манхэттенское расстояние")
        plt.grid(True)

        plt.tight_layout()
        plt.show()
