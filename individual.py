import random
from collections import Counter
from typing import List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from constants import (ROAD, ENTRY, EXIT, SAVE, NON_ROAD,
                       CONFIG_CELL_TO_IDX, CONFIG_COLORS, CONFIG_LABELS,
                       CONFIG_CMAP)


class Individual:
    """Сетка m*n из меток типов ячеек, представляющая одну конфигурацию склада."""

    def __init__(self, grid_array: np.ndarray, fitness: int = None):
        self.grid = grid_array.copy()
        self.m, self.n = self.grid.shape
        self.fitness = fitness

    @classmethod
    def random_individual(cls, m: int, n: int,
                          entries: int, exits: int, saves: int) -> "Individual":
        """Создать случайную корректную особь с заданным количеством ячеек каждого типа."""
        assert entries + exits + saves <= m * n, \
            "Сумма типов ячеек превышает размер матрицы"
        arr = np.full((m, n), ROAD, dtype=object)
        positions = random.sample(
            [(i, j) for i in range(m) for j in range(n)],
            entries + exits + saves,
        )
        idx = 0
        for label, count in zip([ENTRY, EXIT, SAVE], [entries, exits, saves]):
            for _ in range(count):
                arr[positions[idx]] = label
                idx += 1
        return cls(arr)

    def copy(self) -> "Individual":
        """Глубокая копия особи."""
        return Individual(self.grid.copy(), self.fitness)

    def to_position_lists(self) -> Tuple[List[tuple], List[tuple], List[tuple]]:
        """Вернуть (entries, exits, saves) как списки позиций (строка, столбец)."""
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
        """Подсчитать количество ячеек каждого типа."""
        cnt = Counter(self.grid.flatten())
        return {k: cnt.get(k, 0) for k in [ROAD, ENTRY, EXIT, SAVE]}

    def correct_counts(self, desired_entries: int, desired_exits: int,
                       desired_saves: int, rng=random):
        """Восстановить точное количество типов ячеек после скрещивания."""
        desired = {ENTRY: desired_entries, EXIT: desired_exits, SAVE: desired_saves}
        cur = self.count_types()

        # убрать избыток не-дорожных ячеек
        for t in NON_ROAD:
            excess = cur[t] - desired[t]
            if excess > 0:
                positions = [(i, j) for i in range(self.m) for j in range(self.n)
                             if self.grid[i, j] == t]
                for i, j in rng.sample(positions, excess):
                    self.grid[i, j] = ROAD
                cur[t] -= excess
                cur[ROAD] = cur.get(ROAD, 0) + excess

        # заполнить дефицит не-дорожных ячеек из дорожных позиций
        for t in NON_ROAD:
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

    def plot(self, title: str = None) -> None:
        """Визуализация конфигурации склада в виде цветной сетки с легендой."""
        numeric = np.vectorize(CONFIG_CELL_TO_IDX.get)(self.grid)

        fig, ax = plt.subplots(figsize=(8, 8))
        mesh = ax.pcolormesh(numeric, cmap=CONFIG_CMAP, vmin=0, vmax=3,
                             edgecolors="gray", linewidth=0.5)
        mesh.set_clip_on(False)
        ax.set_xlim(0, self.n)
        ax.set_ylim(self.m, 0)
        ax.set_aspect("equal")
        ax.tick_params(bottom=False, left=False,
                       labelbottom=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # легенда
        patches = [mpatches.Patch(color=CONFIG_COLORS[t], label=CONFIG_LABELS[t])
                   for t in (ROAD, ENTRY, EXIT, SAVE)]
        ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.02, 1),
                  fontsize=11, frameon=True)

        # заголовок
        if title is None:
            title = "Конфигурация склада"
            if self.fitness is not None:
                title += f"(fitness = {self.fitness})"
        ax.set_title(title, fontsize=14, pad=10)

        # подзаголовок с размерами и количеством ячеек
        counts = self.count_types()
        subtitle = (f"{self.m}×{self.n}  |  "
                    f"Входов: {counts[ENTRY]}, "
                    f"Выходов: {counts[EXIT]}, "
                    f"Хранилищ: {counts[SAVE]}")
        ax.set_xlabel(subtitle, fontsize=11)

        plt.tight_layout()
        plt.show()

