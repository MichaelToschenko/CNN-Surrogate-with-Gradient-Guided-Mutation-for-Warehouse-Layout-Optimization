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
    """An m*n grid of cell-type labels representing a single warehouse configuration."""

    def __init__(self, grid_array: np.ndarray, fitness: int = None):
        self.grid = grid_array.copy()
        self.m, self.n = self.grid.shape
        self.fitness = fitness

    @classmethod
    def random_individual(cls, m: int, n: int,
                          entries: int, exits: int, saves: int,
                          rng=random) -> "Individual":
        """Create a random valid individual with the requested cell-type counts."""
        assert entries + exits + saves <= m * n, \
            "Sum of cell types exceeds matrix size"
        arr = np.full((m, n), ROAD, dtype=object)
        positions = rng.sample(
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
        """Deep copy of the individual."""
        return Individual(self.grid.copy(), self.fitness)

    def to_position_lists(self) -> Tuple[List[tuple], List[tuple], List[tuple]]:
        """Return (entries, exits, saves) as lists of (row, col) positions."""
        entries = [tuple(p) for p in np.argwhere(self.grid == ENTRY)]
        exits   = [tuple(p) for p in np.argwhere(self.grid == EXIT)]
        saves   = [tuple(p) for p in np.argwhere(self.grid == SAVE)]
        return entries, exits, saves

    def count_types(self) -> dict:
        """Count cells of each type."""
        cnt = Counter(self.grid.flat)
        return {k: cnt.get(k, 0) for k in [ROAD, ENTRY, EXIT, SAVE]}

    def correct_counts(self, desired_entries: int, desired_exits: int,
                       desired_saves: int, rng=random):
        """Restore exact cell-type counts after crossover."""
        desired = {ENTRY: desired_entries, EXIT: desired_exits, SAVE: desired_saves}
        cur = self.count_types()

        # remove excess non-road cells (one np.argwhere per surplus type)
        for t in NON_ROAD:
            excess = cur[t] - desired[t]
            if excess > 0:
                positions = [tuple(p) for p in np.argwhere(self.grid == t)]
                for idx in rng.sample(positions, excess):
                    self.grid[idx] = ROAD
                cur[t] -= excess
                cur[ROAD] = cur.get(ROAD, 0) + excess

        # one ROAD scan covers all deficient types
        if any(desired[t] - cur[t] > 0 for t in NON_ROAD):
            road_positions = [tuple(p) for p in np.argwhere(self.grid == ROAD)]
            for t in NON_ROAD:
                deficit = desired[t] - cur[t]
                if deficit > 0:
                    assert len(road_positions) >= deficit, \
                        "Not enough road positions to cover the deficit"
                    chosen = rng.sample(road_positions, deficit)
                    for idx in chosen:
                        self.grid[idx] = t
                        road_positions.remove(idx)
                    cur[t] += deficit
                    cur[ROAD] -= deficit

    def plot(self, title: str = None) -> None:
        """Visualize the warehouse configuration as a colored grid with a legend."""
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

        # legend
        patches = [mpatches.Patch(color=CONFIG_COLORS[t], label=CONFIG_LABELS[t])
                   for t in (ROAD, ENTRY, EXIT, SAVE)]
        ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.02, 1),
                  fontsize=11, frameon=True)

        # title
        if title is None:
            title = "Warehouse configuration"
            if self.fitness is not None:
                title += f" (fitness = {self.fitness})"
        ax.set_title(title, fontsize=14, pad=10)

        # subtitle with dimensions and cell counts
        counts = self.count_types()
        subtitle = (f"{self.m}×{self.n}  |  "
                    f"Entries: {counts[ENTRY]}, "
                    f"Exits: {counts[EXIT]}, "
                    f"Storage: {counts[SAVE]}")
        ax.set_xlabel(subtitle, fontsize=11)

        plt.tight_layout()
        plt.show()
