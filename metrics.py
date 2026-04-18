from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from individual import Individual


def manhattan(a: tuple, b: tuple) -> int:
    """Manhattan distance between two positions."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def avg_save_to_save_distance(ind: "Individual") -> float:
    """Average pairwise Manhattan distance between storage cells."""
    _, _, saves = ind.to_position_lists()
    if len(saves) < 2:
        return 0.0
    dists = [manhattan(saves[i], saves[j])
             for i in range(len(saves))
             for j in range(i + 1, len(saves))]
    return float(np.mean(dists))


def avg_objects_in_radius(ind: "Individual", R: int = 3) -> float:
    """Average number of non-road objects within radius R of each storage cell."""
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
    """Average distance from a storage cell to the nearest entry or exit."""
    entries, exits, saves = ind.to_position_lists()
    targets = entries + exits
    if not saves or not targets:
        return 0.0
    return float(np.mean([min(manhattan(s, t) for t in targets) for s in saves]))
