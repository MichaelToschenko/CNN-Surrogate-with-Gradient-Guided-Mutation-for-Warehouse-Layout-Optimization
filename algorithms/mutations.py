import random

import numpy as np

from constants import ROAD
from individual import Individual


def random_relocate_mutation(
    individual: Individual, k: int, rng: random.Random,
) -> None:
    """K random relocations: a non-road cell -> a ROAD position."""
    for _ in range(k):
        non_road = [tuple(p) for p in np.argwhere(individual.grid != ROAD)]
        road_pos = [tuple(p) for p in np.argwhere(individual.grid == ROAD)]
        if not non_road or not road_pos:
            return
        src = non_road[rng.randrange(len(non_road))]
        dst = road_pos[rng.randrange(len(road_pos))]
        individual.grid[dst] = individual.grid[src]
        individual.grid[src] = ROAD
