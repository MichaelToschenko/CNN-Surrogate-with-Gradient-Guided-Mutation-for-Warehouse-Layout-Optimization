import json
import os
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")

import numpy as np

from algorithms import CNNGuidedGA
from experiments import plots
from experiments.cases import CNN_PARAMS, VIZ_CASE
from individual import Individual


SNAPSHOT_INTERVAL = 5
SEED = 42


# ---------------------------------------------------------------------------
# Snapshot serialization
# ---------------------------------------------------------------------------

def save_snapshots(
    snapshots: List[Individual], gen_numbers: List[int], path: str,
) -> None:
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


# ---------------------------------------------------------------------------
# Run with snapshot collection via callbacks
# ---------------------------------------------------------------------------

def run_with_snapshots() -> Tuple[List[Individual], List[int]]:
    snapshots: List[Individual] = []
    gen_numbers: List[int] = []

    def on_init(best_overall: Individual) -> None:
        snapshots.append(best_overall.copy())
        gen_numbers.append(0)

    def on_gen(gen: int, best_in_gen: Individual, best_overall: Individual) -> None:
        if gen % SNAPSHOT_INTERVAL == 0:
            snapshots.append(best_overall.copy())
            gen_numbers.append(gen)

    cnn = CNNGuidedGA(**VIZ_CASE, **CNN_PARAMS, rng_seed=SEED)
    cnn.run(verbose=True, on_init_end=on_init, on_generation_end=on_gen)
    return snapshots, gen_numbers


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    log_path = "logs/evolution_cnn_ga.json"

    if os.path.exists(log_path):
        snapshots, gen_numbers = load_snapshots(log_path)
        print(f"Loaded {len(snapshots)} snapshots from {log_path}")
    else:
        print("=== CNN-GA run ===")
        snapshots, gen_numbers = run_with_snapshots()
        save_snapshots(snapshots, gen_numbers, log_path)
        print(f"Logs saved: {log_path}")

    plots.plot_evolution(
        snapshots, gen_numbers,
        "plots/evolution_cnn_ga.png",
        title="CNN-GA: Evolution of Best Warehouse Layout  (15×15)",
    )
    print("Done.")


if __name__ == "__main__":
    main()
