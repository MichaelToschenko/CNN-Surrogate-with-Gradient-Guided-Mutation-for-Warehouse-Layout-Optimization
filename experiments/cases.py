"""Shared warehouse configurations for comparative experiments."""

from typing import Any, Dict, List


# Hard 15×15 case for a fair GA vs CNN-GA comparison
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

# Case for evolution visualization (short run)
VIZ_CASE: Dict[str, Any] = dict(
    m=15, n=15, entries=5, exits=5, saves=30,
    num_robots=12, containers_to_process=160, prob=0.35,
    pop_size=50, generations=20, pm=0.3,
    k_tournament=3, elitism_count=3,
)

# Seeds for reproducible runs
SEEDS: List[int] = [42, 123, 456, 789, 1337]

# Relocation operator parameter: number of relocations per mutation
M_SWAPS: int = 4

# CNN-GA parameters used by both comparison and visualization scripts
CNN_PARAMS: Dict[str, Any] = dict(
    K_swaps=M_SWAPS,
    m_swaps=M_SWAPS,
    n_accumulate=5,
    retrain_interval=3,
    n_train_epochs=30,
    max_buffer_size=10_000,
    oversample_factor=3,
    warmstart_extra=50,
)
