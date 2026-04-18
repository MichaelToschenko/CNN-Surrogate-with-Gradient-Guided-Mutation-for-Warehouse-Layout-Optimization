import os
import random
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from algorithms.mutations import random_relocate_mutation
from individual import Individual
from metrics import (
    avg_save_to_save_distance,
    avg_objects_in_radius,
    avg_save_to_nearest_entry_exit,
)
from simulation import evaluate_configuration


MutationOp = Callable[[Individual, int, random.Random], None]
GenCallback = Callable[[int, Individual, Individual], None]
InitCallback = Callable[[Individual], None]


class BaseGA:
    """
    Base GA: a single evolution loop, shared operators, hooks for subclasses.

    Subclasses may override:
      - _setup(executor)                              — after gen 0 evaluation
      - _generate_children(pop, n_needed, executor)   — child generation
      - _post_generation(gen, children, executor)     — after evaluating children
      - _extra_log()                                  — extras for the final log
    """

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
        rng_seed: int = 42,
        evaluate_fn: Optional[Callable] = None,
        n_jobs: int = -1,
        mutation_op: MutationOp = random_relocate_mutation,
    ) -> None:
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
        self.mutation_op = mutation_op

        self.rng = random.Random(rng_seed)

    # -- Shared operators --------------------------------------------------

    def _init_population(self) -> List[Individual]:
        return [
            Individual.random_individual(
                self.m, self.n, self.entries, self.exits, self.saves, rng=self.rng,
            )
            for _ in range(self.pop_size)
        ]

    def _tournament_select(self, population: List[Individual]) -> Individual:
        contenders = self.rng.sample(population, self.k_tournament)
        return min(
            contenders,
            key=lambda ind: ind.fitness if ind.fitness is not None else float("inf"),
        )

    def _crossover(self, p1: Individual, p2: Individual) -> Individual:
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

    def _mutate(self, individual: Individual) -> None:
        self.mutation_op(individual, self.K_swaps, self.rng)

    # -- Fitness evaluation ------------------------------------------------

    def _make_eval_args(self, ind: Individual) -> tuple:
        entries_arr, exits_arr, saves_arr = ind.to_position_lists()
        return (
            self.m, self.n, self.num_robots,
            entries_arr, exits_arr, saves_arr,
            self.containers_to_process, self.prob,
        )

    def _evaluate_batch(
        self, individuals: List[Individual], executor: ProcessPoolExecutor,
    ) -> None:
        if not individuals:
            return
        args_list = [self._make_eval_args(ind) for ind in individuals]
        metrics = list(executor.map(self._evaluate_fn, args_list))
        for ind, metric in zip(individuals, metrics):
            ind.fitness = metric

    # -- Metrics -----------------------------------------------------------

    @staticmethod
    def _record_metrics(best: Individual, metrics_history: dict) -> None:
        metrics_history["avg_ss_dist"].append(avg_save_to_save_distance(best))
        metrics_history["avg_local_density"].append(avg_objects_in_radius(best, R=3))
        metrics_history["avg_s_to_ex"].append(avg_save_to_nearest_entry_exit(best))

    # -- Subclass hooks (no-op) --------------------------------------------

    def _setup(
        self,
        population: List[Individual],
        executor: ProcessPoolExecutor,
    ) -> None:
        """Called once after the initial population has been evaluated."""

    def _generate_children(
        self,
        population: List[Individual],
        n_needed: int,
        executor: ProcessPoolExecutor,
    ) -> List[Individual]:
        """Default: exactly n_needed children via tournament + crossover + mutate."""
        children: List[Individual] = []
        for _ in range(n_needed):
            p1 = self._tournament_select(population)
            p2 = self._tournament_select(population)
            child = self._crossover(p1, p2)
            if self.rng.random() < self.pm:
                self._mutate(child)
            children.append(child)
        return children

    def _post_generation(
        self,
        gen: int,
        evaluated_children: List[Individual],
        executor: ProcessPoolExecutor,
    ) -> None:
        """Called after evaluating children in generation `gen`."""

    def _extra_log(self) -> Dict[str, Any]:
        """Extra fields appended to the final log."""
        return {}

    # -- Main loop ---------------------------------------------------------

    def run(
        self,
        verbose: bool = True,
        on_init_end: Optional[InitCallback] = None,
        on_generation_end: Optional[GenCallback] = None,
    ) -> Tuple[Individual, Dict[str, Any]]:
        """
        Run the evolution. Returns (best_overall, log).

        log: {best_fitness_per_gen, mean_fitness_per_gen,
              avg_ss_dist, avg_local_density, avg_s_to_ex, **_extra_log()}
        """
        population = self._init_population()

        best_fitness_per_gen: List[int] = []
        mean_fitness_per_gen: List[float] = []
        metrics_history: Dict[str, List[float]] = {
            "avg_ss_dist": [], "avg_local_density": [], "avg_s_to_ex": [],
        }

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            if verbose:
                print("Evaluating initial population...")
            self._evaluate_batch(population, executor)

            best_overall = min(population, key=lambda ind: ind.fitness)
            avg_f0 = sum(ind.fitness for ind in population) / len(population)

            best_fitness_per_gen.append(best_overall.fitness)
            mean_fitness_per_gen.append(avg_f0)
            self._record_metrics(best_overall, metrics_history)

            if verbose:
                print(f"Generation 0: best = {best_overall.fitness}, avg = {avg_f0:.2f}")

            self._setup(population, executor)
            if on_init_end:
                on_init_end(best_overall)

            for gen in range(1, self.generations + 1):
                sorted_pop = sorted(population, key=lambda ind: ind.fitness)
                elites = [ind.copy() for ind in sorted_pop[: self.elitism_count]]
                n_needed = self.pop_size - self.elitism_count

                children = self._generate_children(population, n_needed, executor)
                self._evaluate_batch(children, executor)
                population = elites + children

                best_in_gen = min(population, key=lambda ind: ind.fitness)
                if best_in_gen.fitness < best_overall.fitness:
                    best_overall = best_in_gen.copy()

                avg_fitness = sum(ind.fitness for ind in population) / len(population)
                best_fitness_per_gen.append(best_overall.fitness)
                mean_fitness_per_gen.append(avg_fitness)
                self._record_metrics(best_in_gen, metrics_history)

                self._post_generation(gen, children, executor)

                if on_generation_end:
                    on_generation_end(gen, best_in_gen, best_overall)

                if verbose:
                    print(
                        f"Generation {gen}: "
                        f"best_overall = {best_overall.fitness}, "
                        f"avg = {avg_fitness:.2f}"
                    )

        log: Dict[str, Any] = {
            "best_fitness_per_gen": best_fitness_per_gen,
            "mean_fitness_per_gen": mean_fitness_per_gen,
            **metrics_history,
            **self._extra_log(),
        }
        return best_overall, log
