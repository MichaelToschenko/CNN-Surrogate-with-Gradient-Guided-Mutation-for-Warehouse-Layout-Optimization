from collections import deque
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.base import BaseGA
from algorithms.mutations import random_relocate_mutation
from algorithms.surrogate import CELL_TYPES, FitnessSurrogate, encode_grid
from constants import ROAD
from individual import Individual


class CNNGuidedGA(BaseGA):
    """GA with surrogate oversampling and relocation mutation."""

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
        generations: int = 80,
        k_tournament: int = 3,
        elitism_count: int = 2,
        pm: float = 0.2,
        K_swaps: int = 3,
        m_swaps: Optional[int] = None,
        n_accumulate: int = 5,
        retrain_interval: int = 3,
        n_train_epochs: int = 30,
        min_buffer_size: int = 100,
        max_buffer_size: int = 10_000,
        batch_size: int = 32,
        min_r2: float = 0.1,
        oversample_factor: int = 3,
        warmstart_extra: int = 50,
        use_prescreening: bool = True,
        use_grad_mutation: bool = True,
        rng_seed: int = 42,
        n_jobs: int = -1,
    ) -> None:
        effective_k = m_swaps if m_swaps is not None else K_swaps

        torch.manual_seed(rng_seed)
        super().__init__(
            m=m, n=n,
            entries=entries, exits=exits, saves=saves,
            num_robots=num_robots,
            containers_to_process=containers_to_process,
            prob=prob,
            pop_size=pop_size,
            generations=generations,
            k_tournament=k_tournament,
            elitism_count=elitism_count,
            pm=pm,
            K_swaps=effective_k,
            rng_seed=rng_seed,
            n_jobs=n_jobs,
            mutation_op=random_relocate_mutation,
        )
        self.m_swaps = effective_k

        # Surrogate parameters
        self.n_accumulate = n_accumulate
        self.retrain_interval = retrain_interval
        self.n_train_epochs = n_train_epochs
        self.min_buffer_size = min_buffer_size
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.min_r2 = min_r2
        self.oversample_factor = oversample_factor
        self.warmstart_extra = warmstart_extra
        self.use_prescreening = use_prescreening
        self.use_grad_mutation = use_grad_mutation

        # CNN surrogate
        self.surrogate = FitnessSurrogate(m, n)
        self.optimizer = torch.optim.Adam(
            self.surrogate.parameters(), lr=1e-3, weight_decay=1e-4,
        )
        self.surrogate_trained: bool = False

        # Replay buffer + fitness normalization
        self.buffer: deque = deque(maxlen=max_buffer_size)
        self.fit_mean: float = 0.0
        self.fit_std: float = 1.0

        # Training logs
        self._cnn_loss_per_epoch: List[float] = []
        self._surrogate_r2_per_train: List[float] = []

    # -- Buffer ------------------------------------------------------------

    def _record_to_buffer(self, individuals: List[Individual]) -> None:
        for ind in individuals:
            if ind.fitness is not None:
                self.buffer.append((encode_grid(ind.grid, self.m, self.n), ind.fitness))

    # -- Surrogate predictions ---------------------------------------------

    def _surrogate_predict_batch(
        self, individuals: List[Individual],
    ) -> np.ndarray:
        self.surrogate.eval()
        tensors = torch.cat(
            [encode_grid(ind.grid, self.m, self.n) for ind in individuals], dim=0,
        )
        with torch.no_grad():
            preds = self.surrogate(tensors).numpy()
        return preds * self.fit_std + self.fit_mean

    # -- Gradient-guided relocation mutation -------------------------------

    def _mutate_relocate(self, individual: Individual) -> int:
        """
        Gradient-guided relocation of cells onto ROAD positions.

        For each of m_swaps steps:
          1. Backward pass through the surrogate -> gradient (4, m, n)
          2. For each non-road cell and each road position,
             score = (grad[ch, old] - grad[ch, new]) + (grad[road, new] - grad[road, old])
          3. Top-N candidates -> verification via a batched forward pass
          4. Apply the best move
        """
        n_applied = 0

        for _ in range(self.m_swaps):
            non_road_coords = [tuple(p) for p in np.argwhere(individual.grid != ROAD)]
            road_coords = [tuple(p) for p in np.argwhere(individual.grid == ROAD)]
            if len(non_road_coords) < 1 or len(road_coords) < 1:
                break

            self.surrogate.eval()
            grid_tensor = encode_grid(individual.grid, self.m, self.n)
            grid_tensor = grid_tensor.detach().requires_grad_(True)

            pred_current = self.surrogate(grid_tensor)
            pred_current.backward()

            grad = grid_tensor.grad.squeeze(0).numpy()  # (4, m, n)

            candidates: List[Tuple[float, tuple, tuple, str]] = []
            ch_road = CELL_TYPES.index(ROAD)

            for pos in non_road_coords:
                cell_type = individual.grid[pos]
                ch = CELL_TYPES.index(cell_type)
                g_old_ch = grad[ch, pos[0], pos[1]]
                g_old_road = grad[ch_road, pos[0], pos[1]]

                for road_pos in road_coords:
                    g_new_ch = grad[ch, road_pos[0], road_pos[1]]
                    g_new_road = grad[ch_road, road_pos[0], road_pos[1]]
                    score = (g_old_ch - g_new_ch) + (g_new_road - g_old_road)
                    candidates.append((score, pos, road_pos, cell_type))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[0], reverse=True)
            n_top = min(len(candidates), 30)
            top_candidates = candidates[:n_top]

            base_enc = encode_grid(individual.grid, self.m, self.n)
            batch = base_enc.expand(n_top, -1, -1, -1).clone()

            for idx, (_, pos_old, pos_new, cell_type) in enumerate(top_candidates):
                ch = CELL_TYPES.index(cell_type)
                batch[idx, ch, pos_old[0], pos_old[1]] = 0
                batch[idx, ch_road, pos_old[0], pos_old[1]] = 1
                batch[idx, ch_road, pos_new[0], pos_new[1]] = 0
                batch[idx, ch, pos_new[0], pos_new[1]] = 1

            with torch.no_grad():
                preds_after = self.surrogate(batch).numpy()

            current_pred = pred_current.item()

            best_improvement = -float("inf")
            best_idx = -1
            for idx in range(n_top):
                improvement = current_pred - preds_after[idx]
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_idx = idx

            if best_idx >= 0:
                _, pos_old, pos_new, cell_type = top_candidates[best_idx]
                individual.grid[pos_old] = ROAD
                individual.grid[pos_new] = cell_type
                n_applied += 1

        return n_applied

    # -- Adaptive mutation: overrides BaseGA._mutate -----------------------

    def _mutate(self, individual: Individual) -> None:
        """If the surrogate is trained - gradient relocation. Otherwise - base random relocate."""
        if self.surrogate_trained and self.use_grad_mutation:
            self._mutate_relocate(individual)
        else:
            super()._mutate(individual)

    # -- Surrogate training -------------------------------------------------

    def _augment_buffer(self, data: list) -> list:
        """x4 augmentation via symmetries (H-flip, V-flip, HV-flip)."""
        augmented = list(data)
        for grid_enc, fitness in data:
            for hflip, vflip in [(True, False), (False, True), (True, True)]:
                g = grid_enc
                if hflip:
                    g = g.flip(-1)
                if vflip:
                    g = g.flip(-2)
                augmented.append((g, fitness))
        return augmented

    def _train_surrogate(self) -> Tuple[List[float], float]:
        """Train/fine-tune the surrogate on the buffer. Returns (epoch_losses, r^2)."""
        if len(self.buffer) < self.min_buffer_size:
            return [], 0.0

        data = list(self.buffer)
        fitnesses = np.array([item[1] for item in data], dtype=np.float64)
        self.fit_mean = float(fitnesses.mean())
        self.fit_std = float(max(fitnesses.std(), 1.0))

        data = self._augment_buffer(data)

        epoch_losses: List[float] = []
        self.surrogate.train()

        for _ in range(self.n_train_epochs):
            self.rng.shuffle(data)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(data), self.batch_size):
                batch = data[start : start + self.batch_size]
                if not batch:
                    continue

                grids = torch.cat([item[0] for item in batch], dim=0)
                targets = torch.tensor(
                    [(item[1] - self.fit_mean) / self.fit_std for item in batch],
                    dtype=torch.float32,
                )

                self.optimizer.zero_grad()
                preds = self.surrogate(grids)
                loss = F.mse_loss(preds, targets)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            epoch_losses.append(epoch_loss / max(n_batches, 1))

        self.surrogate.eval()
        r_squared = self._compute_r2()
        return epoch_losses, r_squared

    def _compute_r2(self) -> float:
        if len(self.buffer) < 2:
            return 0.0
        with torch.no_grad():
            grids = torch.cat([item[0] for item in self.buffer], dim=0)
            targets = torch.tensor(
                [(item[1] - self.fit_mean) / self.fit_std for item in self.buffer],
                dtype=torch.float32,
            )
            preds = self.surrogate(grids)
            ss_res = ((preds - targets) ** 2).sum().item()
            ss_tot = ((targets - targets.mean()) ** 2).sum().item()
        return 1.0 - ss_res / max(ss_tot, 1e-8)

    # -- Hooks --------------------------------------------------------------

    def _setup(
        self,
        population: List[Individual],
        executor: ProcessPoolExecutor,
    ) -> None:
        """Record initial population into the buffer + warm-start."""
        self._record_to_buffer(population)

        if self.warmstart_extra > 0:
            warmstart = [
                Individual.random_individual(
                    self.m, self.n, self.entries, self.exits, self.saves, rng=self.rng,
                )
                for _ in range(self.warmstart_extra)
            ]
            self._evaluate_batch(warmstart, executor)
            self._record_to_buffer(warmstart)

    def _generate_children(
        self,
        population: List[Individual],
        n_needed: int,
        executor: ProcessPoolExecutor,
    ) -> List[Individual]:
        """Oversample + surrogate filter when the surrogate is active."""
        use_surrogate = self.surrogate_trained and self.use_prescreening
        n_generate = n_needed * self.oversample_factor if use_surrogate else n_needed

        all_children: List[Individual] = []
        for _ in range(n_generate):
            p1 = self._tournament_select(population)
            p2 = self._tournament_select(population)
            child = self._crossover(p1, p2)
            if self.rng.random() < self.pm:
                self._mutate(child)
            all_children.append(child)

        if use_surrogate and len(all_children) > n_needed:
            predicted = self._surrogate_predict_batch(all_children)
            top_indices = np.argsort(predicted)[:n_needed]
            return [all_children[i] for i in top_indices]
        return all_children

    def _post_generation(
        self,
        gen: int,
        evaluated_children: List[Individual],
        executor: ProcessPoolExecutor,
    ) -> None:
        """Record children into the buffer + train/retrain the surrogate."""
        self._record_to_buffer(evaluated_children)

        in_accumulation = gen <= self.n_accumulate

        if gen == self.n_accumulate:
            losses, r2 = self._train_surrogate()
            self._cnn_loss_per_epoch.extend(losses)
            if losses:
                self._surrogate_r2_per_train.append(r2)
                if r2 >= self.min_r2:
                    self.surrogate_trained = True
        elif (
            not in_accumulation
            and (gen - self.n_accumulate) % self.retrain_interval == 0
        ):
            losses, r2 = self._train_surrogate()
            self._cnn_loss_per_epoch.extend(losses)
            if losses:
                self._surrogate_r2_per_train.append(r2)
                if r2 >= self.min_r2 and not self.surrogate_trained:
                    self.surrogate_trained = True

    def _extra_log(self) -> Dict[str, Any]:
        return {
            "cnn_loss_per_epoch": self._cnn_loss_per_epoch,
            "surrogate_r2_per_train": self._surrogate_r2_per_train,
        }
