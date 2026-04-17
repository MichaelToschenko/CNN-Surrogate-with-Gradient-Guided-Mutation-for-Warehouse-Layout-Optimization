"""
CNN-управляемый генетический алгоритм для оптимизации конфигурации склада.

Ключевые механизмы (отличия от классического ГА):
  1. **Surrogate Pre-screening (Oversampling)** — генерируем oversample_factor×
     больше детей, CNN-суррогат ранжирует их, на дорогую симуляцию отправляем
     только лучших. Число реальных симуляционных оценок — то же, что у ГА.
  2. **Relocation Mutation** — вместо свопа двух не-дорожных ячеек, перемещаем
     ячейку на позицию ROAD. Градиент суррогата показывает, куда лучше
     переместить. Пространство ~7 400 ходов vs ~325 свопов.
  3. **Раннее обучение + warm-start** — n_accumulate=5 (не 20), буфер
     прогревается дополнительными случайными конфигурациями на gen 0.

Фазы:
  1. Warm-start (gen 0): оцениваем начальную популяцию + warmstart_extra
     случайных конфигураций (только для буфера, не в популяцию).
  2. Накопление (gen 1..n_accumulate): случайная мутация, данные копятся.
  3. Обучение FitnessSurrogate (регрессия MSE) на gen n_accumulate.
  4. Инференс+дообучение (gen n_accumulate+1..generations):
     oversampling с суррогатной фильтрацией + relocation mutation.

Честное сравнение с GA: pop_size, generations, pm, elitism, crossover — те же.
"""

import os
import random
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import ENTRY, EXIT, ROAD, SAVE
from individual import Individual
from metrics import (
    avg_save_to_save_distance,
    avg_objects_in_radius,
    avg_save_to_nearest_entry_exit,
)
from simulation import evaluate_configuration


# ---------------------------------------------------------------------------
# Кодировка типов ячеек: 4 канала
# ---------------------------------------------------------------------------

CELL_TYPES: List[str] = [ROAD, ENTRY, EXIT, SAVE]


# ---------------------------------------------------------------------------
# Архитектура CNN-суррогата fitness
# ---------------------------------------------------------------------------

class FitnessSurrogate(nn.Module):
    """
    CNN-суррогат: (B, 4, m, n) → (B,) скалярный fitness.

    4-канальный вход (ROAD, ENTRY, EXIT, SAVE).
    Dilated свёртки расширяют рецептивное поле до 17×17 — покрывает 15×15.

    Архитектура — per-cell scoring + sum:
        Conv(4→32,  3×3, d=1) + BN + ReLU + Drop    RF:  3
        Conv(32→32, 3×3, d=1) + BN + ReLU + Drop    RF:  5
        Conv(32→16, 3×3, d=2) + BN + ReLU + Drop    RF:  9
        Conv(16→8,  3×3, d=4) + BN + ReLU           RF: 17
        Per-cell head: Conv(8→16, 1×1) + ReLU + Conv(16→1, 1×1)
        SUM over spatial dims → scalar fitness (B,)

    Без global pooling: градиент по каждой ячейке — прямой вклад.
    ~15 000 параметров.
    """

    def __init__(self, m: int, n: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.m, self.n = m, n
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(32, 16, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(16, 8, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.cell_head = nn.Sequential(
            nn.Conv2d(8, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)                 # (B, 8, m, n)
        cell_scores = self.cell_head(feats)      # (B, 1, m, n)
        return cell_scores.sum(dim=[1, 2, 3])    # (B,)


# ---------------------------------------------------------------------------
# CNN-управляемый ГА
# ---------------------------------------------------------------------------

class CNNGuidedGA:
    """
    Генетический алгоритм с суррогатным oversampling и relocation mutation.

    Параметры
    ---------
    m, n               : размер сетки склада
    entries/exits/saves : количество ячеек каждого типа
    num_robots          : количество роботов
    containers_to_process, prob : параметры симуляции
    pop_size            : размер популяции
    generations         : число поколений
    k_tournament        : размер турнира
    elitism_count       : число элит
    pm                  : вероятность мутации
    K_swaps             : K_swaps обычного ГА (для документирования)
    m_swaps             : число свапов/relocations за одну мутацию (= K_swaps)
    n_accumulate        : поколений фазы накопления
    retrain_interval    : дообучение каждые N поколений
    n_train_epochs      : эпох за одно обучение
    min_buffer_size     : минимум записей для запуска обучения
    max_buffer_size     : максимальный размер буфера
    batch_size          : батч для обучения
    min_r2              : минимальный R² для использования суррогата
    oversample_factor   : во сколько раз больше кандидатов генерировать
    warmstart_extra     : дополнительных случайных конфигураций для буфера на gen 0
    rng_seed            : seed для воспроизводимости
    n_jobs              : параллелизм оценки (−1 = cpu_count)
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
        rng_seed: int = 42,
        n_jobs: int = -1,
    ) -> None:
        # Параметры склада и симуляции
        self.m, self.n = m, n
        self.entries, self.exits, self.saves = entries, exits, saves
        self.num_robots = num_robots
        self.containers_to_process = containers_to_process
        self.prob = prob

        # Параметры ГА
        self.pop_size = pop_size
        self.generations = generations
        self.k_tournament = k_tournament
        self.elitism_count = elitism_count
        self.pm = pm
        self.K_swaps = K_swaps
        self.m_swaps = m_swaps if m_swaps is not None else K_swaps

        # Параметры суррогата
        self.n_accumulate = n_accumulate
        self.retrain_interval = retrain_interval
        self.n_train_epochs = n_train_epochs
        self.min_buffer_size = min_buffer_size
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.min_r2 = min_r2

        # Параметры oversampling и warm-start
        self.oversample_factor = oversample_factor
        self.warmstart_extra = warmstart_extra

        self.n_jobs = os.cpu_count() if n_jobs == -1 else n_jobs

        # Воспроизводимость
        torch.manual_seed(rng_seed)
        self.rng = random.Random(rng_seed)
        # CNN-суррогат и оптимизатор
        self.surrogate = FitnessSurrogate(m, n)
        self.optimizer = torch.optim.Adam(
            self.surrogate.parameters(), lr=1e-3, weight_decay=1e-4,
        )
        self.surrogate_trained: bool = False

        # Replay buffer: (grid_encoded_tensor, fitness_value)
        self.buffer: deque = deque(maxlen=max_buffer_size)

        # Нормализация fitness
        self.fit_mean: float = 0.0
        self.fit_std: float = 1.0

    # -------------------------------------------------------------------
    # Кодировка сетки
    # -------------------------------------------------------------------

    def _encode_grid(self, grid: np.ndarray) -> torch.Tensor:
        """One-hot 4-канальная кодировка: (m, n) → (1, 4, m, n) float32."""
        encoded = np.zeros((4, self.m, self.n), dtype=np.float32)
        for ch, ct in enumerate(CELL_TYPES):
            encoded[ch] = (grid == ct).astype(np.float32)
        return torch.from_numpy(encoded).unsqueeze(0)  # (1, 4, m, n)

    # -------------------------------------------------------------------
    # Буфер
    # -------------------------------------------------------------------

    def _record_to_buffer(self, individuals: List[Individual]) -> None:
        """Записать оценённых особей в буфер."""
        for ind in individuals:
            if ind.fitness is not None:
                self.buffer.append((self._encode_grid(ind.grid), ind.fitness))

    # -------------------------------------------------------------------
    # Популяция
    # -------------------------------------------------------------------

    def _init_population(self) -> List[Individual]:
        return [
            Individual.random_individual(
                self.m, self.n, self.entries, self.exits, self.saves, rng=self.rng,
            )
            for _ in range(self.pop_size)
        ]

    # -------------------------------------------------------------------
    # Пространственные метрики
    # -------------------------------------------------------------------

    @staticmethod
    def _record_metrics(best: Individual, metrics_history: dict) -> None:
        """Записать пространственные метрики лучшей особи поколения."""
        metrics_history["avg_ss_dist"].append(avg_save_to_save_distance(best))
        metrics_history["avg_local_density"].append(avg_objects_in_radius(best, R=3))
        metrics_history["avg_s_to_ex"].append(avg_save_to_nearest_entry_exit(best))

    # -------------------------------------------------------------------
    # Селекция и скрещивание (идентично GA)
    # -------------------------------------------------------------------

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

    # -------------------------------------------------------------------
    # Оценка пригодности
    # -------------------------------------------------------------------

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
        metrics = list(executor.map(evaluate_configuration, args_list))
        for ind, metric in zip(individuals, metrics):
            ind.fitness = metric

    # -------------------------------------------------------------------
    # Суррогатные предсказания (батчевые)
    # -------------------------------------------------------------------

    def _surrogate_predict_batch(
        self, individuals: List[Individual],
    ) -> np.ndarray:
        """Предсказать fitness для списка особей через суррогат."""
        self.surrogate.eval()
        tensors = torch.cat(
            [self._encode_grid(ind.grid) for ind in individuals], dim=0,
        )
        with torch.no_grad():
            preds = self.surrogate(tensors).numpy()
        # Денормализация
        return preds * self.fit_std + self.fit_mean

    # -------------------------------------------------------------------
    # Мутация: случайный relocation (фаза накопления)
    # -------------------------------------------------------------------

    def _single_relocate_random(self, individual: Individual) -> None:
        """Один случайный relocation: не-дорожная ячейка → позиция ROAD."""
        non_road = [tuple(p) for p in np.argwhere(individual.grid != ROAD)]
        road_pos = [tuple(p) for p in np.argwhere(individual.grid == ROAD)]
        if not non_road or not road_pos:
            return
        src = non_road[self.rng.randrange(len(non_road))]
        dst = road_pos[self.rng.randrange(len(road_pos))]
        individual.grid[dst] = individual.grid[src]
        individual.grid[src] = ROAD

    def _mutate_random(self, individual: Individual) -> None:
        """m_swaps случайных relocations."""
        for _ in range(self.m_swaps):
            self._single_relocate_random(individual)

    # -------------------------------------------------------------------
    # Мутация: gradient-guided relocation
    # -------------------------------------------------------------------

    def _mutate_relocate(self, individual: Individual) -> int:
        """
        Градиентно-направленное перемещение ячеек на позиции ROAD.

        Для каждого из m_swaps шагов:
          1. Backward pass через суррогат → градиент (4, m, n)
          2. Для каждой не-дорожной ячейки и каждой road-позиции,
             оценить score = grad[ch, old] - grad[ch, new]
             (перемещение из «плохой» в «хорошую» позицию)
          3. Top-N кандидатов → верификация батчевым forward pass
          4. Применить лучший ход

        Возвращает число фактически применённых перемещений.
        """
        n_applied = 0

        for _ in range(self.m_swaps):
            non_road_coords = [
                tuple(p) for p in np.argwhere(individual.grid != ROAD)
            ]
            road_coords = [
                tuple(p) for p in np.argwhere(individual.grid == ROAD)
            ]
            if len(non_road_coords) < 1 or len(road_coords) < 1:
                break

            self.surrogate.eval()
            grid_tensor = self._encode_grid(individual.grid)
            grid_tensor = grid_tensor.detach().requires_grad_(True)

            pred_current = self.surrogate(grid_tensor)
            pred_current.backward()

            grad = grid_tensor.grad.squeeze(0).numpy()  # (4, m, n)

            # --- Быстрое ранжирование через градиент ---
            # Для перемещения ячейки типа ch из pos_old в pos_new:
            #   Δf ≈ -grad[ch, old] + grad[road, old] + grad[ch, new] - grad[road, new]
            #   score = -Δf = (grad[ch, old] - grad[ch, new])
            #                + (grad[road, new] - grad[road, old])
            # Положительный score = предсказанное улучшение (уменьшение fitness)
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

            # Отобрать top-N по gradient score
            candidates.sort(key=lambda x: x[0], reverse=True)
            n_top = min(len(candidates), 30)
            top_candidates = candidates[:n_top]

            # --- Верификация суррогатом (batched forward) ---
            base_enc = self._encode_grid(individual.grid)  # (1, 4, m, n)
            batch = base_enc.expand(n_top, -1, -1, -1).clone()

            for idx, (_, pos_old, pos_new, cell_type) in enumerate(top_candidates):
                ch = CELL_TYPES.index(cell_type)
                ch_road = CELL_TYPES.index(ROAD)
                # Убрать ячейку из старой позиции (стала ROAD)
                batch[idx, ch, pos_old[0], pos_old[1]] = 0
                batch[idx, ch_road, pos_old[0], pos_old[1]] = 1
                # Поставить ячейку в новую позицию
                batch[idx, ch_road, pos_new[0], pos_new[1]] = 0
                batch[idx, ch, pos_new[0], pos_new[1]] = 1

            with torch.no_grad():
                preds_after = self.surrogate(batch).numpy()

            current_pred = pred_current.item()

            # Найти лучший ход по суррогатной верификации
            best_improvement = -float("inf")
            best_idx = -1
            for idx in range(n_top):
                improvement = current_pred - preds_after[idx]
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_idx = idx

            # Применить лучший ход (даже если improvement <= 0 — исследование)
            if best_idx >= 0:
                _, pos_old, pos_new, cell_type = top_candidates[best_idx]
                individual.grid[pos_old] = ROAD
                individual.grid[pos_new] = cell_type
                n_applied += 1

        return n_applied

    # -------------------------------------------------------------------
    # Адаптивная мутация
    # -------------------------------------------------------------------

    def _mutate_smart(self, individual: Individual) -> int:
        """
        Если суррогат обучен — relocation mutation.
        Иначе — случайные свопы (фаза накопления).
        Возвращает число применённых мутаций.
        """
        if self.surrogate_trained:
            return self._mutate_relocate(individual)
        else:
            self._mutate_random(individual)
            return self.m_swaps

    # -------------------------------------------------------------------
    # Обучение суррогата
    # -------------------------------------------------------------------

    def _augment_buffer(self, data: list) -> list:
        """Аугментация ×4 через симметрии (H-flip, V-flip, HV-flip)."""
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
        """
        Обучить/дообучить суррогат на буфере.
        Возвращает (epoch_losses, r_squared).
        """
        if len(self.buffer) < self.min_buffer_size:
            return [], 0.0

        data = list(self.buffer)

        # Z-score нормализация fitness
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

        # R² на исходном (неаугментированном) буфере
        self.surrogate.eval()
        r_squared = self._compute_r2()
        return epoch_losses, r_squared

    def _compute_r2(self) -> float:
        """Вычислить R² суррогата на текущем буфере."""
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

    # -------------------------------------------------------------------
    # Основной цикл эволюции
    # -------------------------------------------------------------------

    def run(
        self, verbose: bool = True,
    ) -> Tuple[Individual, Dict[str, Any]]:
        """
        Запустить CNN-управляемый ГА.

        Возвращает
        ----------
        best_individual : Individual
        log : dict
            best_fitness_per_gen, mean_fitness_per_gen,
            successful_mutation_rate_per_gen,
            cnn_loss_per_epoch, surrogate_r2_per_train
        """
        population = self._init_population()

        best_fitness_per_gen: List[int] = []
        mean_fitness_per_gen: List[float] = []
        cnn_loss_per_epoch: List[float] = []
        surrogate_r2_per_train: List[float] = []
        metrics_history = {"avg_ss_dist": [], "avg_local_density": [], "avg_s_to_ex": []}

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:

            # ==============================================================
            # Поколение 0: начальная оценка + warm-start буфера
            # ==============================================================
            if verbose:
                print("CNN-GA: оценка начальной популяции...")
            self._evaluate_batch(population, executor)
            self._record_to_buffer(population)

            # Warm-start: дополнительные случайные конфигурации для буфера
            if self.warmstart_extra > 0:
                if verbose:
                    print(
                        f"  Warm-start: оценка {self.warmstart_extra} "
                        f"дополнительных конфигураций для буфера..."
                    )
                warmstart = [
                    Individual.random_individual(
                        self.m, self.n,
                        self.entries, self.exits, self.saves,
                        rng=self.rng,
                    )
                    for _ in range(self.warmstart_extra)
                ]
                self._evaluate_batch(warmstart, executor)
                self._record_to_buffer(warmstart)

            best_overall = min(population, key=lambda ind: ind.fitness)
            avg_f0 = sum(ind.fitness for ind in population) / len(population)

            best_fitness_per_gen.append(best_overall.fitness)
            mean_fitness_per_gen.append(avg_f0)
            self._record_metrics(best_overall, metrics_history)

            if verbose:
                print(
                    f"  Поколение 0: best={best_overall.fitness}, "
                    f"avg={avg_f0:.1f}, buf={len(self.buffer)}"
                )

            # ==============================================================
            # Основной цикл
            # ==============================================================
            for gen in range(1, self.generations + 1):
                in_accumulation = gen <= self.n_accumulate
                use_surrogate = (
                    not in_accumulation and self.surrogate_trained
                )
                phase = "Накопление" if in_accumulation else "Инференс "

                sorted_pop = sorted(population, key=lambda ind: ind.fitness)
                elites = [ind.copy() for ind in sorted_pop[: self.elitism_count]]

                n_needed = self.pop_size - self.elitism_count

                # -- Генерация кандидатов ----------------------------------
                if use_surrogate:
                    n_generate = n_needed * self.oversample_factor
                else:
                    n_generate = n_needed

                all_children: List[Individual] = []

                for _ in range(n_generate):
                    p1 = self._tournament_select(population)
                    p2 = self._tournament_select(population)
                    child = self._crossover(p1, p2)
                    if self.rng.random() < self.pm:
                        self._mutate_smart(child)
                    all_children.append(child)

                # -- Суррогатная фильтрация --------------------------------
                if use_surrogate and len(all_children) > n_needed:
                    predicted = self._surrogate_predict_batch(all_children)
                    top_indices = np.argsort(predicted)[:n_needed]
                    children = [all_children[i] for i in top_indices]
                else:
                    children = all_children

                # -- Оценка через симуляцию --------------------------------
                self._evaluate_batch(children, executor)
                self._record_to_buffer(children)

                # -- Обновление популяции ----------------------------------
                population = elites + children

                best_in_gen = min(population, key=lambda ind: ind.fitness)
                if best_in_gen.fitness < best_overall.fitness:
                    best_overall = best_in_gen.copy()

                avg_fitness = sum(ind.fitness for ind in population) / len(
                    population
                )
                best_fitness_per_gen.append(best_overall.fitness)
                mean_fitness_per_gen.append(avg_fitness)
                self._record_metrics(best_in_gen, metrics_history)

                # -- Обучение суррогата ------------------------------------
                if gen == self.n_accumulate:
                    if verbose:
                        print(
                            f"  [Обучение суррогата] buffer={len(self.buffer)}..."
                        )
                    losses, r2 = self._train_surrogate()
                    cnn_loss_per_epoch.extend(losses)
                    if losses:
                        surrogate_r2_per_train.append(r2)
                        if r2 >= self.min_r2:
                            self.surrogate_trained = True
                            if verbose:
                                print(
                                    f"  Суррогат обучен: {len(losses)} эпох, "
                                    f"loss={losses[-1]:.4f}, R²={r2:.3f}"
                                )
                        else:
                            if verbose:
                                print(
                                    f"  R²={r2:.3f} < {self.min_r2} — "
                                    f"продолжаем случайную мутацию."
                                )

                elif (
                    not in_accumulation
                    and (gen - self.n_accumulate) % self.retrain_interval == 0
                ):
                    if verbose:
                        print(
                            f"  [Дообучение суррогата] gen {gen}, "
                            f"buffer={len(self.buffer)}..."
                        )
                    losses, r2 = self._train_surrogate()
                    cnn_loss_per_epoch.extend(losses)
                    if losses:
                        surrogate_r2_per_train.append(r2)
                        if r2 >= self.min_r2 and not self.surrogate_trained:
                            self.surrogate_trained = True
                        if verbose:
                            print(
                                f"  Дообучение: loss={losses[-1]:.4f}, "
                                f"R²={r2:.3f}"
                            )

                # -- Вывод прогресса ---------------------------------------
                if verbose:
                    surr_str = "surr" if use_surrogate else "rand"
                    print(
                        f"  [{phase}] gen {gen:3d}: "
                        f"best_overall={best_overall.fitness:7d}, "
                        f"avg={avg_fitness:8.1f}, "
                        f"buf={len(self.buffer)}, "
                        f"{surr_str}"
                    )

        log: Dict[str, Any] = {
            "best_fitness_per_gen": best_fitness_per_gen,
            "mean_fitness_per_gen": mean_fitness_per_gen,
            "cnn_loss_per_epoch": cnn_loss_per_epoch,
            "surrogate_r2_per_train": surrogate_r2_per_train,
            **metrics_history,
        }
        return best_overall, log
