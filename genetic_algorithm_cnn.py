"""
CNN-управляемый генетический алгоритм для оптимизации конфигурации склада.

Фазы:
  1. Накопление (поколения 1..n_accumulate): случайная мутация, m_swaps свапов
     с оценкой fitness после каждого свапа — каждый свап даёт одну чистую запись
     в replay buffer.
  2. Обучение WarehouseCNN на накопленных данных (после поколения n_accumulate).
  3. Инференс+дообучение (поколения n_accumulate+1..generations):
     CNN-управляемая мутация, те же m_swaps свапов с промежуточными оценками —
     данные продолжают накапливаться в буфер.
     Каждые retrain_interval поколений — дообучение.

Честное сравнение с GA: m_swaps CNN-GA = K_swaps обычного ГА.
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
from simulation import evaluate_configuration


# ---------------------------------------------------------------------------
# Кодировка типов ячеек
# ---------------------------------------------------------------------------

CELL_TYPES: List[str] = [ROAD, ENTRY, EXIT, SAVE]  # порядок каналов one-hot


# ---------------------------------------------------------------------------
# Архитектура CNN
# ---------------------------------------------------------------------------

class WarehouseCNN(nn.Module):
    """
    Лёгкая CNN: (B, 4, m, n) → (B, m, n).

    Выход p[i,j] ∈ [0,1] — вероятность того, что обмен ячейки (i,j)
    с другой ячейкой улучшит fitness конфигурации.

    Архитектура (рецептивное поле 7×7, ~4 000 параметров):
        Conv(4→16,  3×3) + BN + ReLU
        Conv(16→16, 3×3) + BN + ReLU
        Conv(16→8,  3×3) + BN + ReLU
        Conv(8→1,   1×1) + Sigmoid
    """

    def __init__(self, m: int, n: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.m = m
        self.n = n
        self.net = nn.Sequential(
            nn.Conv2d(4,  16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(16,  8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8,   1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)  # (B, m, n)


# ---------------------------------------------------------------------------
# CNN-управляемый ГА
# ---------------------------------------------------------------------------

class CNNGuidedGA:
    """
    Генетический алгоритм с CNN-управляемой мутацией.

    Параметры
    ---------
    m, n               : размер сетки склада
    entries/exits/saves: количество ячеек каждого типа
    num_robots         : количество роботов
    containers_to_process, prob : параметры симуляции
    pop_size           : размер популяции
    generations        : число поколений
    k_tournament       : размер турнира
    elitism_count      : число элит
    pm                 : вероятность мутации
    K_swaps            : K_swaps обычного ГА при сравнении (не используется напрямую,
                         только для документирования честного сравнения)
    m_swaps            : число свапов за одну мутацию в обеих фазах
                         (= K_swaps при честном сравнении)
    n_accumulate       : поколений фазы накопления (случайная мутация)
    retrain_interval   : дообучение каждые N поколений после начального обучения
    n_train_epochs     : эпох за одно обучение
    min_buffer_size    : минимум записей в буфере для запуска обучения
    max_buffer_size    : максимальный размер буфера (deque)
    batch_size         : батч для обучения CNN
    rng_seed           : seed для воспроизводимости
    n_jobs             : параллелизм оценки (−1 = cpu_count)
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
        pop_size: int = 40,
        generations: int = 80,
        k_tournament: int = 3,
        elitism_count: int = 2,
        pm: float = 0.2,
        K_swaps: int = 3,
        m_swaps: Optional[int] = None,
        n_accumulate: int = 20,
        retrain_interval: int = 5,
        n_train_epochs: int = 30,
        min_buffer_size: int = 100,
        max_buffer_size: int = 10_000,
        batch_size: int = 32,
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

        # Параметры CNN
        self.n_accumulate = n_accumulate
        self.retrain_interval = retrain_interval
        self.n_train_epochs = n_train_epochs
        self.min_buffer_size = min_buffer_size
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size

        self.n_jobs = os.cpu_count() if n_jobs == -1 else n_jobs

        # Воспроизводимость
        torch.manual_seed(rng_seed)
        self.rng = random.Random(rng_seed)
        self.np_rng = np.random.RandomState(rng_seed + 1000)

        # CNN и оптимизатор
        self.cnn = WarehouseCNN(m, n)
        self.optimizer = torch.optim.Adam(
            self.cnn.parameters(), lr=1e-3, weight_decay=1e-4,
        )
        self.cnn_trained: bool = False

        # Replay buffer: (grid_encoded_tensor, pos_a, pos_b, label)
        # grid_encoded — (1, 4, m, n) float32; pos — (row, col); label — 0/1
        self.buffer: deque = deque(maxlen=max_buffer_size)

    # -----------------------------------------------------------------------
    # Кодировка сетки
    # -----------------------------------------------------------------------

    def _encode_grid(self, grid: np.ndarray) -> torch.Tensor:
        """One-hot кодирование: (m, n) object-array → (1, 4, m, n) float32 тензор."""
        encoded = np.zeros((4, self.m, self.n), dtype=np.float32)
        for ch, ct in enumerate(CELL_TYPES):
            encoded[ch] = (grid == ct).astype(np.float32)
        return torch.from_numpy(encoded).unsqueeze(0)  # (1, 4, m, n)

    # -----------------------------------------------------------------------
    # Популяция
    # -----------------------------------------------------------------------

    def _init_population(self) -> List[Individual]:
        return [
            Individual.random_individual(
                self.m, self.n, self.entries, self.exits, self.saves, rng=self.rng
            )
            for _ in range(self.pop_size)
        ]

    # -----------------------------------------------------------------------
    # Селекция и скрещивание (идентично GeneticAlgorithm)
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Оценка пригодности
    # -----------------------------------------------------------------------

    def _make_eval_args(self, ind: Individual) -> tuple:
        entries_arr, exits_arr, saves_arr = ind.to_position_lists()
        return (
            self.m, self.n, self.num_robots,
            entries_arr, exits_arr, saves_arr,
            self.containers_to_process, self.prob,
        )

    def _evaluate_batch(
        self, individuals: List[Individual], executor: ProcessPoolExecutor
    ) -> None:
        """Параллельная оценка списка особей. evaluate_configuration использует
        random.Random(0) → детерминированный результат для одной конфигурации."""
        if not individuals:
            return
        args_list = [self._make_eval_args(ind) for ind in individuals]
        metrics = list(executor.map(evaluate_configuration, args_list))
        for ind, metric in zip(individuals, metrics):
            ind.fitness = metric

    # -----------------------------------------------------------------------
    # Мутация: один свап (используется в обеих фазах)
    # -----------------------------------------------------------------------

    def _single_swap_random(
        self, individual: Individual
    ) -> Optional[Tuple[tuple, tuple]]:
        """Один случайный своп среди не-дорожных ячеек разных типов."""
        coords = [tuple(p) for p in np.argwhere(individual.grid != ROAD)]
        if len(coords) < 2:
            return None
        pos_a = coords[self.rng.randrange(len(coords))]
        type_a = individual.grid[pos_a]
        candidates = [c for c in coords if individual.grid[c] != type_a]
        if not candidates:
            return None
        pos_b = candidates[self.rng.randrange(len(candidates))]
        individual.grid[pos_a], individual.grid[pos_b] = (
            individual.grid[pos_b],
            individual.grid[pos_a],
        )
        return pos_a, pos_b

    def _single_swap_cnn(
        self, individual: Individual
    ) -> Optional[Tuple[tuple, tuple]]:
        """
        Один CNN-guided своп между ячейками разных типов.

        CNN применяется к текущей сетке → карта вероятностей p.
        Первый узел сэмплируется пропорционально p, затем маскируются
        ячейки того же типа и второй узел выбирается из оставшихся.
        """
        coords = [tuple(p) for p in np.argwhere(individual.grid != ROAD)]
        if len(coords) < 2:
            return None

        self.cnn.eval()
        grid_tensor = self._encode_grid(individual.grid)
        with torch.no_grad():
            p_map: np.ndarray = self.cnn(grid_tensor).squeeze(0).numpy()  # (m, n)

        weights = np.array(
            [max(float(p_map[c[0], c[1]]), 1e-7) for c in coords],
            dtype=np.float64,
        )
        weights /= weights.sum()

        # Выбрать первую ячейку
        idx_a = self.np_rng.choice(len(coords), p=weights)
        pos_a: tuple = coords[idx_a]
        type_a = individual.grid[pos_a]

        # Маска: обнулить веса ячеек того же типа
        mask = np.array([individual.grid[c] != type_a for c in coords])
        weights_b = weights * mask
        if weights_b.sum() == 0:
            return None
        weights_b /= weights_b.sum()

        idx_b = self.np_rng.choice(len(coords), p=weights_b)
        pos_b: tuple = coords[idx_b]

        individual.grid[pos_a], individual.grid[pos_b] = (
            individual.grid[pos_b],
            individual.grid[pos_a],
        )
        return pos_a, pos_b

    # -----------------------------------------------------------------------
    # Обучение CNN
    # -----------------------------------------------------------------------

    def _augment_buffer(
        self, data: list
    ) -> list:
        """Аугментация ×4 через симметрии сетки (H-flip, V-flip, HV-flip)."""
        augmented = list(data)
        for grid_enc, pos_a, pos_b, label in data:
            # grid_enc: (1, 4, m, n)
            for hflip, vflip in [(True, False), (False, True), (True, True)]:
                g = grid_enc
                pa_r, pa_c = pos_a
                pb_r, pb_c = pos_b
                if hflip:
                    g = g.flip(-1)              # flip по столбцам
                    pa_c = self.n - 1 - pa_c
                    pb_c = self.n - 1 - pb_c
                if vflip:
                    g = g.flip(-2)              # flip по строкам
                    pa_r = self.m - 1 - pa_r
                    pb_r = self.m - 1 - pb_r
                augmented.append((g, (pa_r, pa_c), (pb_r, pb_c), label))
        return augmented

    def _train_cnn(self) -> List[float]:
        """
        Обучить (или дообучить) CNN на текущем replay buffer.

        Функция потерь — BCE только по двум позициям свапа:
            loss = BCE(p[pos_a], label) + BCE(p[pos_b], label)

        Если буфер меньше min_buffer_size — возвращает [].
        """
        if len(self.buffer) < self.min_buffer_size:
            return []

        data = self._augment_buffer(list(self.buffer))
        epoch_losses: List[float] = []

        # Динамический pos_weight для компенсации дисбаланса классов
        n_pos = sum(1 for item in data if item[3] == 1)
        n_neg = len(data) - n_pos
        pos_weight = (n_neg / max(n_pos, 1)) ** 0.5

        self.cnn.train()

        for _ in range(self.n_train_epochs):
            self.rng.shuffle(data)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(data), self.batch_size):
                batch = data[start : start + self.batch_size]
                if not batch:
                    continue

                # Тензоры батча
                grids = torch.cat(
                    [item[0] for item in batch], dim=0
                )  # (B, 4, m, n)
                pos_a_list = [item[1] for item in batch]
                pos_b_list = [item[2] for item in batch]
                labels = torch.tensor(
                    [item[3] for item in batch], dtype=torch.float32
                )  # (B,)

                self.optimizer.zero_grad()
                p_maps = self.cnn(grids)  # (B, m, n)

                B = len(batch)
                # Извлекаем значения в позициях двух свапнутых ячеек
                p_a = torch.stack(
                    [p_maps[i, pos_a_list[i][0], pos_a_list[i][1]] for i in range(B)]
                )
                p_b = torch.stack(
                    [p_maps[i, pos_b_list[i][0], pos_b_list[i][1]] for i in range(B)]
                )

                w = torch.where(labels == 1, pos_weight, 1.0)
                loss = F.binary_cross_entropy(p_a, labels, weight=w) + \
                    F.binary_cross_entropy(p_b, labels, weight=w)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            epoch_losses.append(epoch_loss / max(n_batches, 1))

        self.cnn.eval()
        return epoch_losses

    # -----------------------------------------------------------------------
    # Основной цикл эволюции
    # -----------------------------------------------------------------------

    def run(
        self, verbose: bool = True
    ) -> Tuple[Individual, Dict[str, Any]]:
        """
        Запустить CNN-управляемый ГА.

        Возвращает
        ----------
        best_individual : Individual
            Лучшая найденная конфигурация.
        log : dict со следующими ключами:
            best_fitness_per_gen         — лучший fitness по поколениям (gen 0..generations)
            mean_fitness_per_gen         — средний fitness
            successful_mutation_rate_per_gen — доля успешных мутаций (None для gen 0)
            cnn_loss_per_epoch           — loss CNN по всем эпохам обучения
        """
        population = self._init_population()

        best_fitness_per_gen: List[int] = []
        mean_fitness_per_gen: List[float] = []
        successful_mutation_rate_per_gen: List[Optional[float]] = []
        cnn_loss_per_epoch: List[float] = []

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:

            # ----------------------------------------------------------------
            # Поколение 0: начальная оценка
            # ----------------------------------------------------------------
            if verbose:
                print("CNN-GA: оценка начальной популяции...")
            self._evaluate_batch(population, executor)

            best_overall = min(population, key=lambda ind: ind.fitness)
            avg_f0 = sum(ind.fitness for ind in population) / len(population)

            best_fitness_per_gen.append(best_overall.fitness)
            mean_fitness_per_gen.append(avg_f0)
            successful_mutation_rate_per_gen.append(None)

            if verbose:
                print(
                    f"  Поколение 0: best={best_overall.fitness}, avg={avg_f0:.1f}"
                )

            # ----------------------------------------------------------------
            # Основной цикл
            # ----------------------------------------------------------------
            for gen in range(1, self.generations + 1):
                in_accumulation = gen <= self.n_accumulate
                phase = "Накопление" if in_accumulation else "Инференс "

                sorted_pop = sorted(population, key=lambda ind: ind.fitness)
                elites = [ind.copy() for ind in sorted_pop[: self.elitism_count]]

                # ---- Генерация детей с пометкой тех, кто будет мутировать --
                children: List[Individual] = []
                to_mutate_idx: List[int] = []  # индексы в children

                while len(elites) + len(children) < self.pop_size:
                    p1 = self._tournament_select(population)
                    p2 = self._tournament_select(population)
                    child = self._crossover(p1, p2)
                    if self.rng.random() < self.pm:
                        to_mutate_idx.append(len(children))
                    children.append(child)

                # ---- Разделяем детей на мутируемых и нет ------------------
                to_mutate_set = set(to_mutate_idx)
                mutated   = [children[i] for i in to_mutate_idx]
                untouched = [children[i] for i in range(len(children))
                             if i not in to_mutate_set]

                # Немутируемые оцениваем один раз
                self._evaluate_batch(untouched, executor)

                # Начальная оценка мутируемых (перед первым свапом)
                self._evaluate_batch(mutated, executor)

                # ---- m_swaps последовательных свапов с оценкой после каждого
                n_successful = 0
                n_swap_records = 0

                for _ in range(self.m_swaps):
                    # Запоминаем состояние ДО этого свапа
                    pre_states = [
                        (self._encode_grid(c.grid), c.fitness) for c in mutated
                    ]

                    # Применяем по одному свапу к каждому мутируемому ребёнку
                    swaps = []
                    for child in mutated:
                        if in_accumulation or not self.cnn_trained:
                            swap = self._single_swap_random(child)
                        else:
                            swap = self._single_swap_cnn(child)
                        swaps.append(swap)

                    # Оцениваем после свапа
                    self._evaluate_batch(mutated, executor)

                    # Записываем в буфер — одна чистая запись на свап
                    for child, swap, (pre_enc, pre_fit) in zip(
                        mutated, swaps, pre_states
                    ):
                        if swap is None:
                            continue
                        label = 1 if child.fitness < pre_fit else 0
                        n_successful += label
                        n_swap_records += 1
                        self.buffer.append((pre_enc, swap[0], swap[1], label))

                mut_rate: Optional[float] = (
                    n_successful / n_swap_records if n_swap_records > 0 else None
                )
                successful_mutation_rate_per_gen.append(mut_rate)

                # ---- Обновление популяции ----------------------------------
                population = elites + children

                best_in_gen = min(population, key=lambda ind: ind.fitness)
                if best_in_gen.fitness < best_overall.fitness:
                    best_overall = best_in_gen.copy()

                avg_fitness = sum(ind.fitness for ind in population) / len(population)
                best_fitness_per_gen.append(best_overall.fitness)
                mean_fitness_per_gen.append(avg_fitness)

                # ---- Обучение CNN ------------------------------------------
                if gen == self.n_accumulate:
                    if verbose:
                        print(
                            f"  [Обучение CNN] buffer={len(self.buffer)} записей..."
                        )
                    losses = self._train_cnn()
                    cnn_loss_per_epoch.extend(losses)
                    if losses:
                        self.cnn_trained = True
                        if verbose:
                            print(
                                f"  CNN обучена: {len(losses)} эпох, "
                                f"последний loss={losses[-1]:.4f}"
                            )
                    else:
                        if verbose:
                            print(
                                "  Буфер слишком мал для обучения — "
                                "продолжаем случайную мутацию."
                            )

                elif (
                    not in_accumulation
                    and (gen - self.n_accumulate) % self.retrain_interval == 0
                ):
                    if verbose:
                        print(
                            f"  [Дообучение CNN] поколение {gen}, "
                            f"buffer={len(self.buffer)}..."
                        )
                    losses = self._train_cnn()
                    cnn_loss_per_epoch.extend(losses)
                    if losses and not self.cnn_trained:
                        self.cnn_trained = True

                # ---- Вывод прогресса ---------------------------------------
                if verbose:
                    mut_str = (
                        f"mut_ok={mut_rate:.1%}" if mut_rate is not None else "no mut"
                    )
                    print(
                        f"  [{phase}] gen {gen:3d}: "
                        f"best_overall={best_overall.fitness:7d}, "
                        f"avg={avg_fitness:8.1f}, "
                        f"{mut_str}, "
                        f"buf={len(self.buffer)}"
                    )

        log: Dict[str, Any] = {
            "best_fitness_per_gen": best_fitness_per_gen,
            "mean_fitness_per_gen": mean_fitness_per_gen,
            "successful_mutation_rate_per_gen": successful_mutation_rate_per_gen,
            "cnn_loss_per_epoch": cnn_loss_per_epoch,
        }
        return best_overall, log
