import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML

import matplotlib.patches as mpatches

from constants import (
    CMAP, ANIM_LABELS,
    ContainerStatus, SaveStatus, RobotTask,
    VIS_ROAD, VIS_ROBOT_IDLE, VIS_SAVE,
    VIS_CONTAINER_WAIT_SAVE, VIS_CONTAINER_SAVED, VIS_CONTAINER_WAIT_EXIT,
    VIS_ENTRY, VIS_EXIT, VIS_ROBOT_TO_EXIT, VIS_ROBOT_TO_SAVE,
)


class Grid:
    """Отрисовка матрицы склада m*n и обновление кадров анимации."""

    def __init__(self, m: int, n: int):
        self.m = m
        self.n = n
        self.matrix = np.zeros((m, n), dtype=int)
        self.mesh = None

    def update(self, robots, containers, entries, exits, saves):
        """Перерисовать все объекты для текущего кадра."""
        self.matrix.fill(VIS_ROAD)
        for pos in entries:
            self.matrix[pos] = VIS_ENTRY
        for pos in exits:
            self.matrix[pos] = VIS_EXIT
        for save in saves:
            self.matrix[save.position] = VIS_SAVE
        for c in containers:
            if c.status in (ContainerStatus.WAITING_SAVE, ContainerStatus.ASSIGNED_SAVE):
                self.matrix[c.position] = VIS_CONTAINER_WAIT_SAVE
            elif c.status in (ContainerStatus.WAITING_EXIT, ContainerStatus.ASSIGNED_EXIT):
                self.matrix[c.position] = VIS_CONTAINER_WAIT_EXIT
            elif c.status == ContainerStatus.SAVED:
                self.matrix[c.position] = VIS_CONTAINER_SAVED
        for robot in robots:
            if robot.task in (RobotTask.TO_CASE, RobotTask.TO_SAVE_WITHOUT) or robot.is_idle():
                self.matrix[robot.position] = VIS_ROBOT_IDLE
            elif robot.task == RobotTask.TO_SAVE_WITH:
                self.matrix[robot.position] = VIS_ROBOT_TO_SAVE
            else:
                self.matrix[robot.position] = VIS_ROBOT_TO_EXIT

    def draw(self):
        """Создать фигуру и imshow для анимации."""
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.mesh = self.ax.pcolormesh(
            self.matrix, cmap=CMAP, vmin=0, vmax=10,
            edgecolors="gray", linewidth=0.5,
        )
        self.mesh.set_clip_on(False)
        self.ax.set_xlim(0, self.n)
        self.ax.set_ylim(self.m, 0)
        self.ax.set_aspect("equal")
        self.ax.tick_params(bottom=False, left=False,
                            labelbottom=False, labelleft=False)
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        # легенда
        used = [VIS_ROAD, VIS_ENTRY, VIS_EXIT, VIS_SAVE,
                VIS_ROBOT_IDLE, VIS_ROBOT_TO_SAVE, VIS_ROBOT_TO_EXIT,
                VIS_CONTAINER_WAIT_SAVE, VIS_CONTAINER_SAVED,
                VIS_CONTAINER_WAIT_EXIT]
        patches = [mpatches.Patch(color=CMAP.colors[idx], label=ANIM_LABELS[idx])
                   for idx in used]
        self.ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.02, 1),
                       fontsize=9, frameon=True)

        self.ax.set_title("Симуляция работы склада", fontsize=14, pad=10)
        self.fig.tight_layout()

    def refresh(self):
        """Передать обновлённую матрицу в отображаемое изображение."""
        if self.mesh is not None:
            self.mesh.set_array(self.matrix.ravel())
        plt.draw()


@dataclass
class Robot:
    """Робот склада, следующий по заранее вычисленному пути на один шаг за такт."""
    position: tuple
    task: RobotTask = RobotTask.IDLE
    path: list = field(default_factory=list)

    def assign_task(self, path: list, task: RobotTask):
        """Назначить новый путь и тип задачи."""
        self.path = path
        self.task = task

    def is_idle(self) -> bool:
        """Вернуть True, если у робота не осталось шагов пути."""
        return not self.path

    def move(self) -> bool:
        """Сделать один шаг по пути. Возвращает True при доставке на выход."""
        if not self.path:
            return False
        self.position = self.path.pop(0)
        if not self.path and self.task == RobotTask.TO_EXIT:
            return True
        return False


@dataclass
class Container:
    """Контейнер с позицией и статусом жизненного цикла."""
    position: tuple
    status: ContainerStatus


@dataclass
class Save:
    """Ячейка хранения и её состояние занятости."""
    position: tuple
    status: SaveStatus


class Dispatcher:
    """Логика одного такта: генерация контейнеров, назначение задач, движение роботов."""

    def __init__(self, grid: Grid, robots: List[Robot],
                 entries: list, exits: list, saves: List[Save],
                 containers_to_process: int, prob: float = 0.1,
                 rng: random.Random = None):
        self.grid = grid
        self.robots = robots
        self.entries = entries
        self.exits = exits
        self.saves = saves
        self.containers: List[Container] = []
        self.prob = prob
        self.free_exits = set(exits)
        self.containers_to_process = containers_to_process
        self.resulting_metric = 0
        self.rng = rng or random.Random()

    def step(self, plot: bool = False):
        """Выполнить один такт симуляции."""
        self.generate_containers()
        self.assign_tasks()
        self.move_robots()
        if plot:
            self.grid.update(self.robots, self.containers,
                             self.entries, self.exits, self.saves)
            self.grid.refresh()
        if self.containers_to_process > 0:
            self.resulting_metric += 1

    def generate_containers(self):
        """Стохастически порождать контейнеры во входах и выдавать команды на выход."""
        occupied_entries = {c.position for c in self.containers}

        # выдать команды на выход для сохранённых контейнеров
        for container in self.containers:
            if container.status == ContainerStatus.SAVED and self.rng.random() < self.prob:
                for save in self.saves:
                    if save.position == container.position:
                        save.status = SaveStatus.LOADED_ON
                container.status = ContainerStatus.WAITING_EXIT

        # породить новые контейнеры в пустых входных ячейках
        for entry in self.entries:
            if entry not in occupied_entries and self.rng.random() < self.prob:
                self.containers.append(
                    Container(entry, ContainerStatus.WAITING_SAVE)
                )

    def assign_tasks(self):
        """Назначить все ожидающие задачи доступным роботам."""
        idle_robots = [
            r for r in self.robots
            if r.is_idle() and r.task not in (RobotTask.TO_CASE, RobotTask.TO_SAVE_WITHOUT)
        ]
        robots_at_entry = [
            r for r in self.robots
            if r.is_idle() and r.task == RobotTask.TO_CASE
        ]
        free_saves = [s for s in self.saves if s.status == SaveStatus.FREE]
        waiting_containers = [
            c for c in self.containers
            if c.status in (ContainerStatus.WAITING_SAVE, ContainerStatus.WAITING_EXIT)
        ]

        self._dispatch_to_containers(idle_robots, robots_at_entry,
                                     free_saves, waiting_containers)
        self._handle_arrived_robots(free_saves)

    def _dispatch_to_containers(self, idle_robots, robots_at_entry,
                                free_saves, waiting_containers):
        """Для каждого необработанного контейнера найти и назначить ближайшего робота."""
        for container in waiting_containers:
            if container.status == ContainerStatus.WAITING_SAVE:
                if not free_saves:
                    continue
                if len(free_saves) <= len(robots_at_entry):
                    continue
            if not idle_robots:
                break

            robot, path_to_container = self._closest_robot(idle_robots, container.position)
            if robot is None:
                continue

            # если робот только что прибыл в хранилище с контейнером — сначала сдать его
            if robot.task == RobotTask.TO_SAVE_WITH:
                robot.task = RobotTask.IDLE
                for save in self.saves:
                    if save.position == robot.position:
                        save.status = SaveStatus.LOADED_OFF
                self.containers.append(Container(robot.position, ContainerStatus.SAVED))

            if container.status == ContainerStatus.WAITING_SAVE:
                robot.assign_task(path_to_container, RobotTask.TO_CASE)
                container.status = ContainerStatus.ASSIGNED_SAVE
            else:
                path_with_start = self.bfs_path(robot.position, container.position,
                                                include_start=True)
                robot.assign_task(path_with_start, RobotTask.TO_SAVE_WITHOUT)
                container.status = ContainerStatus.ASSIGNED_EXIT

            idle_robots.remove(robot)

    def _handle_arrived_robots(self, free_saves):
        """Обработать роботов, прибывших в пункт назначения."""
        loaded_on_positions = {
            s.position for s in self.saves if s.status == SaveStatus.LOADED_ON
        }
        for robot in self.robots:
            if robot.is_idle() and robot.task == RobotTask.TO_CASE:
                self._robot_arrived_at_entry(robot, free_saves)
            elif robot.is_idle() and robot.position in loaded_on_positions:
                self._robot_arrived_for_exit(robot)
            elif robot.is_idle() and robot.task == RobotTask.TO_SAVE_WITH:
                self._robot_arrived_at_storage(robot)

    def _robot_arrived_at_entry(self, robot, free_saves):
        """Робот прибыл к контейнеру — отправить его в хранилище."""
        save, path = self._closest_target(robot.position, free_saves,
                                          lambda s: s.position)
        if save and path:
            robot.assign_task(path, RobotTask.TO_SAVE_WITH)
            save.status = SaveStatus.WAITING
            free_saves.remove(save)

        for c in self.containers:
            if c.position == robot.position:
                self.containers.remove(c)
                break

    def _robot_arrived_for_exit(self, robot):
        """Робот прибыл в ячейку с командой на выход — отправить на выход."""
        exit_list = list(self.free_exits)
        exit_pos, path = self._closest_target(robot.position, exit_list,
                                              lambda e: e)
        if exit_pos and path:
            robot.assign_task(path, RobotTask.TO_EXIT)
            for save in self.saves:
                if save.position == robot.position:
                    save.status = SaveStatus.FREE
                    break

        for c in self.containers:
            if c.position == robot.position:
                self.containers.remove(c)
                break

    def _robot_arrived_at_storage(self, robot):
        """Робот прибыл в хранилище с контейнером — сдать контейнер."""
        robot.task = RobotTask.IDLE
        for save in self.saves:
            if save.position == robot.position:
                save.status = SaveStatus.LOADED_OFF
        self.containers.append(Container(robot.position, ContainerStatus.SAVED))

    def move_robots(self):
        """Сдвинуть каждого робота на один шаг; считать доставленные контейнеры."""
        for robot in self.robots:
            delivered = robot.move()
            if delivered:
                self.containers_to_process -= 1

    def bfs_path(self, start: tuple, goal: tuple,
                 include_start: bool = False) -> Optional[list]:
        """BFS-кратчайший путь от start до goal на текущей сетке."""
        rows, cols = self.grid.m, self.grid.n
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        queue = deque([start])
        visited = {start}
        parent = {start: None}

        while queue:
            current = queue.popleft()
            if current == goal:
                path = []
                node = goal
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path if include_start else path[1:]

            x, y = current
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)
                mat = self.grid.matrix
                blocked = (VIS_CONTAINER_WAIT_SAVE, VIS_CONTAINER_SAVED,
                           VIS_CONTAINER_WAIT_EXIT)
                if (0 <= nx < rows and 0 <= ny < cols
                        and neighbor not in visited
                        and mat[nx][ny] != VIS_ENTRY
                        and (mat[nx][ny] not in blocked or neighbor == goal)):
                    queue.append(neighbor)
                    visited.add(neighbor)
                    parent[neighbor] = current

        return None

    # -- Вспомогательные методы ---------------------------------------------

    def _closest_robot(self, candidates, target):
        """Вернуть (робот, путь) для кандидата, ближайшего к target."""
        best_robot, best_path, best_dist = None, None, float("inf")
        for robot in candidates:
            path = self.bfs_path(robot.position, target)
            if path is not None and len(path) < best_dist:
                best_dist = len(path)
                best_robot = robot
                best_path = path
        return best_robot, best_path

    def _closest_target(self, origin, targets, position_fn):
        """Вернуть (цель, путь) для цели, ближайшей к origin."""
        best_target, best_path, best_dist = None, None, float("inf")
        for t in targets:
            path = self.bfs_path(origin, position_fn(t))
            if path is not None and len(path) < best_dist:
                best_dist = len(path)
                best_target = t
                best_path = path
        return best_target, best_path


class Simulator:
    """Связывает сетку, роботов и диспетчер; запускает симуляцию."""

    def __init__(self, m: int, n: int, num_robots: int,
                 entries: int, exits: int, saves: int,
                 containers_to_process: int, prob: float,
                 are_position_sampled: bool = True,
                 rng: random.Random = None, **kwargs):
        self.rng = rng or random.Random()
        self.unique_positions: set = set()

        if are_position_sampled:
            self._init_random(m, n, num_robots, entries, exits, saves)
        else:
            self._init_fixed(m, n, num_robots, entries, exits, saves, kwargs)

        self.grid = Grid(m, n)
        self.robots = [Robot(pos) for pos in self._random_positions(num_robots, m, n)]
        self.dispatcher = Dispatcher(
            self.grid, self.robots,
            self.entries, self.exits, self.saves,
            containers_to_process, prob, rng=self.rng,
        )

    def _init_random(self, m, n, num_robots, entries, exits, saves):
        """Случайное размещение объектов на сетке."""
        assert num_robots + entries + exits + saves < 0.8 * m * n, (
            "Слишком много объектов на матрице: уменьшите их кол-во или увеличьте размер"
        )
        self.entries = self._random_positions(entries, m, n)
        self.exits = self._random_positions(exits, m, n)
        self.saves = [Save(pos, SaveStatus.FREE)
                      for pos in self._random_positions(saves, m, n)]

    def _init_fixed(self, m, n, num_robots, entries, exits, saves, kwargs):
        """Размещение объектов по заданным позициям."""
        entries_arr = kwargs.get("entries_arr")
        exits_arr = kwargs.get("exits_arr")
        saves_arr = kwargs.get("saves_arr")

        assert len(entries_arr) == entries, "Несогласованная генерация входных ячеек"
        assert len(exits_arr) == exits,     "Несогласованная генерация выходных ячеек"
        assert len(saves_arr) == saves,     "Несогласованная генерация ячеек хранения"
        assert (not set(saves_arr) & set(exits_arr)
                and not set(entries_arr) & set(exits_arr)
                and not set(saves_arr) & set(entries_arr)), (
            "Позиции инициализации пересекаются"
        )
        assert num_robots + entries + exits + saves < 0.8 * m * n, (
            "Слишком много объектов на матрице: уменьшите их кол-во или увеличьте размер"
        )

        self.entries = entries_arr
        self.exits = exits_arr
        self.saves = [Save(pos, SaveStatus.FREE) for pos in saves_arr]
        self.unique_positions = set(saves_arr) | set(entries_arr) | set(exits_arr)

    def _random_positions(self, count: int, m: int, n: int) -> list:
        """Сгенерировать count уникальных случайных позиций на сетке."""
        positions = []
        for _ in range(count):
            pos = (self.rng.randint(0, m - 1), self.rng.randint(0, n - 1))
            while pos in self.unique_positions:
                pos = (self.rng.randint(0, m - 1), self.rng.randint(0, n - 1))
            positions.append(pos)
            self.unique_positions.add(pos)
        return positions

    def run(self):
        """Запустить симуляцию до завершения (без анимации)."""
        while self.dispatcher.containers_to_process > 0:
            self.dispatcher.step()
        self.metric = self.dispatcher.resulting_metric
        self.out = None

    def run_animated(self):
        """Запустить симуляцию с анимацией matplotlib."""
        self.grid.update(self.robots, self.dispatcher.containers,
                         self.entries, self.exits, self.saves)
        self.grid.draw()
        ani = animation.FuncAnimation(
            self.grid.fig, self._tick, frames=250, interval=5000, repeat=False
        )
        self.out = HTML(ani.to_jshtml())
        plt.close(self.grid.fig)
        self.metric = self.dispatcher.resulting_metric
        if self.dispatcher.containers_to_process > 0:
            print("Не хватило кол-ва кадров (frames) для обработки текущего кол-ва контейнеров")

    def _tick(self, _=None):
        """Колбэк анимации: выполнить один такт."""
        self.dispatcher.step(plot=True)

    def print_metric(self):
        """Вывести итоговое количество шагов симуляции."""
        print("Итоговое количество шагов:", self.dispatcher.resulting_metric)


def evaluate_configuration(args: tuple) -> int:
    """Оценить конфигурацию склада через симуляцию. Вызывается из ProcessPoolExecutor.
       Вспомогательная функция для распараллеливания"""
    m, n, num_robots, entries_arr, exits_arr, saves_arr, containers_to_process, prob = args
    sim = Simulator(
        m, n, num_robots,
        len(entries_arr), len(exits_arr), len(saves_arr),
        containers_to_process, prob,
        are_position_sampled=False,
        entries_arr=entries_arr, exits_arr=exits_arr, saves_arr=saves_arr,
    )
    sim.run()
    return sim.metric
