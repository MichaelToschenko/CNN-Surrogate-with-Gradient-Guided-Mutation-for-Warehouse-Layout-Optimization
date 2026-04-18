import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from constants import (
    CMAP, ANIM_LABELS,
    ContainerStatus, SaveStatus, RobotTask,
    VIS_ROAD, VIS_ROBOT_IDLE, VIS_SAVE,
    VIS_CONTAINER_WAIT_SAVE, VIS_CONTAINER_SAVED, VIS_CONTAINER_WAIT_EXIT,
    VIS_ENTRY, VIS_EXIT, VIS_ROBOT_TO_EXIT, VIS_ROBOT_TO_SAVE,
)


class Grid:
    """Renders the m*n warehouse matrix and updates animation frames."""

    def __init__(self, m: int, n: int):
        self.m = m
        self.n = n
        self.matrix = np.zeros((m, n), dtype=int)
        self.mesh = None

    def update(self, robots, containers, entries, exits, saves):
        """Redraw all objects for the current frame."""
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
        """Create the figure and pcolormesh for animation."""
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

        # legend
        used = [VIS_ROAD, VIS_ENTRY, VIS_EXIT, VIS_SAVE,
                VIS_ROBOT_IDLE, VIS_ROBOT_TO_SAVE, VIS_ROBOT_TO_EXIT,
                VIS_CONTAINER_WAIT_SAVE, VIS_CONTAINER_SAVED,
                VIS_CONTAINER_WAIT_EXIT]
        patches = [mpatches.Patch(color=CMAP.colors[idx], label=ANIM_LABELS[idx])
                   for idx in used]
        self.ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.02, 1),
                       fontsize=9, frameon=True)

        self.ax.set_title("Warehouse simulation", fontsize=14, pad=10)
        self.fig.tight_layout()

    def refresh(self):
        """Push the updated matrix to the displayed image."""
        if self.mesh is not None:
            self.mesh.set_array(self.matrix.ravel())
        plt.draw()


@dataclass
class Robot:
    """Warehouse robot that follows a precomputed path one step per tick."""
    position: tuple
    task: RobotTask = RobotTask.IDLE
    path: deque = field(default_factory=deque)

    def assign_task(self, path: list, task: RobotTask):
        """Assign a new path and task type."""
        self.path = deque(path)
        self.task = task

    def is_idle(self) -> bool:
        """Return True if the robot has no remaining path steps."""
        return not self.path

    def move(self) -> bool:
        """Advance one step along the path. Returns True on delivery to an exit."""
        if not self.path:
            return False
        self.position = self.path.popleft()
        if not self.path and self.task == RobotTask.TO_EXIT:
            return True
        return False


@dataclass
class Container:
    """Container with a position and lifecycle status."""
    position: tuple
    status: ContainerStatus


@dataclass
class Save:
    """Storage cell and its occupancy status."""
    position: tuple
    status: SaveStatus


class Dispatcher:
    """One-tick logic: container generation, task assignment, robot movement."""

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
        self.free_exits = exits  # all exits are always available (no capacity limit)
        self.saves_by_pos: dict = {s.position: s for s in saves}
        self.containers_to_process = containers_to_process
        self.resulting_metric = 0
        self.rng = rng or random.Random()

    def step(self, plot: bool = False):
        """Run one simulation tick."""
        self.generate_containers()
        self.assign_tasks()
        was_processing = self.containers_to_process > 0
        self.move_robots()
        if plot:
            self.grid.update(self.robots, self.containers,
                             self.entries, self.exits, self.saves)
            self.grid.refresh()
        if was_processing:
            self.resulting_metric += 1

    def generate_containers(self):
        """Stochastically spawn containers at entries and issue exit commands."""
        occupied_entries = {c.position for c in self.containers}

        # issue exit commands for stored containers
        for container in self.containers:
            if container.status == ContainerStatus.SAVED and self.rng.random() < self.prob:
                save = self.saves_by_pos.get(container.position)
                if save is not None:
                    save.status = SaveStatus.LOADED_ON
                container.status = ContainerStatus.WAITING_EXIT

        # spawn new containers in empty entry cells
        for entry in self.entries:
            if entry not in occupied_entries and self.rng.random() < self.prob:
                self.containers.append(
                    Container(entry, ContainerStatus.WAITING_SAVE)
                )

    def assign_tasks(self):
        """Assign all pending tasks to available robots."""
        free_saves = [s for s in self.saves if s.status == SaveStatus.FREE]

        # First handle robots that have arrived (finalize deliveries)
        self._handle_arrived_robots(free_saves)

        # Rebuild lists after handling arrivals
        idle_robots = [
            r for r in self.robots
            if r.is_idle() and r.task not in (RobotTask.TO_CASE, RobotTask.TO_SAVE_WITHOUT)
        ]
        robots_at_entry = [
            r for r in self.robots
            if r.is_idle() and r.task == RobotTask.TO_CASE
        ]
        waiting_containers = [
            c for c in self.containers
            if c.status in (ContainerStatus.WAITING_SAVE, ContainerStatus.WAITING_EXIT)
        ]

        self._dispatch_to_containers(idle_robots, robots_at_entry,
                                     free_saves, waiting_containers)

    def _dispatch_to_containers(self, idle_robots, robots_at_entry,
                                free_saves, waiting_containers):
        """For each unhandled container, find and assign the closest robot."""
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

            if container.status == ContainerStatus.WAITING_SAVE:
                robot.assign_task(path_to_container, RobotTask.TO_CASE)
                container.status = ContainerStatus.ASSIGNED_SAVE
            else:
                robot.assign_task(path_to_container, RobotTask.TO_SAVE_WITHOUT)
                container.status = ContainerStatus.ASSIGNED_EXIT

            idle_robots.remove(robot)

    def _handle_arrived_robots(self, free_saves):
        """Process robots that have arrived at their destination."""
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
        """Robot reached the container - send it to storage."""
        save, path = self._closest_target(robot.position, free_saves,
                                          lambda s: s.position)
        if save and path:
            robot.assign_task(path, RobotTask.TO_SAVE_WITH)
            save.status = SaveStatus.WAITING
            free_saves.remove(save)
            self.containers = [c for c in self.containers if c.position != robot.position]

    def _robot_arrived_for_exit(self, robot):
        """Robot reached a cell with an exit command - send it to an exit."""
        exit_list = self.free_exits
        exit_pos, path = self._closest_target(robot.position, exit_list,
                                              lambda e: e)
        if exit_pos and path:
            robot.assign_task(path, RobotTask.TO_EXIT)
            save = self.saves_by_pos.get(robot.position)
            if save is not None:
                save.status = SaveStatus.FREE
            self.containers = [c for c in self.containers if c.position != robot.position]

    def _robot_arrived_at_storage(self, robot):
        """Robot reached storage with a container - drop the container."""
        robot.task = RobotTask.IDLE
        save = self.saves_by_pos.get(robot.position)
        if save is not None:
            save.status = SaveStatus.LOADED_OFF
        self.containers.append(Container(robot.position, ContainerStatus.SAVED))

    def move_robots(self):
        """Advance each robot one step; count delivered containers."""
        for robot in self.robots:
            delivered = robot.move()
            if delivered:
                self.containers_to_process -= 1

    def bfs_path(self, start: tuple, goal: tuple,
                 include_start: bool = False) -> Optional[list]:
        """BFS shortest path from start to goal on the current grid."""
        rows, cols = self.grid.m, self.grid.n
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        queue = deque([start])
        visited = {start}
        parent = {start: None}
        # Logical state instead of the visual matrix:
        # entries block transit movement; waiting containers block their cell
        entries_set = set(self.entries)
        blocked_positions = {
            c.position for c in self.containers
            if c.status in (ContainerStatus.WAITING_SAVE,
                            ContainerStatus.SAVED,
                            ContainerStatus.WAITING_EXIT)
        }

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
                if (0 <= nx < rows and 0 <= ny < cols
                        and neighbor not in visited
                        and (neighbor not in entries_set or neighbor == goal)
                        and (neighbor not in blocked_positions or neighbor == goal)):
                    queue.append(neighbor)
                    visited.add(neighbor)
                    parent[neighbor] = current

        return None

    # -- Helpers ------------------------------------------------------------

    def _closest_robot(self, candidates, target):
        """Return (robot, path) for the candidate closest to target."""
        best_robot, best_path, best_dist = None, None, float("inf")
        for robot in candidates:
            path = self.bfs_path(robot.position, target)
            if path is not None and len(path) < best_dist:
                best_dist = len(path)
                best_robot = robot
                best_path = path
        return best_robot, best_path

    def _closest_target(self, origin, targets, position_fn):
        """Return (target, path) for the target closest to origin."""
        best_target, best_path, best_dist = None, None, float("inf")
        for t in targets:
            path = self.bfs_path(origin, position_fn(t))
            if path is not None and len(path) < best_dist:
                best_dist = len(path)
                best_target = t
                best_path = path
        return best_target, best_path


class Simulator:
    """Wires the grid, robots, and dispatcher together; runs the simulation."""

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
        """Random object placement on the grid."""
        assert num_robots + entries + exits + saves < 0.8 * m * n, (
            "Too many objects on the grid: reduce the count or increase the size"
        )
        self.entries = self._random_positions(entries, m, n)
        self.exits = self._random_positions(exits, m, n)
        self.saves = [Save(pos, SaveStatus.FREE)
                      for pos in self._random_positions(saves, m, n)]

    def _init_fixed(self, m, n, num_robots, entries, exits, saves, kwargs):
        """Object placement at the given positions."""
        entries_arr = kwargs.get("entries_arr")
        exits_arr = kwargs.get("exits_arr")
        saves_arr = kwargs.get("saves_arr")

        assert len(entries_arr) == entries, "Inconsistent entry-cell generation"
        assert len(exits_arr) == exits,     "Inconsistent exit-cell generation"
        assert len(saves_arr) == saves,     "Inconsistent storage-cell generation"
        assert (not set(saves_arr) & set(exits_arr)
                and not set(entries_arr) & set(exits_arr)
                and not set(saves_arr) & set(entries_arr)), (
            "Initialization positions overlap"
        )
        assert num_robots + entries + exits + saves < 0.8 * m * n, (
            "Too many objects on the grid: reduce the count or increase the size"
        )

        self.entries = entries_arr
        self.exits = exits_arr
        self.saves = [Save(pos, SaveStatus.FREE) for pos in saves_arr]
        self.unique_positions = set(saves_arr) | set(entries_arr) | set(exits_arr)

    def _random_positions(self, count: int, m: int, n: int) -> list:
        """Generate `count` unique random positions on the grid."""
        available = [(i, j) for i in range(m) for j in range(n)
                     if (i, j) not in self.unique_positions]
        selected = self.rng.sample(available, count)
        self.unique_positions.update(selected)
        return selected

    def run(self, max_steps: int = 100_000):
        """Run the simulation to completion (no animation).

        max_steps - deadlock guard: if the simulation does not finish within
        this many steps, the current (high) metric value is returned.
        """
        while (self.dispatcher.containers_to_process > 0
               and self.dispatcher.resulting_metric < max_steps):
            self.dispatcher.step()
        self.metric = self.dispatcher.resulting_metric
        self.out = None

    def run_animated(self):
        """Run the simulation with a matplotlib animation."""
        from IPython.display import HTML
        self.grid.update(self.robots, self.dispatcher.containers,
                         self.entries, self.exits, self.saves)
        self.grid.draw()
        ani = animation.FuncAnimation(
            self.grid.fig, self._tick, frames=250, interval=200, repeat=False
        )
        self.out = HTML(ani.to_jshtml())
        plt.close(self.grid.fig)
        self.metric = self.dispatcher.resulting_metric
        if self.dispatcher.containers_to_process > 0:
            print("Not enough animation frames to process all containers")

    def _tick(self, _=None):
        """Animation callback: run one tick."""
        self.dispatcher.step(plot=True)

    def print_metric(self):
        """Print the final simulation step count."""
        print("Total steps:", self.dispatcher.resulting_metric)


def evaluate_configuration(args: tuple) -> int:
    """Evaluate a warehouse configuration via simulation. Called from ProcessPoolExecutor.
       A fixed seed ensures reproducibility and fair comparison between genomes."""
    m, n, num_robots, entries_arr, exits_arr, saves_arr, containers_to_process, prob = args
    sim = Simulator(
        m, n, num_robots,
        len(entries_arr), len(exits_arr), len(saves_arr),
        containers_to_process, prob,
        are_position_sampled=False,
        entries_arr=entries_arr, exits_arr=exits_arr, saves_arr=saves_arr,
        rng=random.Random(0),
    )
    sim.run()
    return sim.metric
