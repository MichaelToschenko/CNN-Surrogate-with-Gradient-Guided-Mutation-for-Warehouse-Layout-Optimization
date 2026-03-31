import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import random
from collections import deque

# ── Цветовая карта ─────────────────────────────────────────────────────────────
_CMAP = ListedColormap([
    'white',   # 0  дорога
    'black',   # 1  свободный робот
    'blue',    # 2  ячейка хранения
    'green',   # 3  (не используется)
    'yellow',  # 4  контейнер ожидает сохранения
    'orange',  # 5  контейнер сохранён в хранилище
    'red',     # 6  входная ячейка
    'purple',  # 7  выходная ячейка
    'gray',    # 8  робот везёт контейнер на выход
    'pink',    # 9  робот везёт контейнер в хранилище
    'teal',    # 10 контейнер ожидает выхода
])


class Grid:
    """Отрисовка матрицы склада m×n и обновление кадров анимации."""

    def __init__(self, m: int, n: int):
        self.m = m
        self.n = n
        self.matrix = np.zeros((m, n), dtype=int)
        self.im = None

    def update(self, robots, containers, entries, exits, saves):
        """Перерисовать все объекты для текущего кадра анимации."""
        self.matrix.fill(0)
        for pos in entries:
            self.matrix[pos] = 6
        for pos in exits:
            self.matrix[pos] = 7
        for save in saves:
            self.matrix[save.position] = 2
        for c in containers:
            if c.status in ('waiting to save', 'assigned/save'):
                self.matrix[c.position] = 4
            elif c.status in ('waiting to exit', 'assigned/exit'):
                self.matrix[c.position] = 10
            elif c.status == 'saved':
                self.matrix[c.position] = 5
        for robot in robots:
            if robot.task in ('to case', 'to save without container') or robot.is_idle():
                self.matrix[robot.position] = 1
            elif robot.task == 'to save with container':
                self.matrix[robot.position] = 9
            else:
                self.matrix[robot.position] = 8

    def draw(self):
        """Создать фигуру и imshow для анимации."""
        self.fig, self.ax = plt.subplots()
        self.ax.set_xticks(np.arange(-0.5, self.n, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.m, 1), minor=True)
        self.ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
        self.ax.tick_params(which='both', bottom=False, left=False,
                            labelbottom=False, labelleft=False)
        self.im = self.ax.imshow(self.matrix, cmap=_CMAP, vmin=0, vmax=10)

    def refresh(self):
        """Передать обновлённую матрицу в отображаемое изображение."""
        if self.im is not None:
            self.im.set_array(self.matrix)
        plt.draw()


class Robot:
    """Робот склада, следующий по заранее вычисленному пути на один шаг за такт.

    Путь хранится как список кортежей (строка, столбец); каждый вызов move()
    извлекает первый элемент и устанавливает его как новую позицию робота.
    """

    def __init__(self, position: tuple):
        self.position = position
        self.task: str = None   # допустимые значения — см. assign_task
        self.path: list = []    # [(r0,c0), (r1,c1), ...]

    def assign_task(self, path: list, task: str):
        """Назначить новый путь и тип задачи.

        Допустимые значения task:
          'to case'                   – движение к контейнеру во входной ячейке
          'to save with container'    – перевозка контейнера из входа в хранилище
          'to save without container' – перемещение к ячейке хранения без контейнера
          'to exit'                   – перевозка контейнера из хранилища на выход
          'None'                      – переходное состояние простоя
        """
        self.path = path
        self.task = task

    def is_idle(self) -> bool:
        """Вернуть True, если у робота не осталось шагов пути."""
        return not self.path

    def move(self) -> bool:
        """Сделать один шаг по пути.

        Возвращает True только в момент доставки контейнера в выходную ячейку
        (путь только что стал пустым и задача была 'to exit').
        """
        if not self.path:
            return False
        self.position = self.path.pop(0)
        if not self.path and self.task == 'to exit':
            return True
        return False


class Container:
    """Отслеживает позицию контейнера и его статус жизненного цикла.

    Допустимые значения status:
      'waiting to save'  – во входной ячейке, ожидает назначения робота
      'assigned/save'    – во входной ячейке, робот назначен (для отрисовки)
      'saved'            – находится в ячейке хранения, команды на выход нет
      'waiting to exit'  – в хранилище, команда на выход выдана
      'assigned/exit'    – в хранилище, робот назначен для вывоза
    """

    def __init__(self, position: tuple, status: str):
        self.position = position
        self.status = status


class Save:
    """Ячейка хранения и её состояние занятости.

    Допустимые значения status:
      'free'       – свободна для входящего контейнера
      'waiting'    – зарезервирована; робот с контейнером в пути
      'loaded/off' – контейнер присутствует, команды на выход нет
      'loaded/on'  – контейнер присутствует, команда на выход выдана
    """

    def __init__(self, position: tuple, status: str):
        self.position = position
        self.status = status


class Dispatcher:
    """Основная логика симуляции за один такт: генерация контейнеров, назначение задач, движение роботов."""

    def __init__(self, grid: Grid, robots, entries, exits, saves,
                 containers_to_process: int, prob: float = 0.1):
        self.grid = grid
        self.robots = robots
        self.entries = entries
        self.exits = exits
        self.saves = saves
        self.containers = []
        self.prob = prob
        self.free_exits = set(exits)
        self.containers_to_process = containers_to_process
        self.resulting_metric = 0

    # ── Публичный такт ────────────────────────────────────────────────────────

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

    # ── Генерация контейнеров ─────────────────────────────────────────────────

    def generate_containers(self):
        """Стохастически порождать контейнеры во входах и выдавать команды на выход."""
        occupied_entries = {c.position for c in self.containers}

        # выдать команды на выход для сохранённых контейнеров
        for container in self.containers:
            if container.status == 'saved' and random.random() < self.prob:
                for save in self.saves:
                    if save.position == container.position:
                        save.status = 'loaded/on'
                container.status = 'waiting to exit'

        # породить новые контейнеры в пустых входных ячейках
        for entry in self.entries:
            if entry not in occupied_entries and random.random() < self.prob:
                self.containers.append(Container(entry, 'waiting to save'))

    # ── Назначение задач ──────────────────────────────────────────────────────

    def assign_tasks(self):
        """Назначить все ожидающие задачи доступным роботам."""
        idle_robots = [
            r for r in self.robots
            if r.is_idle() and r.task not in ('to case', 'to save without container')
        ]
        robots_at_entry = [
            r for r in self.robots
            if r.is_idle() and r.task == 'to case'
        ]
        free_saves = [s for s in self.saves if s.status == 'free']
        waiting_containers = [
            c for c in self.containers
            if c.status in ('waiting to save', 'waiting to exit')
        ]

        self._dispatch_to_containers(idle_robots, robots_at_entry,
                                     free_saves, waiting_containers)
        self._handle_arrived_robots(free_saves)

    def _dispatch_to_containers(self, idle_robots, robots_at_entry,
                                 free_saves, waiting_containers):
        """Для каждого необработанного контейнера найти и назначить ближайшего свободного робота."""
        for container in waiting_containers:
            if container.status == 'waiting to save':
                # пропустить, если нет свободных ячеек хранения или все уже заняты
                if not free_saves:
                    continue
                if len(free_saves) <= len(robots_at_entry):
                    continue
            if not idle_robots:
                break

            robot, path_to_container = self._closest_robot(idle_robots, container.position)
            if robot is None:
                continue

            # если выбранный робот только что прибыл в хранилище с контейнером — сначала сдать его
            if robot.task == 'to save with container':
                robot.task = 'None'
                for save in self.saves:
                    if save.position == robot.position:
                        save.status = 'loaded/off'
                self.containers.append(Container(robot.position, 'saved'))

            if container.status == 'waiting to save':
                # переиспользуем путь, уже вычисленный при поиске ближайшего робота
                robot.assign_task(path_to_container, 'to case')
                container.status = 'assigned/save'
            else:
                # для 'to save without container' нужен include_start=True для корректной анимации
                path_with_start = self.bfs_path(robot.position, container.position,
                                                include_start=True)
                robot.assign_task(path_with_start, 'to save without container')
                container.status = 'assigned/exit'

            idle_robots.remove(robot)

    def _handle_arrived_robots(self, free_saves):
        """Обработать роботов, только что прибывших в пункт назначения."""
        loaded_on_positions = {
            s.position for s in self.saves if s.status == 'loaded/on'
        }
        for robot in self.robots:
            if robot.is_idle() and robot.task == 'to case':
                self._robot_arrived_at_entry(robot, free_saves)
            elif robot.is_idle() and robot.position in loaded_on_positions:
                self._robot_arrived_for_exit(robot)
            elif robot.is_idle() and robot.task == 'to save with container':
                self._robot_arrived_at_storage(robot)

    def _robot_arrived_at_entry(self, robot, free_saves):
        """Робот прибыл к контейнеру во входной ячейке — отправить его в хранилище."""
        save, path = self._closest_target(robot.position, free_saves,
                                          lambda s: s.position)
        if save and path:
            robot.assign_task(path, 'to save with container')
            save.status = 'waiting'
            free_saves.remove(save)

        # убрать контейнер, который робот только что забрал
        for c in self.containers:
            if c.position == robot.position:
                self.containers.remove(c)
                break

    def _robot_arrived_for_exit(self, robot):
        """Робот прибыл в ячейку хранения с командой на выход — отправить его на выход."""
        exit_list = list(self.free_exits)
        exit_pos, path = self._closest_target(robot.position, exit_list,
                                              lambda e: e)
        if exit_pos and path:
            robot.assign_task(path, 'to exit')
            for save in self.saves:
                if save.position == robot.position:
                    save.status = 'free'
                    break

        # убрать контейнер, покидающий хранилище
        for c in self.containers:
            if c.position == robot.position:
                self.containers.remove(c)
                break

    def _robot_arrived_at_storage(self, robot):
        """Робот прибыл в хранилище с контейнером — сдать контейнер."""
        robot.task = 'None'
        for save in self.saves:
            if save.position == robot.position:
                save.status = 'loaded/off'
        self.containers.append(Container(robot.position, 'saved'))

    # ── Движение роботов ──────────────────────────────────────────────────────

    def move_robots(self):
        """Сдвинуть каждого робота на один шаг; считать доставленные контейнеры."""
        for robot in self.robots:
            delivered = robot.move()
            if delivered:
                self.containers_to_process -= 1

    # ── Поиск пути ────────────────────────────────────────────────────────────

    def bfs_path(self, start: tuple, goal: tuple,
                 include_start: bool = False) -> list:
        """BFS-кратчайший путь от start до goal на текущей сетке.

        Аргументы:
            include_start: если True, возвращаемый список начинается с `start`
                           (используется для задачи 'to save without container',
                           чтобы робот визуально задерживался на один такт).

        Возвращает:
            Список кортежей (строка, столбец) или None, если путь не найден.
        """
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
                if (0 <= nx < rows and 0 <= ny < cols
                        and neighbor not in visited
                        and mat[nx][ny] != 6
                        and (mat[nx][ny] not in (4, 5, 10) or neighbor == goal)):
                    queue.append(neighbor)
                    visited.add(neighbor)
                    parent[neighbor] = current

        return None

    # ── Вспомогательные методы ────────────────────────────────────────────────

    def _closest_robot(self, candidates, target):
        """Вернуть (робот, путь) для кандидата, ближайшего к target."""
        best_robot, best_path, best_dist = None, None, float('inf')
        for robot in candidates:
            path = self.bfs_path(robot.position, target)
            if path is not None and len(path) < best_dist:
                best_dist = len(path)
                best_robot = robot
                best_path = path
        return best_robot, best_path

    def _closest_target(self, origin, targets, position_fn):
        """Вернуть (цель, путь) для цели, ближайшей к origin.

        position_fn извлекает позицию (строка, столбец) из элемента targets.
        """
        best_target, best_path, best_dist = None, None, float('inf')
        for t in targets:
            path = self.bfs_path(origin, position_fn(t))
            if path is not None and len(path) < best_dist:
                best_dist = len(path)
                best_target = t
                best_path = path
        return best_target, best_path


class Simulator:
    """Главный оркестратор: связывает сетку, роботов и диспетчер воедино."""

    def __init__(self, m: int, n: int, num_robots: int,
                 entries: int, exits: int, saves: int,
                 containers_to_process: int, prob: float,
                 are_position_sampled: bool = True, **kwargs):
        """
        Аргументы:
            m, n:                   размеры сетки
            num_robots:             количество роботов
            entries, exits, saves:  количество ячеек каждого типа
            containers_to_process:  критерий завершения симуляции
            prob:                   вероятность появления контейнера / команды на выход за такт
            are_position_sampled:   True → случайное размещение; False → использовать kwargs
            kwargs (при are_position_sampled=False):
                entries_arr, exits_arr, saves_arr: списки кортежей (строка, столбец)
        """
        self.unique_positions: set = set()
        if are_position_sampled:
            self._init_random(m, n, num_robots, entries, exits, saves)
        else:
            self._init_fixed(m, n, num_robots, entries, exits, saves, kwargs)

        self.grid = Grid(m, n)
        self.robots = [Robot(pos) for pos in self.random_positions(num_robots, m, n)]
        self.dispatcher = Dispatcher(
            self.grid, self.robots,
            self.entries, self.exits, self.saves,
            containers_to_process, prob
        )

    def _init_random(self, m, n, num_robots, entries, exits, saves):
        assert num_robots + entries + exits + saves < 0.8 * m * n, (
            'Слишком много объектов на матрице: уменьшите их кол-во или увеличьте размер матрицы'
        )
        self.entries = self.random_positions(entries, m, n)
        self.exits = self.random_positions(exits, m, n)
        self.saves = [Save(pos, 'free') for pos in self.random_positions(saves, m, n)]

    def _init_fixed(self, m, n, num_robots, entries, exits, saves, kwargs):
        entries_arr = kwargs.get('entries_arr')
        exits_arr = kwargs.get('exits_arr')
        saves_arr = kwargs.get('saves_arr')

        assert len(entries_arr) == entries, 'Несогласованная генерация входных ячеек'
        assert len(exits_arr) == exits,     'Несогласованная генерация выходных ячеек'
        assert len(saves_arr) == saves,     'Несогласованная генерация ячеек хранения'
        assert (not set(saves_arr) & set(exits_arr)
                and not set(entries_arr) & set(exits_arr)
                and not set(saves_arr) & set(entries_arr)), (
            'Позиции инициализации пересекаются'
        )
        assert num_robots + entries + exits + saves < 0.8 * m * n, (
            'Слишком много объектов на матрице: уменьшите их кол-во или увеличьте размер матрицы'
        )

        self.entries = entries_arr
        self.exits = exits_arr
        self.saves = [Save(pos, 'free') for pos in saves_arr]
        self.unique_positions = set(saves_arr) | set(entries_arr) | set(exits_arr)

    def random_positions(self, count: int, m: int, n: int) -> list:
        """Сгенерировать `count` уникальных случайных позиций (строка, столбец) на сетке."""
        positions = []
        for _ in range(count):
            pos = (random.randint(0, m - 1), random.randint(0, n - 1))
            while pos in self.unique_positions:
                pos = (random.randint(0, m - 1), random.randint(0, n - 1))
            positions.append(pos)
            self.unique_positions.add(pos)
        return positions

    def run(self):
        """Запустить симуляцию до завершения без анимации (подходит для параллельного запуска)."""
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
            self.grid.fig, self._tick, frames=1000, interval=5000, repeat=False
        )
        self.out = HTML(ani.to_jshtml())
        plt.close(self.grid.fig)
        self.metric = self.dispatcher.resulting_metric
        if self.dispatcher.containers_to_process > 0:
            print('Не хватило кол-ва кадров (frames) для обработки текущего кол-ва контейнеров')

    def _tick(self, _=None):
        """Колбэк анимации: выполнить один такт симуляции."""
        self.dispatcher.step(plot=True)

    def print_metric(self):
        """Вывести итоговое количество шагов симуляции."""
        print('Итоговое количество шагов:', self.dispatcher.resulting_metric)


def evaluate_configuration(args: tuple) -> int:
    """Функция верхнего уровня для параллельной оценки пригодности через executor.map().

    Аргументы:
        args: (m, n, num_robots, entries_arr, exits_arr, saves_arr,
               containers_to_process, prob)

    Возвращает:
        Метрику симуляции (суммарное количество шагов для обработки всех контейнеров).
    """
    m, n, num_robots, entries_arr, exits_arr, saves_arr, containers_to_process, prob = args
    sim = Simulator(
        m, n, num_robots,
        len(entries_arr), len(exits_arr), len(saves_arr),
        containers_to_process, prob,
        are_position_sampled=False,
        entries_arr=entries_arr, exits_arr=exits_arr, saves_arr=saves_arr
    )
    sim.run()
    return sim.metric
