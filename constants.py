from enum import Enum

from matplotlib.colors import ListedColormap

# -- Типы ячеек сетки склада ------------------------------------------------

ROAD  = "Road"
ENTRY = "Entry"
EXIT  = "Exit"
SAVE  = "Save"

NON_ROAD   = (ENTRY, EXIT, SAVE)


# -- Статусы объектов  ---------------

class ContainerStatus(str, Enum):
    """Статус жизненного цикла контейнера."""
    WAITING_SAVE  = "waiting to save"
    ASSIGNED_SAVE = "assigned/save"
    SAVED         = "saved"
    WAITING_EXIT  = "waiting to exit"
    ASSIGNED_EXIT = "assigned/exit"


class SaveStatus(str, Enum):
    """Статус ячейки хранения."""
    FREE       = "free"
    WAITING    = "waiting"
    LOADED_OFF = "loaded/off"
    LOADED_ON  = "loaded/on"


class RobotTask(str, Enum):
    """Тип текущей задачи робота."""
    TO_CASE         = "to case"
    TO_SAVE_WITH    = "to save with container"
    TO_SAVE_WITHOUT = "to save without container"
    TO_EXIT         = "to exit"
    IDLE            = "None"


# -- Визуальные коды для матрицы отрисовки ----------------------------------

VIS_ROAD                = 0
VIS_ROBOT_IDLE          = 1
VIS_SAVE                = 2
VIS_CONTAINER_WAIT_SAVE = 4
VIS_CONTAINER_SAVED     = 5
VIS_ENTRY               = 6
VIS_EXIT                = 7
VIS_ROBOT_TO_EXIT       = 8
VIS_ROBOT_TO_SAVE       = 9
VIS_CONTAINER_WAIT_EXIT = 10

# -- Палитра для статической визуализации конфигурации -----------------------

CONFIG_CELL_TO_IDX = {ROAD: 0, ENTRY: 1, EXIT: 2, SAVE: 3}

CONFIG_COLORS = {
    ROAD:  "#E8E8E8",   # светло-серый
    ENTRY: "#E74C3C",   # красный
    EXIT:  "#3498DB",   # синий
    SAVE:  "#2ECC71",   # зелёный
}

CONFIG_LABELS = {
    ROAD:  "Дорога",
    ENTRY: "Вход",
    EXIT:  "Выход",
    SAVE:  "Хранилище",
}

CONFIG_CMAP = ListedColormap([
    CONFIG_COLORS[ROAD],
    CONFIG_COLORS[ENTRY],
    CONFIG_COLORS[EXIT],
    CONFIG_COLORS[SAVE],
])

# -- Палитра для анимации симуляции -----------------------------------------
# Базовые типы ячеек (0, 2, 6, 7) согласованы с CONFIG_COLORS

CMAP = ListedColormap([
    CONFIG_COLORS[ROAD],   # 0  дорога
    "#2C3E50",             # 1  свободный робот (тёмно-серый)
    CONFIG_COLORS[SAVE],   # 2  ячейка хранения
    "#27AE60",             # 3  (не используется)
    "#F1C40F",             # 4  контейнер ожидает сохранения
    "#E67E22",             # 5  контейнер сохранён
    CONFIG_COLORS[ENTRY],  # 6  входная ячейка
    CONFIG_COLORS[EXIT],   # 7  выходная ячейка
    "#95A5A6",             # 8  робот везёт контейнер на выход
    "#AF7AC5",             # 9  робот везёт контейнер в хранилище
    "#1ABC9C",             # 10 контейнер ожидает выхода
])

ANIM_LABELS = {
    VIS_ROAD:                "Дорога",
    VIS_ROBOT_IDLE:          "Робот (свободен)",
    VIS_SAVE:                "Хранилище",
    VIS_CONTAINER_WAIT_SAVE: "Контейнер → хранилище",
    VIS_CONTAINER_SAVED:     "Контейнер (сохранён)",
    VIS_ENTRY:               "Вход",
    VIS_EXIT:                "Выход",
    VIS_ROBOT_TO_EXIT:       "Робот → выход",
    VIS_ROBOT_TO_SAVE:       "Робот → хранилище",
    VIS_CONTAINER_WAIT_EXIT: "Контейнер → выход",
}
