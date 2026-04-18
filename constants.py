from enum import Enum

from matplotlib.colors import ListedColormap

# -- Warehouse grid cell types ---------------------------------------------

ROAD  = "Road"
ENTRY = "Entry"
EXIT  = "Exit"
SAVE  = "Save"

NON_ROAD   = (ENTRY, EXIT, SAVE)


# -- Object statuses --------------------------------------------------------

class ContainerStatus(str, Enum):
    """Container lifecycle status."""
    WAITING_SAVE  = "waiting to save"
    ASSIGNED_SAVE = "assigned/save"
    SAVED         = "saved"
    WAITING_EXIT  = "waiting to exit"
    ASSIGNED_EXIT = "assigned/exit"


class SaveStatus(str, Enum):
    """Storage cell status."""
    FREE       = "free"
    WAITING    = "waiting"
    LOADED_OFF = "loaded/off"
    LOADED_ON  = "loaded/on"


class RobotTask(str, Enum):
    """Type of the robot's current task."""
    TO_CASE         = "to case"
    TO_SAVE_WITH    = "to save with container"
    TO_SAVE_WITHOUT = "to save without container"
    TO_EXIT         = "to exit"
    IDLE            = "None"


# -- Visual codes for the rendering matrix ---------------------------------

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

# -- Palette for static configuration visualization ------------------------

CONFIG_CELL_TO_IDX = {ROAD: 0, ENTRY: 1, EXIT: 2, SAVE: 3}

CONFIG_COLORS = {
    ROAD:  "#E8E8E8",
    ENTRY: "#E74C3C",
    EXIT:  "#3498DB",
    SAVE:  "#2ECC71",
}

CONFIG_LABELS = {
    ROAD:  "Road",
    ENTRY: "Entry",
    EXIT:  "Exit",
    SAVE:  "Storage",
}

CONFIG_CMAP = ListedColormap([
    CONFIG_COLORS[ROAD],
    CONFIG_COLORS[ENTRY],
    CONFIG_COLORS[EXIT],
    CONFIG_COLORS[SAVE],
])

# -- Palette for the simulation animation ----------------------------------
# Base cell types (0, 2, 6, 7) are consistent with CONFIG_COLORS

CMAP = ListedColormap([
    CONFIG_COLORS[ROAD],     # 0  road
    "#2C3E50",               # 1  idle robot (dark gray)
    CONFIG_COLORS[SAVE],     # 2  storage cell
    "#27AE60",               # 3  (unused)
    "#F1C40F",               # 4  container waiting to be stored
    "#E67E22",               # 5  container stored
    CONFIG_COLORS[ENTRY],    # 6  entry cell
    CONFIG_COLORS[EXIT],     # 7  exit cell
    "#95A5A6",               # 8  robot carrying container to exit
    "#AF7AC5",               # 9  robot carrying container to storage
    "#1ABC9C",               # 10 container waiting to exit
])

ANIM_LABELS = {
    VIS_ROAD:                "Road",
    VIS_ROBOT_IDLE:          "Robot (idle)",
    VIS_SAVE:                "Storage",
    VIS_CONTAINER_WAIT_SAVE: "Container → storage",
    VIS_CONTAINER_SAVED:     "Container (stored)",
    VIS_ENTRY:               "Entry",
    VIS_EXIT:                "Exit",
    VIS_ROBOT_TO_EXIT:       "Robot → exit",
    VIS_ROBOT_TO_SAVE:       "Robot → storage",
    VIS_CONTAINER_WAIT_EXIT: "Container → exit",
}
